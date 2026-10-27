"""数据准备的通用工具：下载、校验和、编码探测、规范 CSV 写出。

设计原则（对应作业第 9 节"异常情况处理"）：

1. **每个失败点都有明确的异常类型和中文提示**，不抛裸 Exception；
2. **下载带重试与退避**，中转/镜像抖动不应直接判死；
3. **原子写入**：先写 ``*.tmp`` 再 ``os.replace``，避免中断后留下半截 CSV；
4. **编码兜底**：读 CSV 按 utf-8 → utf-8-sig → gbk 顺序尝试，全失败才报错；
5. **可复现**：所有写出都显式指定 ``encoding`` / ``index=False`` / ``na_rep``，
   随机性为零（本任务不涉及随机采样）。
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

# 统一 User-Agent：部分镜像会拒绝 Python-urllib 默认 UA
USER_AGENT = "Mozilla/5.0 (compatible; DataPrepHomework/1.0; +https://example.invalid)"

# 读取时按顺序尝试的编码（第 9 节：编码错误处理）
CANDIDATE_ENCODINGS = ("utf-8", "utf-8-sig", "gbk", "gb18030")


class DataPrepError(RuntimeError):
    """本模块所有可预期错误的基类，便于上层统一捕获并继续处理其它数据集。"""


class DownloadError(DataPrepError):
    """下载失败（网络不可达、HTTP 非 200、重试耗尽）。"""


class IntegrityError(DataPrepError):
    """文件完整性校验失败（校验和不符、魔数错误、行列数不符）。"""


class EncodingDetectionError(DataPrepError):
    """所有候选编码均无法解码。"""


# ==========================================================================
# 路径
# ==========================================================================
def ensure_dir(path: Path) -> Path:
    """确保目录存在（幂等）。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def human_size(num_bytes: int) -> str:
    """字节数转人类可读字符串，用于日志。"""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


# ==========================================================================
# 下载
# ==========================================================================
def download(url: str, dest: Path, *, retries: int = 3, timeout: int = 60,
             min_bytes: int = 64) -> Path:
    """把 ``url`` 下载到 ``dest``；``dest`` 已存在且非空时直接复用（缓存）。

    Parameters
    ----------
    retries : 失败重试次数（指数退避 2s / 4s / 8s）
    min_bytes : 小于该字节数视作下载不完整，触发重试
    """
    dest = Path(dest)
    ensure_dir(dest.parent)

    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"    [缓存] {dest.name}（{human_size(dest.stat().st_size)}）")
        return dest

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    raise DownloadError(f"HTTP {resp.status}")
                with open(tmp, "wb") as fh:
                    shutil.copyfileobj(resp, fh, length=1 << 20)
            size = tmp.stat().st_size
            if size < min_bytes:
                tmp.unlink(missing_ok=True)
                raise DownloadError(
                    f"下载内容过小（{size} 字节，可能被重定向到错误页）"
                )
            os.replace(tmp, dest)          # 原子替换，杜绝半截文件
            print(f"    [下载] {dest.name}（{human_size(size)}）")
            return dest
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                DownloadError) as exc:
            last_err = exc
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                wait = 2 ** attempt
                print(f"    [重试 {attempt}/{retries - 1}] {url} -> {exc}，{wait}s 后重试")
                time.sleep(wait)
    raise DownloadError(f"下载失败（已重试 {retries} 次）：{url}\n    最后错误：{last_err}")


def download_first(urls, dest: Path, **kwargs) -> tuple[Path, str]:
    """按顺序尝试多个地址，返回 ``(本地路径, 命中的 URL)``。全部失败则抛 DownloadError。"""
    errors = []
    for url in urls:
        if not url:
            continue
        try:
            return download(url, dest, **kwargs), url
        except DownloadError as exc:
            errors.append(f"{url} -> {exc}")
    raise DownloadError("所有候选地址均不可用：\n    " + "\n    ".join(errors))


# ==========================================================================
# 校验和
# ==========================================================================
def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """分块计算 SHA256，避免大文件（MNIST 训练集约 120 MB）一次性读入内存。"""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def check_md5(path: Path, expected: str, *, strict: bool = False) -> bool:
    """比对 MD5。``strict=False`` 时仅警告不中断（镜像可能重新压缩过）。"""
    actual = md5_of(path)
    ok = actual.lower() == expected.lower()
    if not ok:
        msg = f"{path.name} MD5 不符：期望 {expected}，实际 {actual}"
        if strict:
            raise IntegrityError(msg)
        print(f"    [警告] {msg}（视为镜像重打包，继续）")
    return ok


# ==========================================================================
# 编码与读取
# ==========================================================================
def detect_encoding(path: Path) -> str:
    """按 ``CANDIDATE_ENCODINGS`` 顺序探测可解码编码，全部失败抛 EncodingDetectionError。"""
    raw = Path(path).read_bytes()[:1 << 20]   # 取前 1 MB 足够判定
    for enc in CANDIDATE_ENCODINGS:
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    raise EncodingDetectionError(
        f"{path.name} 无法用以下任一编码解码：{', '.join(CANDIDATE_ENCODINGS)}"
    )


def read_csv_safely(path: Path, **kwargs) -> tuple[pd.DataFrame, str]:
    """读 CSV，自动尝试多种编码。返回 ``(DataFrame, 实际使用的编码)``。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    if path.stat().st_size == 0:
        raise DataPrepError(f"文件为空（0 字节）：{path}")
    enc = detect_encoding(path)
    df = pd.read_csv(path, encoding=enc, **kwargs)
    if df.empty:
        raise DataPrepError(f"文件解析后无数据行：{path}")
    return df, enc


# ==========================================================================
# 规范写出：作业第 2 节的全部约束集中在这里
# ==========================================================================
def write_csv_standard(df: pd.DataFrame, dest: Path, *, has_header: bool = True,
                       encoding: str = "utf-8") -> Path:
    """按作业第 2 节规范写出 CSV。

    - UTF-8 编码（无 BOM，保证 ``pd.read_csv`` 默认参数即可直读、列名不带 \\ufeff）
    - 英文逗号分隔
    - 表头可选（MNIST 为 False）
    - ``index=False``：不保存行索引
    - ``na_rep=""``：缺失值保留为空字段，不写 "NaN"/"nan"
    - 不改列名、不加单位与千分位（原样传递）
    - ``quoting=csv.QUOTE_MINIMAL``（值为 0）：只给"含分隔符/引号/换行"的字段
      加引号。注意 **QUOTE_MINIMAL 是 0 而不是 1**，写 1 会变成 QUOTE_ALL，
      把每个字段都包上引号（文件体积膨胀，且不利于人工核对）
    - 原子写入：先写 .tmp 再 replace
    """
    dest = Path(dest)
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        df.to_csv(
            tmp,
            sep=",",
            header=has_header,
            index=False,
            na_rep="",
            encoding=encoding,
            lineterminator="\n",   # 统一换行符，跨平台可复现
            quoting=csv.QUOTE_MINIMAL,  # 仅必要字段加引号
        )
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


# ==========================================================================
# IDX 格式解析（MNIST）
#
# IDX 头部结构（全部大端序）：
#     byte 0-1 : 恒为 0x0000
#     byte 2   : 数据类型 (0x08 = unsigned byte)
#     byte 3   : 维数 ndim  ← 注意：维数藏在"魔数"的第 4 个字节里，
#                并没有独立的 4 字节维数段（这里踩过坑，见 README"遇到的问题"）
#     byte 4.. : ndim 个 4 字节无符号整数，依次是各维长度
# 因此整体读 4 字节恰好等于 0x00000803（图像，ndim=3）/ 0x00000801（标签，ndim=1），
# 之后**紧接**就是维度长度，不能再多读一次 ndim。
# ==========================================================================
IDX_MAGIC_IMAGES = 0x00000803   # 图像：dtype=0x08, ndim=3
IDX_MAGIC_LABELS = 0x00000801   # 标签：dtype=0x08, ndim=1


def _read_idx_header(fh, path: Path, expected_magic: int, expected_ndim: int):
    """读取 IDX 头部，返回各维长度列表。魔数或维数不符时抛 IntegrityError。"""
    magic = int.from_bytes(fh.read(4), "big")
    if magic != expected_magic:
        raise IntegrityError(
            f"{path.name} 魔数错误：期望 0x{expected_magic:08X}，"
            f"实际 0x{magic:08X}（可能不是 IDX{expected_ndim} 文件）"
        )
    ndim = magic & 0xFF                      # 维数在魔数最低字节
    if ndim != expected_ndim:
        raise IntegrityError(
            f"{path.name} 维数应为 {expected_ndim}，实际 {ndim}"
        )
    dims = [int.from_bytes(fh.read(4), "big") for _ in range(ndim)]
    if any(d <= 0 for d in dims):
        raise IntegrityError(f"{path.name} 维度长度非法：{dims}")
    return dims


def read_idx_images(path: Path):
    """解析 IDX 图像文件，返回 ``np.ndarray``，形状 ``(n, rows, cols)``，dtype uint8。"""
    import numpy as np

    path = Path(path)
    with gzip.open(path, "rb") as fh:
        dims = _read_idx_header(fh, path, IDX_MAGIC_IMAGES, 3)
        payload = fh.read()

    expected = dims[0] * dims[1] * dims[2]
    if len(payload) != expected:
        raise IntegrityError(
            f"{path.name} 数据长度不符：期望 {expected} 字节"
            f"（{dims[0]}×{dims[1]}×{dims[2]}），实际 {len(payload)} 字节"
        )
    return np.frombuffer(payload, dtype=np.uint8).reshape(dims)


def read_idx_labels(path: Path):
    """解析 IDX 标签文件，返回 ``np.ndarray``，形状 ``(n,)``，dtype uint8。"""
    import numpy as np

    path = Path(path)
    with gzip.open(path, "rb") as fh:
        dims = _read_idx_header(fh, path, IDX_MAGIC_LABELS, 1)
        payload = fh.read()

    if len(payload) != dims[0]:
        raise IntegrityError(
            f"{path.name} 标签数不符：期望 {dims[0]}，实际 {len(payload)}"
        )
    return np.frombuffer(payload, dtype=np.uint8)


# ==========================================================================
# 报告
# ==========================================================================
def missing_summary(df: pd.DataFrame) -> dict[str, int]:
    """返回"缺失数 > 0"的列及其缺失数；全无缺失时返回空字典。"""
    counts = df.isna().sum()
    return {col: int(n) for col, n in counts.items() if n > 0}
