#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据准备主脚本（第 8 周实践二）。

每个数据集一个 ``prepare_*`` 函数，统一返回一份"准备结果"字典；
主流程逐个调用并**隔离异常**——任何一个数据集失败只记录、不中断整体
（作业第 9 节验收："不因单个文件失败导致整个脚本中断"）。

用法
----
    python prepare_data.py                  # 准备全部数据集
    python prepare_data.py --only mnist     # 只准备 MNIST
    python prepare_data.py --only boston,titanic
    python prepare_data.py --skip-download  # 全部用 raw/ 里的缓存，缺文件即报错
    python prepare_data.py --force          # 忽略缓存，强制重新下载并重写 CSV
    python prepare_data.py --list           # 列出支持的数据集

产物
----
    data/                    5 个规范 CSV
    results/data_report.md   本次运行的明细报告
    results/manifest.json    机器可读清单（含每个 CSV 的 SHA256，便于比对）
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import dataset_specs
import prep_utils as U
from watermelon3_data import load_watermelon3

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = BASE_DIR / "raw"
RESULT_DIR = BASE_DIR / "results"

# 数据获取日期：落盘当天，写进 README / SOURCES / manifest
FETCH_DATE = datetime.now().strftime("%Y-%m-%d")


# ==========================================================================
# 各数据集的准备函数
# ==========================================================================
def prepare_watermelon(force: bool = False, skip_download: bool = False) -> dict:
    """西瓜数据集 3.0：内嵌常量 → data/watermelon3.0.csv（UTF-8，含表头）。"""
    spec = dataset_specs.get("watermelon")
    df = load_watermelon3()

    if df.shape != spec.expected_shape:
        raise U.IntegrityError(
            f"西瓜数据集形状异常：期望 {spec.expected_shape}，实际 {df.shape}"
        )

    dest = U.write_csv_standard(
        df, DATA_DIR / spec.filename,
        has_header=spec.has_header, encoding=spec.encoding,
    )
    print(f"    写出 {dest.name}：{df.shape[0]} 行 × {df.shape[1]} 列")
    return {"path": dest, "rows": df.shape[0], "cols": df.shape[1],
            "missing": U.missing_summary(df), "df": df,
            "used_source": "教材附录原表转录（内嵌常量）"}


def prepare_boston(force: bool = False, skip_download: bool = False) -> dict:
    """波士顿房价：UCI housing.data（空白分隔、无表头）→ 506×14 CSV。

    严格按作业第 4 节：不使用 sklearn.datasets.load_boston，不删行、不改列名。
    """
    spec = dataset_specs.get("boston")
    data_urls = [spec.source_url, *spec.alt_urls]
    raw_path = RAW_DIR / "housing.data"

    if skip_download and not raw_path.exists():
        raise U.DownloadError(f"--skip-download 已指定，但缓存文件不存在：{raw_path}")

    used_url = spec.source_url
    if not skip_download:
        _, used_url = U.download_first(data_urls, raw_path, min_bytes=10_000)
        # 顺手取回官方字段说明，作为"来源可查"的证据留档（失败不影响主流程）
        try:
            U.download("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                       "housing/housing.names", RAW_DIR / "housing.names",
                       retries=1, min_bytes=1_000)
        except U.DownloadError as exc:
            print(f"    [提示] 字段说明 housing.names 未取到（不影响 CSV）：{exc}")
    elif not raw_path.exists():
        raise FileNotFoundError(f"缓存缺失：{raw_path}")

    # 空白分隔（\s+），无表头；列名由 dataset_specs 提供
    df = pd.read_csv(raw_path, sep=r"\s+", header=None,
                     names=list(spec.columns), encoding="utf-8")
    if df.shape[1] != len(spec.columns):
        raise U.IntegrityError(
            f"列数不一致：解析出 {df.shape[1]} 列，期望 {len(spec.columns)} 列"
            f"（原始文件可能不只 14 个字段）"
        )
    if df.shape != spec.expected_shape:
        raise U.IntegrityError(
            f"形状异常：期望 {spec.expected_shape}，实际 {df.shape}"
        )

    dest = U.write_csv_standard(df, DATA_DIR / spec.filename, has_header=True)
    print(f"    写出 {dest.name}：{df.shape[0]} 行 × {df.shape[1]} 列")
    return {"path": dest, "rows": df.shape[0], "cols": df.shape[1],
            "missing": U.missing_summary(df), "df": df, "used_url": used_url,
            "used_source": "UCI housing.data 原始文件（空白分隔 → CSV 列映射）"}


def prepare_titanic(force: bool = False, skip_download: bool = False) -> dict:
    """泰坦尼克号：Kaggle 训练集原始 891×12 → CSV，原始缺失原样保留。

    作业第 5 节红线：不删行、不把 Sex 编成 0/1、不裁剪列、不填充缺失。
    """
    spec = dataset_specs.get("titanic")
    raw_path = RAW_DIR / "titanic.csv"

    if skip_download and not raw_path.exists():
        raise U.DownloadError(f"--skip-download 已指定，但缓存文件不存在：{raw_path}")
    used_url = spec.source_url
    if not skip_download:
        _, used_url = U.download_first([spec.source_url], raw_path, min_bytes=40_000)

    # 该文件列名含逗号（Name）与引号，必须让 pandas 正常做引号解析
    df, enc = U.read_csv_safely(raw_path)
    print(f"    原始文件编码探测结果：{enc}")

    if tuple(df.columns) != spec.columns:
        raise U.IntegrityError(
            f"列名与原始 12 列不一致。\n      实际：{list(df.columns)}\n"
            f"      期望：{list(spec.columns)}"
        )
    if df.shape != spec.expected_shape:
        raise U.IntegrityError(
            f"形状异常：期望 {spec.expected_shape}，实际 {df.shape}（原始训练集应为 891 行）"
        )

    dest = U.write_csv_standard(df, DATA_DIR / spec.filename, has_header=True)
    miss = U.missing_summary(df)
    print(f"    写出 {dest.name}：{df.shape[0]} 行 × {df.shape[1]} 列，"
          f"缺失 {miss}")
    return {"path": dest, "rows": df.shape[0], "cols": df.shape[1],
            "missing": miss, "df": df, "used_url": used_url,
            "used_source": "Kaggle train.csv 原始文件（原样落盘，仅重写编码/换行）"}


def prepare_mnist(force: bool = False, skip_download: bool = False) -> dict:
    """MNIST：解析 IDX 二进制 → 无表头 CSV（60000×785 与 10000×785）。

    优先走 IDX 原始文件解析（作业第 3 节"工具技能"第一条）；
    四个 IDX 文件都取不到时，降级为 Keras 打包的 mnist.npz。
    """
    spec = dataset_specs.get("mnist")
    train_csv = DATA_DIR / spec.filename
    test_csv = DATA_DIR / spec.extra["filename_test"]

    images_tr = labels_tr = images_te = labels_te = None
    used_source = ""

    # 四个 IDX 原始文件都在缓存里时，即使 --skip-download 也优先走 IDX 路径：
    # download() 自身会先查缓存，命中就不会发起任何网络请求。
    raw_paths = {name: RAW_DIR / name for name in spec.raw_files}
    have_all_raw = all(p.exists() for p in raw_paths.values())

    if (not skip_download) or have_all_raw:
        try:
            paths = {}
            for name in spec.raw_files:
                if skip_download:
                    paths[name] = raw_paths[name]        # 只用缓存，绝不联网
                else:
                    paths[name] = U.download(
                        spec.source_url + name, raw_paths[name], min_bytes=1_000)
            # 用官方 MD5 确认完整性（不符只告警，不中断——镜像可能重新打包过）
            for name, expected in spec.extra["md5"].items():
                U.check_md5(paths[name], expected)
            images_tr = U.read_idx_images(paths["train-images-idx3-ubyte.gz"])
            labels_tr = U.read_idx_labels(paths["train-labels-idx1-ubyte.gz"])
            images_te = U.read_idx_images(paths["t10k-images-idx3-ubyte.gz"])
            labels_te = U.read_idx_labels(paths["t10k-labels-idx1-ubyte.gz"])
            used_source = "IDX 原始文件（S3 镜像）"
        except (U.DownloadError, U.IntegrityError) as exc:
            print(f"    [降级] IDX 路径失败：{exc}")
            print("    [降级] 改用 Keras 打包的 mnist.npz 作为可复现的替代方案")
            images_tr = None

    if images_tr is None:
        npz_path = RAW_DIR / "mnist.npz"
        if skip_download and not npz_path.exists():
            raise U.DownloadError(
                "--skip-download 已指定，但 IDX 缓存不完整且无 mnist.npz 备选"
            )
        if not npz_path.exists():
            U.download(spec.alt_urls[0], npz_path, min_bytes=1_000_000)
        with np.load(npz_path) as npz:
            images_tr, labels_tr = npz["x_train"], npz["y_train"]
            images_te, labels_te = npz["x_test"], npz["y_test"]
        used_source = "Keras mnist.npz（IDX 不可达时的降级来源）"

    # ---- 形状与取值范围硬校验（写盘前拦截，避免产出错文件）----
    for tag, imgs, lbls, expect_rows in (
        ("训练集", images_tr, labels_tr, spec.expected_shape[0]),
        ("测试集", images_te, labels_te, spec.extra["expected_shape_test"][0]),
    ):
        if imgs.shape[1:] != (28, 28):
            raise U.IntegrityError(f"MNIST {tag}图像尺寸应为 28×28，实际 {imgs.shape[1:]}")
        if imgs.shape[0] != expect_rows or lbls.shape[0] != expect_rows:
            raise U.IntegrityError(
                f"MNIST {tag}样本数异常：图像 {imgs.shape[0]}，标签 {lbls.shape[0]}，"
                f"期望 {expect_rows}"
            )
        if lbls.min() < 0 or lbls.max() > 9:
            raise U.IntegrityError(
                f"MNIST {tag}标签越界：[{lbls.min()}, {lbls.max()}]，应为 0–9"
            )
        if imgs.max() > 255:
            raise U.IntegrityError(f"MNIST {tag}像素越界：最大 {imgs.max()}，应 ≤ 255")

    _write_mnist_csv(train_csv, labels_tr, images_tr)
    _write_mnist_csv(test_csv, labels_te, images_te)

    n_tr, n_te = labels_tr.shape[0], labels_te.shape[0]
    print(f"    写出 {train_csv.name}：{n_tr} 行 × 785 列（无表头）")
    print(f"    写出 {test_csv.name}：{n_te} 行 × 785 列（无表头）")
    print(f"    标签分布 训练 {np.bincount(labels_tr, minlength=10).tolist()}")
    print(f"               测试 {np.bincount(labels_te, minlength=10).tolist()}")

    # 用很小的代价重建一个"预览用"的 df（整表 60000×785 转 DataFrame 要 ~380 MB，没必要）
    return {
        "path": train_csv, "rows": n_tr, "cols": 785,
        "missing": {}, "used_source": used_source,
        "extra_paths": [test_csv],
        "extra_shapes": {test_csv.name: (n_te, 785)},
        "label_counts": {
            "train": np.bincount(labels_tr, minlength=10).tolist(),
            "test": np.bincount(labels_te, minlength=10).tolist(),
        },
        "pixel_range": [int(images_tr.min()), int(images_tr.max())],
    }


def _write_mnist_csv(dest: Path, labels, images, chunk: int = 5000) -> Path:
    """分块写 MNIST CSV，避免 60000×785 的 int64 矩阵把内存吃满。

    不用 ``np.savetxt``（Python 层逐元素格式化，慢）；不用一次性
    ``DataFrame.to_csv``（会为 47M 个元素分配 int64，约 380 MB）。
    这里每 5000 行建一个临时 DataFrame，写进 StringIO 再落盘。
    """
    U.ensure_dir(dest.parent)
    n = labels.shape[0]
    flat = images.reshape(n, -1)
    tmp = dest.with_suffix(".csv.tmp")

    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            block = np.empty((stop - start, 785), dtype=np.uint8)
            block[:, 0] = labels[start:stop]
            block[:, 1:] = flat[start:stop]
            buf = io.StringIO()
            pd.DataFrame(block).to_csv(
                buf, header=False, index=False, sep=",", lineterminator="\n"
            )
            fh.write(buf.getvalue())

    import os
    os.replace(tmp, dest)
    return dest


# ==========================================================================
# 编排
# ==========================================================================
PREPARERS = {
    "watermelon": ("西瓜数据集 3.0", prepare_watermelon),
    "boston": ("波士顿房价", prepare_boston),
    "titanic": ("泰坦尼克号", prepare_titanic),
    "mnist": ("MNIST", prepare_mnist),
}


def run(keys: list[str], *, force: bool = False, skip_download: bool = False) -> int:
    """依次准备 ``keys`` 中的数据集，返回失败个数。"""
    U.ensure_dir(DATA_DIR)
    U.ensure_dir(RAW_DIR)
    U.ensure_dir(RESULT_DIR)

    if force:
        for key in keys:
            spec = dataset_specs.get(key)
            targets = [DATA_DIR / spec.filename]
            if key == "mnist":
                targets.append(DATA_DIR / spec.extra["filename_test"])
            for t in targets:
                t.unlink(missing_ok=True)
        print("[--force] 已清除目标 CSV，将重新生成\n")

    print(f"数据获取日期：{FETCH_DATE}")
    print(f"输出目录：{DATA_DIR}\n")

    results: list[dict] = []
    for key in keys:
        display, fn = PREPARERS[key]
        spec = dataset_specs.get(key)
        print(f"[{key}] {display}")
        print(f"    来源：{spec.source_name}")
        started = datetime.now()
        try:
            info = fn(force=force, skip_download=skip_download)
            rows, cols = info["rows"], info["cols"]
            # 产物校验和（MNIST 只对训练集算，测试集另记）
            digest = U.sha256_of(info["path"])
            size = info["path"].stat().st_size
            extra = {}
            for p in info.get("extra_paths", []):
                extra[p.name] = {
                    "sha256": U.sha256_of(p),
                    "size_bytes": p.stat().st_size,
                    "shape": list(info.get("extra_shapes", {}).get(p.name, ())),
                }
            results.append({
                "key": key, "display_name": display, "status": "ok",
                "filename": spec.filename, "rows": rows, "cols": cols,
                "shape": [rows, cols],
                "missing": info.get("missing", {}),
                "missing_total": int(sum(info.get("missing", {}).values())),
                "sha256": digest, "size_bytes": size, "extra_files": extra,
                "source_name": spec.source_name, "source_url": spec.source_url,
                "used_url": info.get("used_url", spec.source_url),
                "used_source": info.get("used_source", "原始下载"),
                "version": spec.version,
                "fetch_date": FETCH_DATE,
                "elapsed_s": round((datetime.now() - started).total_seconds(), 2),
                "label_counts": info.get("label_counts"),
                "pixel_range": info.get("pixel_range"),
            })
            print(f"    [成功] {U.human_size(size)}，耗时 "
                  f"{results[-1]['elapsed_s']}s\n")
        except Exception as exc:                      # noqa: BLE001 —— 故意兜住一切
            # 单个数据集失败：记录完整堆栈，继续下一个（第 9 节验收要求）
            print(f"    [失败] {type(exc).__name__}: {exc}")
            print("    后续数据集继续执行，不影响整体流程\n")
            results.append({
                "key": key, "display_name": display, "status": "failed",
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(limit=3),
                "source_name": spec.source_name, "source_url": spec.source_url,
                "version": spec.version, "fetch_date": FETCH_DATE,
                "elapsed_s": round((datetime.now() - started).total_seconds(), 2),
            })

    ok = sum(1 for r in results if r["status"] == "ok")
    failed = len(results) - ok
    print("=" * 72)
    print(f"完成：成功 {ok} 个，失败 {failed} 个")
    for r in results:
        if r["status"] == "ok":
            print(f"  ✓ {r['filename']:<22} {r['rows']:>6} × {r['cols']:<4} "
                  f"缺失 {r['missing_total']:<4} {U.human_size(r['size_bytes'])}")
        else:
            print(f"  ✗ {r['display_name']:<22} {r['error_type']}: {r['error']}")
    print("=" * 72)

    _write_reports(results)
    print(f"报告：{RESULT_DIR / 'data_report.md'}")
    print(f"清单：{RESULT_DIR / 'manifest.json'}")
    return failed


def _write_reports(results: list[dict]) -> None:
    """写 results/data_report.md 与 results/manifest.json。"""
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fetch_date": FETCH_DATE,
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "datasets": results,
    }
    (RESULT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 数据准备运行报告",
        "",
        f"- 生成时间：{manifest['generated_at']}",
        f"- 数据获取日期：{FETCH_DATE}",
        f"- 运行环境：Python {manifest['python']} / pandas {manifest['pandas']} / "
        f"numpy {manifest['numpy']}",
        "",
        "## 总览",
        "",
        "| 数据集 | 文件 | 行 | 列 | 缺失单元格 | 大小 | 状态 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r["status"] == "ok":
            lines.append(
                f"| {r['display_name']} | `{r['filename']}` | {r['rows']} | {r['cols']} "
                f"| {r['missing_total']} | {U.human_size(r['size_bytes'])} | 成功 |"
            )
        else:
            lines.append(
                f"| {r['display_name']} | — | — | — | — | — | "
                f"失败（{r['error_type']}） |"
            )

    lines += ["", "## 明细", ""]
    for r in results:
        lines.append(f"### {r['display_name']}（`{r['key']}`）")
        lines.append("")
        if r["status"] != "ok":
            lines.append(f"- **状态**：失败")
            lines.append(f"- **错误类型**：`{r['error_type']}`")
            lines.append(f"- **错误信息**：{r['error']}")
            lines.append("")
            continue
        lines.append(f"- 来源：{r['source_name']}")
        lines.append(f"- 实际使用地址：`{r['used_url']}`")
        lines.append(f"- 版本：{r['version']}")
        lines.append(f"- 获取日期：{r['fetch_date']}")
        lines.append(f"- 形状：{r['rows']} 行 × {r['cols']} 列")
        if r["missing"]:
            miss = "、".join(f"{k} {v}" for k, v in r["missing"].items())
            lines.append(f"- 缺失值：{miss}（共 {r['missing_total']} 个单元格，"
                         f"保留为空字段）")
        else:
            lines.append("- 缺失值：无")
        if r.get("label_counts"):
            lines.append(f"- 标签分布：训练 {r['label_counts']['train']}")
            lines.append(f"　　　　　　测试 {r['label_counts']['test']}")
        if r.get("pixel_range"):
            lines.append(f"- 像素取值范围：{r['pixel_range'][0]}–{r['pixel_range'][1]}"
                         f"（未归一化）")
        lines.append(f"- SHA256：`{r['sha256']}`")
        for name, meta in r.get("extra_files", {}).items():
            lines.append(f"- 附加文件 `{name}`：形状 {meta['shape']}，"
                         f"SHA256 `{meta['sha256']}`")
        lines.append(f"- 耗时：{r['elapsed_s']} s")
        lines.append("")

    (RESULT_DIR / "data_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="第 8 周实践二：数据集收集与准备",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n  python prepare_data.py\n"
               "  python prepare_data.py --only mnist --force\n"
               "  python prepare_data.py --skip-download",
    )
    parser.add_argument("--only", default="",
                        help=f"只准备指定数据集，逗号分隔。可选：{', '.join(PREPARERS)}")
    parser.add_argument("--force", action="store_true",
                        help="忽略已有 CSV，强制重新生成")
    parser.add_argument("--skip-download", action="store_true",
                        help="只用 raw/ 缓存，不发起任何网络请求")
    parser.add_argument("--list", action="store_true", help="列出支持的数据集后退出")
    args = parser.parse_args(argv)

    if args.list:
        print("支持的数据集：")
        for key, (display, _) in PREPARERS.items():
            spec = dataset_specs.get(key)
            print(f"  {key:<12} {display:<16} {spec.filename:<22} "
                  f"期望形状 {spec.expected_shape}")
        return 0

    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        unknown = [k for k in keys if k not in PREPARERS]
        if unknown:
            print(f"错误：未知数据集 {unknown}，可选：{', '.join(PREPARERS)}",
                  file=sys.stderr)
            return 2
    else:
        keys = list(PREPARERS)

    failed = run(keys, force=args.force, skip_download=args.skip_download)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
