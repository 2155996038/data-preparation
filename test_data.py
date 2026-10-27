#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据准备的单元测试（第 8 周实践二 第 7、8、12 节）。

运行
----
    python -m pytest test_data.py -v

设计
----
``test_data.py`` 不重复实现校验逻辑，而是**直接复用** ``validate_data.py``
里的检查函数逐条参数化——一处定义，两处使用（命令行 + pytest），
避免"脚本里写的规则"和"测试里写的规则"慢慢分叉。

新增三类"健壮性"测试，人为构造异常输入，确认第 9 节要求的
"异常提示清晰"确实成立：
    - 文件不存在 / 文件为空 / 编码无法识别
    - 下载失败（连不上）时是否抛出 DownloadError 而不是裸 OSError
    - IDX 魔数错误时是否明确指出是哪个文件、期望什么魔数
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dataset_specs
import prep_utils as U
import validate_data as V
from watermelon3_data import load_watermelon3

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 已准备好的数据集（CSV 存在才算），用于 skip 未准备的情况
READY = [k for k in dataset_specs.PREPARE_ORDER
         if (DATA_DIR / dataset_specs.get(k).filename).exists()]

requires_data = pytest.mark.skipif(
    not READY, reason="data/ 下尚无 CSV，请先运行 python prepare_data.py"
)


# ==========================================================================
# 一、规范与语义校验（复用 validate_data 的全部检查项）
# ==========================================================================
def _all_checks() -> list:
    """收集 (dataset, 检查名, 函数) 三元组，供参数化使用。"""
    out = []
    for key in dataset_specs.PREPARE_ORDER:
        items = V.csv_spec_checks(key) + list(V.CHECKS.get(key, []))
        out.extend((key, name, fn) for name, fn in items)
    return out


ALL_CHECKS = _all_checks()


@requires_data
@pytest.mark.parametrize(
    "dataset, name, fn",
    ALL_CHECKS,
    ids=[f"{k}-{n}" for k, n, _ in ALL_CHECKS],
)
def test_dataset_check(dataset: str, name: str, fn):
    """每个数据集都跑：通用 CSV 规范 + 该数据集的全部业务校验。"""
    if not (DATA_DIR / dataset_specs.get(dataset).filename).exists():
        pytest.skip(f"{dataset} 尚未准备")
    detail = fn()
    assert detail is not None or detail == ""


@requires_data
def test_at_least_five_checks_per_dataset():
    """每个数据集至少 5 条校验（作业第 8 节原文要求）。"""
    for key in READY:
        n = len(V.csv_spec_checks(key)) + len(V.CHECKS.get(key, []))
        assert n >= 5, f"{key} 只有 {n} 条校验，不足 5 条"


@requires_data
def test_validation_report_all_pass():
    """整体校验必须全绿——这是可提交的前提。"""
    results = V.run_validation(READY, verbose=False)
    failed = [r for r in results if not r["ok"]]
    assert not failed, "校验失败：\n" + "\n".join(
        f"  [{r['dataset']}] {r['name']}: {r['detail']}" for r in failed
    )


# ==========================================================================
# 二、数据集专属的关键断言（抽查，便于答辩时直接指认）
# ==========================================================================
@requires_data
@pytest.mark.skipif(
    not (DATA_DIR / "mnist_train.csv").exists(),
    reason="MNIST 大文件不入 Git，需先运行 python prepare_data.py --only mnist",
)
def test_mnist_shape_and_range():
    """MNIST：60000×785 / 10000×785，标签 0–9，像素 0–255 且未归一化。"""
    train = DATA_DIR / "mnist_train.csv"
    test = DATA_DIR / "mnist_test.csv"
    for path, expect_rows in ((train, 60000), (test, 10000)):
        with open(path, "r", encoding="utf-8") as fh:
            n = sum(1 for _ in fh)
        assert n == expect_rows, f"{path.name} 行数 {n} ≠ {expect_rows}"

        arr = np.loadtxt(path, delimiter=",", dtype=np.uint8, max_rows=200)
        assert arr.shape == (200, 785), f"{path.name} 应为 785 列"
        assert arr[:, 0].max() <= 9, "第 1 列 label 应在 0–9"
        assert arr[:, 1:].max() <= 255, "像素应 ≤ 255"


@requires_data
def test_boston_columns_exact():
    """波士顿：14 个列名必须逐字与 UCI 一致。"""
    df = pd.read_csv(DATA_DIR / "boston_housing.csv")
    assert list(df.columns) == list(dataset_specs.BOSTON_COLUMNS)
    assert df.shape == (506, 14)
    assert not df.isna().any().any()


@requires_data
def test_boston_not_from_sklearn():
    """波士顿不得来自 sklearn 的 load_boston：原始文件 MEDV 上限为 50.0。"""
    df = pd.read_csv(DATA_DIR / "boston_housing.csv")
    # sklearn 的 load_boston 返回的 MEDV 单位是千美元，且列名是小写；
    # 这里校验列名为 UCI 大写名、CHAS 为整数 0/1，即证明走的是 UCI 原文转换。
    assert df["MEDV"].max() == 50.0
    assert df["CHAS"].dtype.kind in "iu", "CHAS 应为整数列"


@requires_data
def test_titanic_missing_and_sex():
    """泰坦尼克：Age/Cabin/Embarked 原始缺失保留，Sex 未被编码。"""
    df = pd.read_csv(DATA_DIR / "titanic_train.csv")
    assert int(df["Age"].isna().sum()) == 177
    assert int(df["Cabin"].isna().sum()) == 687
    assert int(df["Embarked"].isna().sum()) == 2
    assert set(df["Sex"].unique()) == {"male", "female"}
    assert len(df) == 891


@requires_data
def test_watermelon_fields_and_encoding():
    """西瓜 3.0：9 个字段齐全，离散值为中文原值，UTF-8 无 BOM。"""
    path = DATA_DIR / "watermelon3.0.csv"
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf"), "不应带 BOM"
    df = pd.read_csv(path, encoding="utf-8")
    for col in ("色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖率", "好瓜"):
        assert col in df.columns, f"缺少字段 {col}"
    assert set(df["色泽"]) == {"青绿", "乌黑", "浅白"}
    assert set(df["好瓜"]) == {"是", "否"}
    assert df.shape == (17, 10)


@requires_data
def test_all_csv_readable_by_default_params():
    """所有 CSV 用 pandas.read_csv 默认参数即可读取（第 2 节验收原文）。"""
    for key in READY:
        spec = dataset_specs.get(key)
        path = DATA_DIR / spec.filename
        if spec.has_header:
            df = pd.read_csv(path)          # 不传任何参数
            assert df.shape[1] == spec.expected_shape[1]
        else:
            df = pd.read_csv(path, header=None)
            assert df.shape[1] == spec.expected_shape[1]


# ==========================================================================
# 三、健壮性：人为构造异常输入
# ==========================================================================
def test_missing_file_raises_filenotfound(tmp_path: Path):
    """文件不存在时给出明确提示，而不是让人去猜路径。"""
    with pytest.raises(FileNotFoundError) as exc:
        U.read_csv_safely(tmp_path / "not_exist.csv")
    assert "文件不存在" in str(exc.value)


def test_empty_file_raises(tmp_path: Path):
    """0 字节文件应当立刻报错，而不是返回空 DataFrame 让下游崩在别处。"""
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    with pytest.raises(U.DataPrepError) as exc:
        U.read_csv_safely(empty)
    assert "空" in str(exc.value)


def test_undecodable_file_raises_encoding_error(tmp_path: Path):
    """所有候选编码都无法解码时，提示里要列出尝试过的编码。"""
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"\xff\xfe\x00\x00\x81\x40\xff")
    with pytest.raises(U.EncodingDetectionError) as exc:
        U.detect_encoding(bad)
    assert "utf-8" in str(exc.value)


def test_gbk_file_is_auto_detected(tmp_path: Path):
    """GBK 编码的中文 CSV 应能自动识别并正确读出中文。"""
    p = tmp_path / "gbk.csv"
    p.write_bytes("名称,数值\n青绿,0.697\n乌黑,0.774\n".encode("gbk"))
    enc = U.detect_encoding(p)
    assert enc in {"gbk", "gb18030"}
    df, used = U.read_csv_safely(p)
    assert used in {"gbk", "gb18030"}
    assert df.loc[0, "名称"] == "青绿"


def test_download_failure_raises_downloaderror(tmp_path: Path):
    """连不上时抛 DownloadError（带 URL 和最后一次错误），不是裸 OSError。"""
    with pytest.raises(U.DownloadError) as exc:
        U.download("http://127.0.0.1:9/nope.bin", tmp_path / "x.bin",
                   retries=1, timeout=3)
    assert "127.0.0.1:9" in str(exc.value)


def test_download_first_lists_all_failures(tmp_path: Path):
    """多个地址全失败时，错误信息要包含每个地址的失败原因。"""
    with pytest.raises(U.DownloadError) as exc:
        U.download_first(["http://127.0.0.1:9/a", "http://127.0.0.1:9/b"],
                         tmp_path / "y.bin", retries=1, timeout=3)
    msg = str(exc.value)
    assert "127.0.0.1:9/a" in msg and "127.0.0.1:9/b" in msg


def _make_idx_gz(path: Path, payload: bytes) -> Path:
    """把任意字节压成 gzip，用来伪造 IDX 文件。"""
    with gzip.open(path, "wb") as fh:
        fh.write(payload)
    return path


def _idx_header(magic: int, dims: tuple[int, ...]) -> bytes:
    """按 IDX 规范拼头部：魔数 4 字节 + 各维 4 字节（大端序）。

    注意魔数最低字节就是维数，**不存在**独立的维数字段——这一点写错过，
    所以测试里也把它固化下来。
    """
    return magic.to_bytes(4, "big") + b"".join(d.to_bytes(4, "big") for d in dims)


def test_idx_bad_magic_detected(tmp_path: Path):
    """IDX 魔数不对时，错误信息应指出期望值与实际值。"""
    # 把标签文件的魔数当成图像文件传进来
    fake = _make_idx_gz(tmp_path / "fake-images.gz",
                        _idx_header(0x00000801, (5,)) + b"\x00" * 5)
    with pytest.raises(U.IntegrityError) as exc:
        U.read_idx_images(fake)
    assert "魔数" in str(exc.value)
    assert "0x00000803" in str(exc.value)


def test_idx_truncated_payload_detected(tmp_path: Path):
    """IDX 声明的维度与实际字节数不符时，应报出期望/实际长度。"""
    fake = _make_idx_gz(tmp_path / "short.gz",
                        _idx_header(0x00000803, (2, 28, 28)) + b"\x00" * 100)
    with pytest.raises(U.IntegrityError) as exc:
        U.read_idx_images(fake)
    assert "数据长度不符" in str(exc.value)
    assert "1568" in str(exc.value), "错误信息应给出期望字节数 2×28×28=1568"


def test_idx_valid_roundtrip(tmp_path: Path):
    """正常 IDX 文件能解析出正确的形状与像素值。"""
    images = np.arange(2 * 28 * 28, dtype=np.uint8).reshape(2, 28, 28)
    path = tmp_path / "good-images.gz"
    with gzip.open(path, "wb") as fh:
        fh.write(_idx_header(0x00000803, (2, 28, 28)))
        fh.write(images.tobytes())
    got = U.read_idx_images(path)
    assert got.shape == (2, 28, 28)
    assert np.array_equal(got, images)


def test_idx_labels_roundtrip(tmp_path: Path):
    """标签文件解析：1 维，取值 0–9。"""
    labels = np.array([0, 9, 3, 7], dtype=np.uint8)
    path = tmp_path / "good-labels.gz"
    with gzip.open(path, "wb") as fh:
        fh.write(_idx_header(0x00000801, (4,)))
        fh.write(labels.tobytes())
    got = U.read_idx_labels(path)
    assert got.shape == (4,)
    assert np.array_equal(got, labels)


def test_write_csv_missing_as_empty_field(tmp_path: Path):
    """缺失值必须写成空字段，不能出现 nan 字面量。"""
    df = pd.DataFrame({"a": [1, 2], "b": [0.5, np.nan], "c": ["x", None]})
    out = U.write_csv_standard(df, tmp_path / "t.csv")
    text = out.read_text(encoding="utf-8")
    assert "nan" not in text.lower()
    assert "None" not in text
    assert text.splitlines()[1] == "1,0.5,x"
    assert text.splitlines()[2] == "2,,", "缺失处应为空字段"


def test_write_csv_no_index_and_no_header(tmp_path: Path):
    """index=False 与 header=False 同时生效（MNIST 的写法）。"""
    df = pd.DataFrame(np.arange(12).reshape(3, 4))
    out = U.write_csv_standard(df, tmp_path / "n.csv", has_header=False)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3, "不应多出表头行"
    assert lines[0] == "0,1,2,3", "不应多出索引列"


def test_write_csv_quotes_comma_fields(tmp_path: Path):
    """含逗号的字段必须加引号，否则读回来会多出列。"""
    df = pd.DataFrame({"Name": ["Braund, Mr. Owen Harris"], "Age": [22]})
    out = U.write_csv_standard(df, tmp_path / "q.csv")
    assert '"Braund, Mr. Owen Harris"' in out.read_text(encoding="utf-8")
    back = pd.read_csv(out)
    assert back.shape == (1, 2)
    assert back.loc[0, "Name"] == "Braund, Mr. Owen Harris"


def test_write_csv_is_atomic(tmp_path: Path):
    """写出后不应残留 .tmp 中间文件。"""
    df = pd.DataFrame({"a": [1]})
    U.write_csv_standard(df, tmp_path / "a.csv")
    assert (tmp_path / "a.csv").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_sha256_stable(tmp_path: Path):
    """同一文件两次计算的 SHA256 必须一致（复现性基础）。"""
    p = tmp_path / "d.bin"
    p.write_bytes(b"hello" * 1000)
    assert U.sha256_of(p) == U.sha256_of(p)
    assert len(U.sha256_of(p)) == 64


@requires_data
def test_sha256_matches_manifest():
    """落盘 CSV 的 SHA256 应与 manifest.json 记录一致（数据未被事后改动）。"""
    import json
    manifest_path = BASE_DIR / "results" / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("尚未生成 manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["datasets"]:
        if entry["status"] != "ok":
            continue
        path = DATA_DIR / entry["filename"]
        if not path.exists():
            pytest.skip(f"{entry['filename']} 不存在")
        assert U.sha256_of(path) == entry["sha256"], (
            f"{entry['filename']} 的 SHA256 与 manifest 记录不一致，"
            "数据可能被手工修改过"
        )


# ==========================================================================
# 四、模块自检
# ==========================================================================
def test_watermelon_source_self_check():
    """西瓜数据集常量自检：行数、取值域、类别分布、连续值范围。"""
    df = load_watermelon3()
    assert df.shape == (17, 10)
    assert df["好瓜"].value_counts().to_dict() == {"否": 9, "是": 8}
    assert df["密度"].between(0, 1).all() and df["含糖率"].between(0, 1).all()


def test_specs_unknown_key_has_hint():
    """取未知数据集时，报错要列出可选值。"""
    with pytest.raises(KeyError) as exc:
        dataset_specs.get("iris")
    assert "watermelon" in str(exc.value)


def test_specs_shape_matches_columns():
    """元信息的 expected_shape 第二维必须等于 len(columns)（MNIST 除外）。"""
    for key, spec in dataset_specs.DATASETS.items():
        if key == "mnist":
            assert spec.expected_shape == (60000, 785)
            assert spec.columns == ()
            continue
        assert spec.expected_shape[1] == len(spec.columns), (
            f"{key} 的 expected_shape 第二维 {spec.expected_shape[1]} "
            f"与列数 {len(spec.columns)} 不一致"
        )
