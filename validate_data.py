#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据校验脚本（第 8 周实践二 第 7、8、12 节）。

分两层：

**第一层 · 通用 CSV 规范校验（作业第 2 节）**
    对 ``data/`` 下每个 CSV 逐条检查：UTF-8 无 BOM、逗号分隔、每行列数一致、
    无字面量 ``nan``/``NaN``/``None``、无多余索引列、换行符统一。
    这一层不依赖任何业务语义，用流式读取实现——MNIST 训练集 120 MB，
    若用 ``pandas.read_csv`` 整表载入会产生约 380 MB 的 int64 矩阵。

**第二层 · 业务语义校验（第 3、4、5、6 节）**
    形状、列名、缺失分布、取值范围、离散取值是否被误编码为数字，
    并把落盘 CSV 与**原始来源**（raw/ 里的原始文件、或内嵌常量）逐值比对。
    逐值比对是关键：只查形状和列名，无法发现"数值错位一列"这类错误。

用法
----
    python validate_data.py              # 校验全部，失败时退出码 1
    python validate_data.py --only boston
    python validate_data.py --verbose    # 逐条打印每个检查项

测试用例见 ``test_data.py``（pytest）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import dataset_specs
import prep_utils as U
from watermelon3_data import DISCRETE_DOMAINS, WATERMELON_3_0_ROWS, load_watermelon3

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = BASE_DIR / "raw"
RESULT_DIR = BASE_DIR / "results"

MNIST_PIXEL_COLS = 784
MNIST_LABEL_DOMAIN = set(range(10))

#: 检查注册表：dataset_key -> [(检查名, 可调用对象)]
CHECKS: dict[str, list[tuple[str, Callable[[], str]]]] = defaultdict(list)


def check(dataset: str):
    """装饰器：把函数登记为 ``dataset`` 的一条检查。

    被装饰函数无需参数，**通过即正常返回**（可返回一段说明字符串），
    **失败则抛 AssertionError**。
    """
    def deco(fn: Callable[[], str]):
        CHECKS[dataset].append((fn.__doc__.strip().splitlines()[0], fn))
        return fn
    return deco


class CheckFailure(AssertionError):
    """校验失败。单独定义便于和代码自身的 bug 区分。"""


def require(condition: bool, message: str) -> None:
    """``assert`` 的可读版本：失败时抛出带中文说明的 CheckFailure。"""
    if not condition:
        raise CheckFailure(message)


# ==========================================================================
# 通用工具
# ==========================================================================
_DF_CACHE: dict[str, pd.DataFrame] = {}


def load(key: str) -> pd.DataFrame:
    """读取并缓存数据集 DataFrame（重复调用只读一次盘）。"""
    if key not in _DF_CACHE:
        spec = dataset_specs.get(key)
        path = DATA_DIR / spec.filename
        # read_csv_safely 内部已按候选编码自动探测，这里不能再传 encoding
        df, enc = U.read_csv_safely(path, header=0 if spec.has_header else None)
        _DF_CACHE[key] = df
        _DF_CACHE[f"{key}:__encoding__"] = enc  # type: ignore[assignment]
    return _DF_CACHE[key]


def raw_bytes(path: Path, n: int = 8) -> bytes:
    with open(path, "rb") as fh:
        return fh.read(n)


def iter_lines(path: Path, encoding: str = "utf-8"):
    """逐行读取（流式），产出 (行号, 已去掉换行符的字符串)。"""
    with open(path, "r", encoding=encoding, newline="") as fh:
        for i, line in enumerate(fh, 1):
            yield i, line.rstrip("\r\n")


# ==========================================================================
# 第一层：通用 CSV 规范（对每个数据集都跑一遍）
# ==========================================================================
def csv_spec_checks(key: str) -> list[tuple[str, Callable[[], str]]]:
    """生成某个数据集上的通用规范检查项（第 2 节验收）。"""
    spec = dataset_specs.get(key)
    path = DATA_DIR / spec.filename
    expected_cols = spec.expected_shape[1]

    def _exists() -> str:
        """文件存在且非空"""
        require(path.exists(), f"文件不存在：{path}")
        require(path.stat().st_size > 0, f"文件为 0 字节：{path}")
        return U.human_size(path.stat().st_size)

    def _utf8_no_bom() -> str:
        """UTF-8 编码且无 BOM"""
        head = raw_bytes(path, 3)
        require(
            head != b"\xef\xbb\xbf",
            "文件带 UTF-8 BOM，pandas 以默认 utf-8 读取时首列名会变成 '\\ufeff...'",
        )
        try:
            path.read_bytes()[:1 << 20].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CheckFailure(f"前 1 MB 无法按 UTF-8 解码：{exc}") from exc
        return "无 BOM"

    def _comma_delimited() -> str:
        """英文逗号分隔、每行列数一致、无空行

        必须用 ``csv.reader`` 而不是 ``line.count(',')``——泰坦尼克号的 Name
        字段本身含逗号（如 ``"Braund, Mr. Owen Harris"``），按字符数逗号会把
        合法的带引号字段误判成多列。``csv.reader`` 会正确处理引号包裹。
        """
        n_cols = None
        reader = csv.reader(line for _, line in iter_lines(path))
        for lineno, row in enumerate(reader, 1):
            if not row:
                raise CheckFailure(f"第 {lineno} 行为空行")
            if n_cols is None:
                n_cols = len(row)
                require(
                    n_cols == expected_cols,
                    f"首行列数为 {n_cols}，期望 {expected_cols}"
                    "（分隔符可能不是英文逗号）",
                )
                continue
            require(
                len(row) == n_cols,
                f"第 {lineno} 行解析出 {len(row)} 个字段，与首行的 {n_cols} 不一致"
                "（可能存在未加引号的逗号或空行）",
            )
        return f"{n_cols} 列/行"

    def _no_literal_nan() -> str:
        """缺失值写成空字段，不出现字面量 nan / NaN / None"""
        bad = {"nan", "NaN", "NAN", "None", "null", "NULL"}
        hits = []
        reader = csv.reader(line for _, line in iter_lines(path))
        for lineno, row in enumerate(reader, 1):
            for cell in row:
                if cell in bad:
                    hits.append(f"第 {lineno} 行 '{cell}'")
                    if len(hits) >= 3:
                        break
            if len(hits) >= 3:
                break
        require(not hits, f"发现字面量缺失标记：{hits}")
        return "全部为空字段"

    def _no_index_column() -> str:
        """无多余索引列（列数与预期一致，且无 Unnamed 列）"""
        if not spec.has_header:
            # MNIST 无表头：只读首行判列数，避免把 105 MB 整表载入成 int64
            _, first = next(iter_lines(path))
            n_cols = first.count(",") + 1
            require(
                n_cols == expected_cols,
                f"首行列数为 {n_cols}，期望 {expected_cols}，"
                "多出的那一列很可能是 to_csv 时未加 index=False 产生的索引列",
            )
            require(
                first.split(",")[0].isdigit(),
                "首列不是 0–9 的数字标签，说明第一列是多余的行索引",
            )
            return f"{n_cols} 列，无索引列（无表头，按首行判定）"

        df = load(key)
        require(
            df.shape[1] == expected_cols,
            f"实际 {df.shape[1]} 列，期望 {expected_cols} 列，"
            "多出的那一列很可能是 to_csv 时未加 index=False 产生的索引列",
        )
        unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
        require(not unnamed, f"存在无名列（典型的多余索引列）：{unnamed}")
        return f"{df.shape[1]} 列，无索引列"

    def _unix_newline() -> str:
        """换行符统一为 LF，文件以换行结尾"""
        data = path.read_bytes()
        require(b"\r\n" not in data, "存在 CRLF 换行，跨平台复现时会产生 diff")
        require(data.endswith(b"\n"), "文件末尾缺少换行符")
        return "LF 换行"

    def _minimal_quoting() -> str:
        """仅必要字段加引号（未使用 QUOTE_ALL）"""
        _, first = next(iter_lines(path))
        fields = next(csv.reader([first]))
        if len(fields) > 1 and first.startswith('"') and \
                first.count('",') == len(fields) - 1:
            raise CheckFailure(
                "整行每个字段都被引号包裹，说明 to_csv 用了 QUOTE_ALL"
                "（quoting=1）；应使用 QUOTE_MINIMAL（quoting=0），"
                "否则文件体积膨胀且不便人工核对"
            )
        return "仅必要字段加引号"

    return [
        ("文件存在且非空", _exists),
        ("UTF-8 编码且无 BOM", _utf8_no_bom),
        ("逗号分隔且每行列数一致", _comma_delimited),
        ("缺失值写为空字段（无 nan 字面量）", _no_literal_nan),
        ("无多余索引列", _no_index_column),
        ("仅必要字段加引号", _minimal_quoting),
        ("换行符统一为 LF", _unix_newline),
    ]


# ==========================================================================
# 第二层：MNIST（流式校验，不整表载入）
# ==========================================================================
#: 流式扫描结果的缓存，供多个检查项复用，避免 120 MB 文件被反复读
_MNIST_STATS: dict[str, dict] = {}


def _scan_mnist_csv(path: Path, expected_rows: int) -> dict:
    """流式扫描 MNIST CSV，累计行数、列宽、标签分布、像素极值与空字段数。"""
    label_counts = np.zeros(10, dtype=np.int64)
    pixel_min, pixel_max = 255, 0
    empty_cells = 0
    bad_width = 0
    rows = 0

    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            rows += 1
            if len(row) != MNIST_PIXEL_COLS + 1:
                bad_width += 1
                continue
            if any(cell == "" for cell in row):
                empty_cells += 1
            label = int(row[0])
            label_counts[label] += 1
            vals = np.array(row[1:], dtype=np.int16)
            pixel_min = min(pixel_min, int(vals.min()))
            pixel_max = max(pixel_max, int(vals.max()))
            if rows > expected_rows:      # 早停：行数超了就没必要读完
                break

    return {
        "rows": rows, "label_counts": label_counts.tolist(),
        "pixel_min": pixel_min, "pixel_max": pixel_max,
        "empty_rows": empty_cells, "bad_width_rows": bad_width,
    }


@check("mnist")
def _mnist_files() -> str:
    """训练集与测试集 CSV 均存在"""
    paths = [DATA_DIR / "mnist_train.csv", DATA_DIR / "mnist_test.csv"]
    for p in paths:
        require(p.exists(), f"缺少 {p.name}")
    sizes = [U.human_size(p.stat().st_size) for p in paths]
    return f"train {sizes[0]} / test {sizes[1]}"


@check("mnist")
def _mnist_no_header() -> str:
    """无表头：首行即数据，首列取值为 0–9 的整数标签"""
    for name in ("mnist_train.csv", "mnist_test.csv"):
        with open(DATA_DIR / name, "r", encoding="utf-8") as fh:
            first = fh.readline().strip()
        first_col = first.split(",")[0]
        require(
            first_col.isdigit(),
            f"{name} 首行首列为 {first_col!r}，不是数字标签——说明写入了表头",
        )
        require(
            int(first_col) in MNIST_LABEL_DOMAIN,
            f"{name} 首行标签 {first_col} 不在 0–9 内",
        )
    return "首行即数据"


@check("mnist")
def _mnist_shape() -> str:
    """形状为 60000×785 与 10000×785"""
    spec = dataset_specs.get("mnist")
    stats = {}
    for name, expect in (("mnist_train.csv", spec.expected_shape[0]),
                         ("mnist_test.csv", spec.extra["expected_shape_test"][0])):
        s = _scan_mnist_csv(DATA_DIR / name, expect)
        require(
            s["rows"] == expect,
            f"{name} 行数 {s['rows']}，期望 {expect}",
        )
        require(
            s["bad_width_rows"] == 0,
            f"{name} 有 {s['bad_width_rows']} 行列数不等于 785",
        )
        stats[name] = s
    _MNIST_STATS.update(stats)
    return "60000×785 / 10000×785"


@check("mnist")
def _mnist_labels() -> str:
    """第 1 列 label 取值在 0–9，且 10 个类别全部出现"""
    for name, expect in (("mnist_train.csv", 60000), ("mnist_test.csv", 10000)):
        s = _MNIST_STATS.get(name) or _scan_mnist_csv(DATA_DIR / name, expect)
        counts = s["label_counts"]
        require(sum(counts) == expect, f"{name} 标签总数 {sum(counts)} ≠ {expect}")
        require(
            all(c > 0 for c in counts),
            f"{name} 存在缺失类别：{counts}",
        )
    return f"train {_MNIST_STATS['mnist_train.csv']['label_counts']}"


@check("mnist")
def _mnist_pixel_range() -> str:
    """像素取值在 0–255，且未被归一化（最大值必须为 255）"""
    s = _MNIST_STATS.get("mnist_train.csv") or _scan_mnist_csv(
        DATA_DIR / "mnist_train.csv", 60000)
    require(
        0 <= s["pixel_min"] and s["pixel_max"] <= 255,
        f"像素越界：[{s['pixel_min']}, {s['pixel_max']}]，应为 0–255",
    )
    require(
        s["pixel_max"] == 255,
        f"像素最大值 {s['pixel_max']} ≠ 255，数据似乎已被归一化（违反第 3 节要求）",
    )
    return f"[{s['pixel_min']}, {s['pixel_max']}]，未归一化"


@check("mnist")
def _mnist_no_missing() -> str:
    """784 个像素列无空字段"""
    total_empty = sum(
        (_MNIST_STATS.get(n) or _scan_mnist_csv(DATA_DIR / n, e))["empty_rows"]
        for n, e in (("mnist_train.csv", 60000), ("mnist_test.csv", 10000))
    )
    require(total_empty == 0, f"有 {total_empty} 行存在空字段")
    return "无空字段"


@check("mnist")
def _mnist_matches_raw() -> str:
    """与 raw/ 中的原始 IDX 文件逐像素比对"""
    idx_dir = RAW_DIR
    need = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"]
    if not all((idx_dir / n).exists() for n in need):
        return "跳过（raw/ 无 IDX 原始文件，可能走的是 npz 降级路径）"

    images = U.read_idx_images(idx_dir / need[0])
    labels = U.read_idx_labels(idx_dir / need[1])
    with open(DATA_DIR / "mnist_train.csv", "r", encoding="utf-8") as fh:
        for i in range(images.shape[0]):
            row = fh.readline().rstrip("\n").split(",")
            require(len(row) == 785, f"第 {i + 1} 行列数 {len(row)} ≠ 785")
            require(
                int(row[0]) == int(labels[i]),
                f"第 {i + 1} 行标签 {row[0]} ≠ 原始标签 {labels[i]}",
            )
            flat = np.array(row[1:], dtype=np.uint8).reshape(28, 28)
            require(
                np.array_equal(flat, images[i]),
                f"第 {i + 1} 行像素与原始 IDX 不一致（展平顺序或数值有误）",
            )
    return f"60000 行 × 784 像素逐值一致"


# ==========================================================================
# 第二层：波士顿房价
# ==========================================================================
@check("boston")
def _boston_shape() -> str:
    """形状为 506 行 × 14 列"""
    df = load("boston")
    require(df.shape == (506, 14), f"实际 {df.shape}，期望 (506, 14)")
    return "506 × 14"


@check("boston")
def _boston_columns() -> str:
    """列名与顺序完全等于 UCI 原始 14 列"""
    df = load("boston")
    require(
        tuple(df.columns) == dataset_specs.BOSTON_COLUMNS,
        f"列名不符。\n      实际：{list(df.columns)}\n"
        f"      期望：{list(dataset_specs.BOSTON_COLUMNS)}",
    )
    return "14 列全部一致"


@check("boston")
def _boston_no_missing() -> str:
    """无缺失值（UCI 原始数据本身完整）"""
    df = load("boston")
    n = int(df.isna().sum().sum())
    require(n == 0, f"存在 {n} 个缺失单元格")
    return "0 个缺失"


@check("boston")
def _boston_value_domains() -> str:
    """CHAS ∈ {0,1}、MEDV ∈ [5,50]、RAD/TAX 为非负整数"""
    df = load("boston")
    require(
        set(df["CHAS"].unique()) <= {0, 1},
        f"CHAS 应只取 0/1，实际 {sorted(df['CHAS'].unique())}",
    )
    require(
        df["MEDV"].between(5, 50).all(),
        f"MEDV 越界：[{df['MEDV'].min()}, {df['MEDV'].max()}]，应落在 [5, 50]",
    )
    for col in ("RAD", "TAX"):
        require(
            (df[col] >= 0).all() and (df[col] % 1 == 0).all(),
            f"{col} 应为非负整数",
        )
    return f"CHAS {sorted(df['CHAS'].unique())} / MEDV [{df['MEDV'].min()}, {df['MEDV'].max()}]"


@check("boston")
def _boston_first_row() -> str:
    """首行与 UCI 原始 housing.data 首行一致"""
    df = load("boston")
    row = df.iloc[0]
    require(abs(row["CRIM"] - 0.00632) < 1e-9, f"CRIM 首行应为 0.00632，实际 {row['CRIM']}")
    require(abs(row["RM"] - 6.575) < 1e-9, f"RM 首行应为 6.575，实际 {row['RM']}")
    require(abs(row["MEDV"] - 24.0) < 1e-9, f"MEDV 首行应为 24.0，实际 {row['MEDV']}")
    return "CRIM=0.00632, MEDV=24.0"


@check("boston")
def _boston_matches_raw() -> str:
    """与 raw/housing.data 逐值比对（防错列、防丢行）"""
    raw = RAW_DIR / "housing.data"
    if not raw.exists():
        return "跳过（raw/housing.data 不存在）"
    df = load("boston")
    raw_df = pd.read_csv(raw, sep=r"\s+", header=None,
                         names=list(dataset_specs.BOSTON_COLUMNS))
    require(
        raw_df.shape == df.shape,
        f"与原始文件形状不一致：原始 {raw_df.shape}，CSV {df.shape}",
    )
    diff = (raw_df - df).abs().to_numpy().max()
    require(diff < 1e-9, f"与原始文件最大数值差异 {diff}，说明发生过舍入或错列")
    return "506 × 14 逐值一致（最大差异 < 1e-9）"


# ==========================================================================
# 第二层：泰坦尼克号
# ==========================================================================
@check("titanic")
def _titanic_shape() -> str:
    """形状为 891 行 × 12 列（未删行）"""
    df = load("titanic")
    require(df.shape == (891, 12), f"实际 {df.shape}，期望 (891, 12)")
    return "891 × 12"


@check("titanic")
def _titanic_columns() -> str:
    """原始 12 列齐全且顺序一致"""
    df = load("titanic")
    require(
        tuple(df.columns) == dataset_specs.TITANIC_COLUMNS,
        f"列名不符。\n      实际：{list(df.columns)}\n"
        f"      期望：{list(dataset_specs.TITANIC_COLUMNS)}",
    )
    return "12 列全部一致"


@check("titanic")
def _titanic_missing_preserved() -> str:
    """Age=177、Cabin=687、Embarked=2 处原始缺失原样保留"""
    df = load("titanic")
    expect = {"Age": 177, "Cabin": 687, "Embarked": 2}
    actual = {c: int(df[c].isna().sum()) for c in expect}
    require(
        actual == expect,
        f"缺失分布不符：实际 {actual}，期望 {expect}"
        "（若为 0，说明发生了填充；若更大，说明混入了测试集或解析出错）",
    )
    total = int(df.isna().sum().sum())
    require(total == 866, f"总缺失数应为 866，实际 {total}")
    return "Age 177 / Cabin 687 / Embarked 2"


@check("titanic")
def _titanic_sex_not_encoded() -> str:
    """Sex 仍为原始字符串 male/female，未被编码为 0/1"""
    df = load("titanic")
    values = set(df["Sex"].dropna().unique())
    require(
        values == {"male", "female"},
        f"Sex 取值为 {sorted(values)}，应为 {{'female', 'male'}}——"
        "被编码成数字违反第 5 节验收要求",
    )
    return f"male {int((df['Sex'] == 'male').sum())} / female {int((df['Sex'] == 'female').sum())}"


@check("titanic")
def _titanic_survived() -> str:
    """Survived ∈ {0,1} 且正类 342 人"""
    df = load("titanic")
    require(set(df["Survived"].unique()) == {0, 1},
            f"Survived 应为 0/1，实际 {sorted(df['Survived'].unique())}")
    n = int(df["Survived"].sum())
    require(n == 342, f"生存人数应为 342，实际 {n}")
    return f"342 生存 / {891 - 342} 遇难"


@check("titanic")
def _titanic_name_quoting() -> str:
    """Name 含逗号，正确加引号后仍能完整解析回原值"""
    df = load("titanic")
    first = df.loc[0, "Name"]
    require(
        first == "Braund, Mr. Owen Harris",
        f"首行 Name 解析为 {first!r}，期望 'Braund, Mr. Owen Harris'——"
        "含逗号的字段未正确加引号时会被切成多列",
    )
    n_with_comma = int(df["Name"].str.contains(",").sum())
    require(n_with_comma == 891, f"含逗号的 Name 应有 891 个，实际 {n_with_comma}")
    return f"891 个含逗号字段均正确转义"


@check("titanic")
def _titanic_matches_raw() -> str:
    """与 raw/titanic.csv 逐单元格比对"""
    raw = RAW_DIR / "titanic.csv"
    if not raw.exists():
        return "跳过（raw/titanic.csv 不存在）"
    raw_df, _ = U.read_csv_safely(raw)
    df = load("titanic")
    require(raw_df.shape == df.shape,
            f"与原始文件形状不一致：原始 {raw_df.shape}，CSV {df.shape}")
    pd.testing.assert_frame_equal(raw_df, df, check_dtype=False)
    return "891 × 12 逐单元格一致"


# ==========================================================================
# 第二层：西瓜数据集 3.0
# ==========================================================================
@check("watermelon")
def _wm_shape() -> str:
    """形状为 17 行 × 10 列"""
    df = load("watermelon")
    require(df.shape == (17, 10), f"实际 {df.shape}，期望 (17, 10)")
    return "17 × 10"


@check("watermelon")
def _wm_fields() -> str:
    """作业要求的 9 个业务字段全部存在"""
    df = load("watermelon")
    required = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖率", "好瓜"]
    missing = [c for c in required if c not in df.columns]
    require(not missing, f"缺少字段：{missing}")
    return "9 个字段齐全"


@check("watermelon")
def _wm_discrete_not_encoded() -> str:
    """6 个离散属性保持中文字符串，未编码为数字"""
    df = load("watermelon")
    for col, domain in DISCRETE_DOMAINS.items():
        actual = set(df[col].astype(str))
        require(
            actual <= domain,
            f"{col} 出现非法取值 {sorted(actual - domain)}；合法集合为 {sorted(domain)}",
        )
        require(
            not actual <= {str(i) for i in range(10)},
            f"{col} 取值全是数字，说明离散特征被编码成了数字",
        )
        # 注意：pandas 3 起字符串列的 dtype 是 str 而非 object，
        # 所以判定"未被编码为数字"要用 is_numeric_dtype，不能写 dtype == object
        require(
            not pd.api.types.is_numeric_dtype(df[col]),
            f"{col} 的 dtype 为 {df[col].dtype}（数值类型），"
            "说明离散特征被编码成了数字",
        )
    return "6 个离散字段均为中文原值"


@check("watermelon")
def _wm_label_balance() -> str:
    """好瓜 8 个、否 9 个（教材 3.0 原表）"""
    df = load("watermelon")
    vc = df["好瓜"].value_counts().to_dict()
    require(vc.get("是") == 8 and vc.get("否") == 9,
            f"类别分布应为 是 8 / 否 9，实际 {vc}")
    return "是 8 / 否 9"


@check("watermelon")
def _wm_numeric_ranges() -> str:
    """密度、含糖率为浮点数且落在 (0,1)"""
    df = load("watermelon")
    for col in ("密度", "含糖率"):
        require(
            pd.api.types.is_float_dtype(df[col]),
            f"{col} 的 dtype 为 {df[col].dtype}，应为浮点",
        )
        require(
            df[col].between(0, 1, inclusive="neither").all(),
            f"{col} 存在越界值：[{df[col].min()}, {df[col].max()}]",
        )
    return f"密度 [{df['密度'].min():.3f}, {df['密度'].max():.3f}]"


@check("watermelon")
def _wm_no_missing() -> str:
    """17 个样本无缺失值"""
    df = load("watermelon")
    n = int(df.isna().sum().sum())
    require(n == 0, f"存在 {n} 个缺失单元格")
    return "0 个缺失"


@check("watermelon")
def _wm_matches_source() -> str:
    """与内嵌的教材原表逐行逐字段比对"""
    df = load("watermelon")
    ref = load_watermelon3()
    require(raw_bytes(DATA_DIR / "watermelon3.0.csv", 3) != b"\xef\xbb\xbf",
            "带 BOM")
    pd.testing.assert_frame_equal(ref, df, check_dtype=False,
                                  check_names=True, check_exact=False)
    # 再逐行核对原始常量元组，防止常量本身被误改
    for i, expected in enumerate(WATERMELON_3_0_ROWS):
        actual = tuple(df.iloc[i].tolist())
        require(
            all(
                (abs(a - e) < 1e-9) if isinstance(e, float) else (a == e)
                for a, e in zip(actual, expected)
            ),
            f"第 {i + 1} 行与教材原表不一致：\n      实际 {actual}\n      期望 {expected}",
        )
    return "17 行 × 10 列逐值一致"


# ==========================================================================
# 运行器
# ==========================================================================
def run_validation(keys: list[str] | None = None, *, verbose: bool = False) -> list[dict]:
    """执行全部检查，返回结果列表（每项含 dataset / name / ok / detail）。"""
    keys = keys or list(dataset_specs.PREPARE_ORDER)
    results: list[dict] = []

    for key in keys:
        spec = dataset_specs.get(key)
        items = csv_spec_checks(key) + list(CHECKS.get(key, []))
        print(f"\n[{key}] {spec.display_name}（{len(items)} 项检查）")
        for name, fn in items:
            try:
                detail = fn() or ""
                results.append({"dataset": key, "name": name, "ok": True,
                                "detail": str(detail)})
                if verbose:
                    print(f"  ✓ {name}  —  {detail}")
            except (AssertionError, FileNotFoundError, U.DataPrepError) as exc:
                results.append({"dataset": key, "name": name, "ok": False,
                                "detail": str(exc)})
                print(f"  ✗ {name}\n      {str(exc).replace(chr(10), chr(10) + '      ')}")
        passed = sum(1 for r in results if r["dataset"] == key and r["ok"])
        total = sum(1 for r in results if r["dataset"] == key)
        print(f"  → {passed}/{total} 通过")

    ok = sum(1 for r in results if r["ok"])
    print("\n" + "=" * 72)
    print(f"校验汇总：{ok}/{len(results)} 项通过")
    if ok != len(results):
        print("失败项：")
        for r in results:
            if not r["ok"]:
                print(f"  - [{r['dataset']}] {r['name']}")
    print("=" * 72)
    return results


def write_report(results: list[dict]) -> Path:
    """把校验结果写成 results/validation_report.md。"""
    U.ensure_dir(RESULT_DIR)
    ok = sum(1 for r in results if r["ok"])
    lines = [
        "# 数据校验报告",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 结果：**{ok} / {len(results)} 项通过**",
        "",
        "| 数据集 | 检查项 | 结果 | 说明 |",
        "|---|---|---|---|",
    ]
    for r in results:
        detail = str(r["detail"]).replace("\n", " ").replace("|", "\\|")[:90]
        lines.append(
            f"| {r['dataset']} | {r['name']} | {'通过' if r['ok'] else '**失败**'} "
            f"| {detail} |"
        )
    lines.append("")
    path = RESULT_DIR / "validation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="第 8 周实践二：数据校验")
    parser.add_argument("--only", default="", help="只校验指定数据集，逗号分隔")
    parser.add_argument("--verbose", action="store_true", help="逐条打印通过的检查项")
    parser.add_argument("--quiet", action="store_true", help="只输出汇总")
    args = parser.parse_args(argv)

    if not DATA_DIR.exists() or not any(DATA_DIR.glob("*.csv")):
        print(f"错误：{DATA_DIR} 下没有 CSV，请先运行 prepare_data.py", file=sys.stderr)
        return 2

    keys = ([k.strip() for k in args.only.split(",") if k.strip()]
            if args.only else None)
    if keys:
        unknown = [k for k in keys if k not in dataset_specs.DATASETS]
        if unknown:
            print(f"错误：未知数据集 {unknown}", file=sys.stderr)
            return 2
        missing = [k for k in keys if not (DATA_DIR / dataset_specs.get(k).filename).exists()]
        if missing:
            print(f"错误：以下数据集尚未准备：{missing}", file=sys.stderr)
            return 2

    if args.quiet:
        import contextlib
        import io as _io
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            results = run_validation(keys, verbose=False)
    else:
        results = run_validation(keys, verbose=args.verbose)

    path = write_report(results)
    print(f"报告：{path}")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
