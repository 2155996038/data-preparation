"""数据集元信息（唯一事实来源 / single source of truth）。

作业第 10 节要求"记录每个数据集的来源链接或仓库名称、版本号、获取日期"。
本模块把这些信息集中在一处，``prepare_data.py``、``validate_data.py``、
``README.md`` 的生成以及 ``SOURCES.md`` 都从这里取数，避免多处硬编码不一致。

字段说明
--------
key            : 内部标识，同时是命令行 ``--only`` 的取值
filename       : 最终 CSV 文件名（data/ 下）
display_name   : 中文名，用于日志和报告
source_name    : 数据来源名称
source_url     : 可访问的来源链接（原始下载地址）
alt_urls       : 备用镜像，主地址失败时按顺序尝试
version        : 版本号 / 快照标识
raw_files      : 原始文件名（写进 README，便于他人核对）
raw_format     : 原始文件格式说明
expected_shape : 期望的最终形状 (行数, 列数)
columns        : 期望列名元组；MNIST 无表头时为空元组
encoding       : CSV 编码
has_header     : 是否写表头
notes          : 转换方式备注
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    filename: str
    display_name: str
    source_name: str
    source_url: str
    version: str
    raw_files: tuple[str, ...]
    raw_format: str
    expected_shape: tuple[int, int]
    columns: tuple[str, ...] = ()
    encoding: str = "utf-8"
    has_header: bool = True
    alt_urls: tuple[str, ...] = ()
    notes: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# 波士顿房价：列名必须与 UCI 原始 housing.names 完全一致（作业第 4 节）
# 注意：作业明确禁止使用已弃用的 sklearn.datasets.load_boston。
# --------------------------------------------------------------------------
BOSTON_COLUMNS = (
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV",
)

# --------------------------------------------------------------------------
# 泰坦尼克号：Kaggle 训练集原始 12 列，顺序与原始文件一致（作业第 5 节）
# --------------------------------------------------------------------------
TITANIC_COLUMNS = (
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
)

# --------------------------------------------------------------------------
# 西瓜数据集 3.0：《机器学习》（周志华，2016）附录表格，9 个业务字段 + 编号（作业第 6 节）
# --------------------------------------------------------------------------
WATERMELON_COLUMNS = (
    "编号", "色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖率", "好瓜",
)


DATASETS: dict[str, DatasetSpec] = {
    # ------------------------------------------------------------------ MNIST
    "mnist": DatasetSpec(
        key="mnist",
        filename="mnist_train.csv",  # 训练集；测试集见 extra["filename_test"]
        display_name="MNIST 手写数字",
        source_name="MNIST 官方 IDX 文件（LeCun 等）· S3 镜像",
        source_url="https://ossci-datasets.s3.amazonaws.com/mnist/",
        alt_urls=(
            # Keras 打包版，作为 IDX 全部失败时的降级方案
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
        ),
        version="原始 IDX v3（60000 / 10000 标准划分）",
        raw_files=(
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ),
        raw_format="IDX（大端序二进制）经 gzip 压缩，图像 28×28 uint8",
        expected_shape=(60000, 785),
        columns=(),           # 无表头
        has_header=False,     # 作业第 3 节：无表头
        extra={
            "filename_test": "mnist_test.csv",
            "expected_shape_test": (10000, 785),
            # 四个原始文件的标准 MD5，用于确认下载完整（来源：MNIST 官方页面）
            "md5": {
                "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
                "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
                "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
                "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
            },
        },
        notes="IDX 二进制解析为 uint8 数组，28×28 按行展平为 784 列；"
              "第 1 列 label，后 784 列像素；不归一化、不写表头、不写索引。",
    ),

    # -------------------------------------------------------------- 波士顿房价
    "boston": DatasetSpec(
        key="boston",
        filename="boston_housing.csv",
        display_name="波士顿房价（Boston Housing）",
        source_name="UCI Machine Learning Repository · Housing",
        source_url="https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
        alt_urls=(
            "https://archive.ics.uci.edu/static/public/53/boston+housing.zip",
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv",
        ),
        version="UCI Housing（506 条，1993 年快照，未做任何删改）",
        raw_files=("housing.data", "housing.names"),
        raw_format="空白分隔纯文本，无表头，506 行 × 14 个数值字段",
        expected_shape=(506, 14),
        columns=BOSTON_COLUMNS,
        notes="按 14 个已知列名解析空白分隔文本；不使用 sklearn.datasets.load_boston，"
              "不删除任何行、不改列名、不做任何清洗。",
    ),

    # ---------------------------------------------------------------- 泰坦尼克号
    "titanic": DatasetSpec(
        key="titanic",
        filename="titanic_train.csv",
        display_name="泰坦尼克号（Titanic）",
        source_name="Kaggle · Titanic: Machine Learning from Disaster（社区镜像 datasciencedojo/datasets）",
        source_url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        alt_urls=(),
        version="Kaggle train.csv 原始快照（891 行 × 12 列）",
        raw_files=("titanic.csv",),
        raw_format="UTF-8 CSV，含表头，891 行 × 12 列，字段含逗号与引号需正确转义",
        expected_shape=(891, 12),
        columns=TITANIC_COLUMNS,
        notes="原样读取后原样落盘：保留 Age / Cabin / Embarked 的原始缺失（空字段），"
              "不删行、不填充、不把 Sex 编码成 0/1、不裁剪列。",
    ),

    # ------------------------------------------------------------ 西瓜数据集 3.0
    "watermelon": DatasetSpec(
        key="watermelon",
        filename="watermelon3.0.csv",
        display_name="西瓜数据集 3.0",
        source_name="《机器学习》（周志华，清华大学出版社，2016）附录 西瓜数据集 3.0",
        source_url="（教材附录表格，无公开下载链接，故以代码内嵌常量形式落盘）",
        alt_urls=(),
        version="3.0（17 个样本；区别于 2.0 的 17 样本无密度/含糖率、4.0 的 17 样本含纹理）",
        raw_files=("（内嵌常量）",),
        raw_format="教材附录中的二维表格，9 个业务字段，其中 6 个为中文离散取值",
        expected_shape=(17, 10),
        columns=WATERMELON_COLUMNS,
        notes="离散特征（色泽/根蒂/敲声/纹理/脐部/触感）与标签（好瓜）保持中文字符串，"
              "不编码为数字；密度、含糖率保持原始 3 位小数。",
    ),
}

# 数据准备顺序：小表在前，便于快速验证；大表（MNIST）最后
PREPARE_ORDER = ("watermelon", "boston", "titanic", "mnist")


def get(key: str) -> DatasetSpec:
    """按 key 取数据集元信息，key 非法时抛出带提示的 KeyError。"""
    try:
        return DATASETS[key]
    except KeyError:
        raise KeyError(
            f"未知数据集 {key!r}，可选：{', '.join(DATASETS)}"
        ) from None


def all_specs() -> list[DatasetSpec]:
    """按准备顺序返回全部数据集元信息。"""
    return [DATASETS[k] for k in PREPARE_ORDER]
