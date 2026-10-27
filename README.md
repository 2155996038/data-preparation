# 2025111142_陈锦煌_数据准备

> 第 8 周实践二《数据集收集与准备》（人工智能与机器学习）
> **学号：2025111142　姓名：陈锦煌　完成日期：2026-10-27**

第 8 周实践二《数据集收集与准备》。目标：把 4 个数据集从**原始来源**转换为**统一规范**的
CSV，过程可追溯、结果可复现、异常有提示、提交有代码而不只是成品数据。

---

## 一、目录结构

```
2025111142_陈锦煌_数据准备/
├── data/                       ← 5 个规范 CSV（提交物）
│   ├── mnist_train.csv             60000 × 785，无表头，104.5 MB
│   ├── mnist_test.csv              10000 × 785，无表头，17.4 MB
│   ├── boston_housing.csv          506 × 14，有表头
│   ├── titanic_train.csv           891 × 12，有表头
│   └── watermelon3.0.csv           17 × 10，有表头
├── raw/                        ← 原始文件缓存（原始下载物，便于核对，不提交到 git）
│   ├── train-images-idx3-ubyte.gz 等 4 个 MNIST IDX 文件
│   ├── mnist.npz                   降级备选来源
│   ├── housing.data / housing.names
│   └── titanic.csv
├── results/                    ← 运行产物
│   ├── data_report.md              本次运行的明细报告（形状、缺失、SHA256）
│   ├── manifest.json               机器可读清单（含每个 CSV 的 SHA256）
│   └── validation_report.md        55 项校验的逐条结果
├── prepare_data.py             ← 主脚本：下载 → 解析 → 转换 → 落盘 → 出报告
├── validate_data.py            ← 校验脚本：规范校验 + 语义校验，55 项
├── test_data.py                ← pytest：82 个测试用例
├── prep_utils.py               ← 通用工具：下载重试、编码探测、原子写出、IDX 解析
├── dataset_specs.py            ← 数据集元信息（来源、版本、期望形状、列名）
├── watermelon3_data.py         ← 西瓜数据集 3.0 的教材原表常量（含自检）
├── make_zip.py                 ← 打包：按目录名生成 <目录名>.zip
├── requirements.txt
└── README.md
```

**为什么把元信息单独放 `dataset_specs.py`**：来源、版本、期望形状这些信息在
`prepare_data.py`、`validate_data.py`、README、`SOURCES.md` 里都要用到。
集中一处定义，改一个地方全部同步，不会出现"报告里说 506 行、校验里写 507 行"这种自相矛盾。

---

## 二、运行环境

| 项 | 值 |
|---|---|
| Python | 3.13 |
| pandas | 3.1.0 |
| numpy | 2.5.3 |
| 其它 | 仅标准库（urllib / gzip / csv / hashlib / argparse / json） |

```bash
pip install -r requirements.txt
```

**不需要安装 TensorFlow / Keras / scikit-learn。** MNIST 走 IDX 二进制手动解析，
Keras 打包的 `mnist.npz` 只作为备用来源用 `numpy.load` 读取；波士顿房价严格不用
`sklearn.datasets.load_boston`（该方法在新版 sklearn 已被移除，且作业明确禁止）。

---

## 三、复现步骤

```bash
# 1) 生成全部 CSV（首次运行会下载约 22 MB 原始文件）
python prepare_data.py

# 2) 校验（55 项，全绿才可提交）
python validate_data.py

# 3) 单元测试（82 个用例）
python -m pytest test_data.py -q

# 4) 打包提交（目录改名后会生成 <新目录名>.zip）
python make_zip.py
```

常见参数：

```bash
python prepare_data.py --list                  # 列出支持的数据集与期望形状
python prepare_data.py --only mnist             # 只准备某一个
python prepare_data.py --only boston,titanic    # 只准备多个
python prepare_data.py --force                  # 忽略已有 CSV，强制重生成
python prepare_data.py --skip-download          # 完全离线，只用 raw/ 缓存
python validate_data.py --verbose               # 逐条打印通过的检查项
```

**复现性说明**：本任务不涉及随机采样，所有输出都是确定性的。
连续运行两次 `prepare_data.py`，5 个 CSV 的 SHA256 逐字节一致。
脚本启动时把当天日期写入报告，因此只有 `results/` 下的时间戳字段会变。

---

## 四、CSV 统一规范（作业第 2 节）

所有 CSV 统一由 `prep_utils.write_csv_standard()` 写出，规范集中在一处：

| 规范 | 实现 | 校验方式 |
|---|---|---|
| UTF-8 编码 | `encoding="utf-8"`，**不带 BOM** | 读前 3 字节比对 `EF BB BF` |
| 英文逗号分隔 | `sep=","` | `csv.reader` 解析后字段数一致 |
| 第一行表头（MNIST 除外） | `header=True`，MNIST 用 `header=False` | 首行首列是否为 0–9 数字 |
| 不保存行索引 | `index=False` | 列数与期望一致、无 `Unnamed` 列 |
| 缺失值留空字段 | `na_rep=""` | 全文扫描无 `nan`/`NaN`/`None` 字面量 |
| 不改列名、不加单位与千分位 | 原样传递 | 列名逐字比对；波士顿无 `$`、无 `,` 千分位 |
| — （附加）仅必要字段加引号 | `quoting=csv.QUOTE_MINIMAL`（=0） | 首行不应整行被引号包裹 |
| — （附加）换行符统一 | `lineterminator="\n"` | 全文无 `\r\n`，末尾有换行 |

**为什么强调"无 BOM"**：带 BOM 时 `pd.read_csv(path)` 用默认 utf-8 读出来，
第一列列名会变成 `\ufeff编号`，肉眼看不出来但按列名取数会全部失败。
这正是作业"用 pandas.read_csv 可直接读取、不改列名"的验收点。

---

## 五、四个数据集的来源与转换（作业第 3、4、5、6、8、9 节）

### 5.1 MNIST 手写数字

| 项 | 内容 |
|---|---|
| **来源** | MNIST 官方 IDX 文件（LeCun, Cortes, Burges），S3 镜像 `https://ossci-datasets.s3.amazonaws.com/mnist/` |
| **获取日期** | 2026-10-27 |
| **原始文件名** | `train-images-idx3-ubyte.gz`(9.5 MB)、`train-labels-idx1-ubyte.gz`、`t10k-images-idx3-ubyte.gz`、`t10k-labels-idx1-ubyte.gz` |
| **原始格式** | IDX：大端序二进制 + gzip 压缩。头部 4 字节为魔数（`0x00000803` 图像 / `0x00000801` 标签），**低 8 位就是维数**，紧随其后是各维长度（各 4 字节），之后是裸 uint8 数据 |
| **转换方式** | `gzip` 解压 → 按 IDX 头解析为 `np.ndarray` → 28×28 按行展平为 784 维 → 与标签拼成 785 列 → 分块写 CSV |
| **最终行列数** | `mnist_train.csv` **60000 × 785**；`mnist_test.csv` **10000 × 785** |
| **未做处理** | 未归一化、未打乱、未做任何增强；像素保持 0–255 整数 |
| **缺失值** | 无（0 个空字段） |
| **完整性校验** | 4 个原始文件比对官方 MD5 全部通过；标签分布训练集 `[5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]`，合计 60000 |

**遇到的问题①：IDX 头部多读了一个字段。**
第一版解析器按"魔数(4B) + 维数(4B) + 各维长度"实现，结果报错
`维数应为 3，实际 60000`。原因是 IDX 规范里**没有独立的维数字段**——维数藏
在魔数的第 4 个字节里（`byte 3`），紧接魔数之后就是 `dim[0]`。修正后正确解析出
`(60000, 28, 28)`。这一点已写进 `prep_utils.py` 的注释和测试用例，防止回归。

**遇到的问题②：确认两条独立路径产出一致。**
修正解析器前，脚本走的是降级路径（Keras 打包的 `mnist.npz`）。为了确认修正后
没有引入偏差，把两条路径各自生成的 `mnist_train.csv` 做了 SHA256 比对：

```
IDX 路径  : fb60bc58af4dac3554e394af262b3184479833d3cc540ff8783f274b73492d5d
npz 路径  : fb60bc58af4dac3554e394af262b3184479833d3cc540ff8783f274b73492d5d
```

逐字节一致。这既证明了 IDX 解析器正确，也说明降级方案是可用的。

**遇到的问题③：105 MB 的 CSV 不能整表载入。**
`pd.read_csv('mnist_train.csv')` 会为 60000×785 个元素分配 int64，约 380 MB；
用 `pd.DataFrame(data).to_csv()` 整表写同样吃内存。解决办法：
- **写**：每 5000 行建一个临时 DataFrame，写进 `StringIO` 再落盘（峰值内存 < 40 MB）；
- **校验**：用 `csv.reader` 流式扫描，只累计行数、标签计数、像素极值，
  不构造任何大数组（见 `validate_data._scan_mnist_csv`）。

### 5.2 波士顿房价（Boston Housing）

| 项 | 内容 |
|---|---|
| **来源** | UCI Machine Learning Repository · Housing，`https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data` |
| **获取日期** | 2026-10-27 |
| **原始文件名** | `housing.data`（47.9 KB）、`housing.names`（字段说明） |
| **原始格式** | **空白分隔**纯文本，**无表头**，506 行 × 14 个数值字段 |
| **转换方式** | `pd.read_csv(sep=r"\s+", header=None, names=[14 个已知列名])` → 直接落盘 |
| **最终行列数** | **506 × 14** |
| **列名** | `CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV` |
| **未做处理** | 不删行、不改列名、不做缺失值填充、不做异常值剔除（UCI 原表本身无缺失） |
| **缺失值** | 无 |
| **数值范围抽查** | `CHAS ∈ {0,1}`；`MEDV ∈ [5.0, 50.0]`（UCI 原表将 50 以上的房价截断在 50） |

**遇到的问题**：作业禁用 `sklearn.datasets.load_boston`，而多个公开 CSV 副本与 UCI
原文并不一致（有的删掉了 `B` 列、有的把 `CHAS` 转成字符串、有的行数不是 506）。
处理办法是**只认 UCI 官方 `housing.data`**，并在校验里做**逐值比对**：
把落盘 CSV 与 `raw/housing.data` 重新解析后逐元素相减，要求最大差异 `< 1e-9`。
只查形状和列名是不够的——错位一列时形状依然正确。

### 5.3 泰坦尼克号（Titanic）

| 项 | 内容 |
|---|---|
| **来源** | Kaggle 竞赛 *Titanic: Machine Learning from Disaster* 的 `train.csv` 原始快照，使用社区镜像 `datasciencedojo/datasets`（Kaggle 下载需登录，固定该镜像可保证同学间数据一致） |
| **获取日期** | 2026-10-27 |
| **原始文件名** | `titanic.csv`（58.9 KB） |
| **原始格式** | UTF-8 CSV，含表头，891 行 × 12 列 |
| **转换方式** | `pd.read_csv` → 校验列名与形状 → 原样落盘（仅重写编码、换行符与引号策略） |
| **最终行列数** | **891 × 12** |
| **列名** | `PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked` |
| **缺失值** | `Age` 177、`Cabin` 687、`Embarked` 2，**共 866 个单元格，全部保留为空字段** |
| **未做处理** | 不删行、不填充、不把 `Sex` 编码成 0/1、不裁剪列、不派生 `FamilySize` 等新特征 |

**缺失数为什么必须精确等于 177 / 687 / 2**：这三个数字是 Kaggle 原始训练集的指纹。
如果变成 0，说明有人做了填充；如果变大，说明混入了测试集或解析出错；
如果变小，说明删了行。校验脚本把这三个数字写成硬断言。

**遇到的问题：`Name` 字段含逗号。**
首行 `Braund, Mr. Owen Harris` 里有一个逗号。若写出时不加引号，读回来这一行会被
切成 13 列；而如果用 `line.count(',')` 去校验列数，又会把**合法的**带引号字段误判
成多列（我第一版校验就踩了这个坑）。正确做法是：写出用 `QUOTE_MINIMAL` 自动加引号，
校验用 `csv.reader` 解析而不是数字符。

### 5.4 西瓜数据集 3.0

| 项 | 内容 |
|---|---|
| **来源** | 周志华《机器学习》（清华大学出版社，2016）**附录"西瓜数据集"3.0 版** |
| **获取日期** | 2026-10-27 |
| **原始格式** | 教材附录中的二维表格，**无官方下载地址**，故以代码内嵌常量形式转录（`watermelon3_data.py`） |
| **转换方式** | Python 元组常量 → `pd.DataFrame` → 显式指定 dtype → 落盘 |
| **最终行列数** | **17 × 10** |
| **字段** | `编号, 色泽, 根蒂, 敲声, 纹理, 脐部, 触感, 密度, 含糖率, 好瓜` |
| **未做处理** | 6 个离散属性与标签 `好瓜` **保持中文字符串，未编码为数字**；密度、含糖率保留原始 3 位小数 |
| **缺失值** | 无 |
| **类别分布** | 好瓜 8 / 否 9；密度 `[0.243, 0.774]`、含糖率 `[0.042, 0.460]` |

**为什么内嵌而不用网上现成的 CSV**：网上流传的多个副本在字段名（"含糖率" vs
"含糖律"）、取值写法、小数位数上互不一致，直接引用会造成作业第 10 节点名的
"同学间数据不一致"。内嵌后由 `watermelon3_data._self_check()` 用硬断言锁死：
行数 17、编号 1–17 连续、每个离散字段的取值集合、类别分布 8/9、两列连续属性
的最小值与最大值。任何转录错误会在落盘之前就崩掉。

**版本说明（作业第 6 节要求"版本明确"）**：

| 版本 | 样本数 | 差异 |
|---|---|---|
| 2.0 | 17 | 只有 6 个离散属性，**无**密度与含糖率 |
| **3.0（本文件）** | 17 | 在 2.0 基础上**增加**密度、含糖率 |
| 4.0 | 17 | 离散属性取值改为三种（如纹理：清晰/稍糊/模糊），标签写法不同 |

若后续课程需要 4.0，**另存为 `watermelon4.0.csv` 并在 README 说明用途，不得覆盖本文件**。

---

## 六、异常处理（作业第 9 节）

要求"异常提示清晰，不因单个文件失败导致整个脚本中断"。实现分三层：

**第一层 · 明确的异常类型。** `prep_utils` 里定义了继承自 `DataPrepError` 的三个异常：
`DownloadError`（网络/HTTP）、`IntegrityError`（校验和不符、魔数错误、形状不符）、
`EncodingDetectionError`（编码全部失败）。抛出的消息都带**文件名、期望值、实际值**，
不使用裸 `Exception`。

**第二层 · 单点失败不影响整体。** `prepare_data.run()` 对每个数据集独立 `try/except`，
捕获后打印异常类型、消息和三层堆栈，写入报告，然后**继续处理下一个数据集**。
最后输出汇总表：成功几个、失败几个、失败原因。

**第三层 · 可复现的替代方案。**

| 异常场景 | 处理方式 |
|---|---|
| 文件不存在 | `FileNotFoundError`，提示完整路径 |
| 文件为 0 字节 | `DataPrepError`，提示"文件为空（0 字节）" |
| 编码错误 | 按 `utf-8 → utf-8-sig → gbk → gb18030` 依次尝试，全失败才报错并列出试过的编码 |
| 下载失败 | 重试 3 次，指数退避 2s/4s/8s；多地址时逐个尝试，最终报错列出**每个地址的失败原因** |
| 下载内容过小 | 视为"被重定向到错误页"，删掉临时文件并重试 |
| 列数不一致 | `IntegrityError`，指出解析出几列、期望几列 |
| IDX 魔数错误 | 指出文件、期望魔数、实际魔数 |
| IDX 数据长度不符 | 指出期望字节数与实际字节数 |
| MNIST 主源不可达 | **自动降级**到 Keras 打包的 `mnist.npz`，并在报告中注明"降级来源" |
| 写盘中途失败 | 原子写入（先写 `.tmp` 再 `os.replace`），不会留下半截 CSV |

**实测记录**：首次运行时 `t10k-images-idx3-ubyte.gz` 出现一次读超时，
脚本按退避重试后成功（日志见终端输出 `[重试 1/2] ... 2s 后重试`）；
泰坦尼克号镜像也出现过一次 `SSL: UNEXPECTED_EOF_WHILE_READING`，同样重试成功。
说明重试机制不是摆设。

---

## 七、校验与单元测试（作业第 7、8、12 节）

### 7.1 校验脚本 `validate_data.py`

两层，共 **55 项**，当前 **55/55 全部通过**：

| 数据集 | 通用规范检查 | 业务语义检查 | 合计 |
|---|---|---|---|
| 西瓜数据集 3.0 | 7 | 7 | **14** |
| 波士顿房价 | 7 | 6 | **13** |
| 泰坦尼克号 | 7 | 7 | **14** |
| MNIST | 7 | 7 | **14** |
| **合计** | | | **55** |

**通用规范检查**（7 项，每个 CSV 都跑）：文件存在且非空、UTF-8 无 BOM、
逗号分隔且每行字段数一致、无 `nan` 字面量、无多余索引列、仅必要字段加引号、
换行符统一为 LF。

**业务语义检查**（每个数据集 6–7 项），举几个关键的：

- MNIST：形状 60000×785 / 10000×785；标签在 0–9 且 10 类齐全；
  **像素最大值必须等于 255**（若为 1 说明被归一化了，违反第 3 节）；
  **与 `raw/` 原始 IDX 文件逐像素、逐标签比对 60000 行**。
- 波士顿：14 个列名逐字比对；首行 `CRIM=0.00632 / MEDV=24.0`；
  **与 `raw/housing.data` 逐值比对，最大差异 < 1e-9**。
- 泰坦尼克号：缺失数必须精确等于 `Age 177 / Cabin 687 / Embarked 2`；
  `Sex` 只能是 `male/female`；`Survived` 正类 342 人；首行 `Name` 必须解析回
  `"Braund, Mr. Owen Harris"`；**与 `raw/titanic.csv` 逐单元格比对**。
- 西瓜 3.0：9 个业务字段齐全；6 个离散字段取值落在合法集合内且**非数值 dtype**；
  类别分布 8/9；**与内嵌教材原表逐行逐字段比对**。

### 7.2 pytest `test_data.py`

**82 个用例全部通过**：

- **51 个** 直接把 `validate_data` 里的检查函数逐条参数化（一处定义，两处使用，
  避免脚本和测试的规则慢慢分叉）；
- **1 个** 断言"每个数据集至少 5 条校验"（第 8 节原文要求）；
- **1 个** 断言"整体校验全绿"；
- **2 个** 断言所有 CSV 都能用 `pd.read_csv` **默认参数**读取；
- **~21 个** 健壮性测试，人为构造异常输入：文件不存在 / 0 字节 / 无法解码 /
  GBK 自动识别 / 下载失败 / 多地址全失败 / IDX 魔数错误 / IDX 长度不足 /
  IDX 正常往返 / 缺失值写成空字段 / 无表头无索引 / 含逗号字段加引号 /
  原子写入无残留 / SHA256 稳定 / 与 manifest 记录的哈希一致；
- **6 个** 数据集专属抽查断言。

```bash
$ python -m pytest test_data.py -q
82 passed in 33.48s
```

---

## 八、数据来源与版本记录（作业第 10 节）

| 数据集 | 来源 | 版本 | 获取日期 | 原始文件 |
|---|---|---|---|---|
| MNIST | [MNIST 官方](http://yann.lecun.com/exdb/mnist/)，镜像 `ossci-datasets.s3.amazonaws.com/mnist/` | IDX v3，60000/10000 标准划分 | 2026-10-27 | `train-images-idx3-ubyte.gz` 等 4 个 |
| 波士顿房价 | [UCI ML Repo · Housing](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data) | UCI Housing，506 条 | 2026-10-27 | `housing.data`、`housing.names` |
| 泰坦尼克号 | Kaggle *Titanic* 竞赛 `train.csv`，镜像 `datasciencedojo/datasets` | 891 行 × 12 列原始快照 | 2026-10-27 | `titanic.csv` |
| 西瓜数据集 3.0 | 《机器学习》周志华，清华大学出版社 2016，**附录表格** | **3.0**（17 样本） | 2026-10-27 | 无（教材附录，内嵌常量） |

**泰坦尼克号固定使用同一来源**：Kaggle 官方下载需要登录，直接给链接无法保证
同学间拿到同一份文件。本项目固定使用 `datasciencedojo/datasets` 仓库的
`titanic.csv`，并在校验中用 `Age/Cabin/Embarked` 的缺失数 177/687/2 与
总行数 891 作为指纹，任何来源不一致都会立刻被发现。

### 落盘文件的校验和（SHA256）

| 文件 | 形状 | 大小 | SHA256 |
|---|---|---|---|
| `watermelon3.0.csv` | 17 × 10 | 1.1 KB | `29d03effe207088a414894e11f1173a2d69a318cae6bbb00b69f046fe68f9427` |
| `boston_housing.csv` | 506 × 14 | 36.3 KB | `d0a8dba1d66346fb31288ad7f629f78c7a9709e5d8a2b0eff718aa352bad1b48` |
| `titanic_train.csv` | 891 × 12 | 60.5 KB | `8331e5a2532f5fdb66153a8fc053df08a7dc9d2584f246c7fb1752f0a7139ca5` |
| `mnist_train.csv` | 60000 × 785 | 104.5 MB | `fb60bc58af4dac3554e394af262b3184479833d3cc540ff8783f274b73492d5d` |
| `mnist_test.csv` | 10000 × 785 | 17.4 MB | `51c292478d94ec3a01461bdfa82eb0885d262eb09e615679b2d69dedb6ad09e7` |

完整信息见 `results/manifest.json`。若重跑后发现某个 SHA256 变了，说明数据被
改过或来源变了，应当排查而不是直接提交。

---

## 九、作业条目与代码位置对照

| 作业条目 | 实现位置 | 验收证据 |
|---|---|---|
| 1 工程环境与目录规范 | 本目录 | `data/`、`prepare_data.py`、`README.md` 齐备，脚本可运行 |
| 2 CSV 统一规范 | `prep_utils.write_csv_standard()` | 55 项校验中 28 项属于规范检查，全绿 |
| 3 MNIST 准备 | `prepare_data.prepare_mnist()` + `prep_utils.read_idx_*` | 60000×785 / 10000×785，无表头无索引，未归一化 |
| 4 波士顿房价准备 | `prepare_data.prepare_boston()` | 506×14，14 列名逐字一致，未用 `load_boston` |
| 5 泰坦尼克号准备 | `prepare_data.prepare_titanic()` | 891×12，缺失 177/687/2 原样保留，`Sex` 未编码 |
| 6 西瓜数据集准备 | `watermelon3_data.py` + `prepare_data.prepare_watermelon()` | 17×10，9 字段齐全，离散值未编码，版本 3.0 明确 |
| 7 数据获取与转换脚本 | `prepare_data.py`（全流程） | 每份数据都能说明来源与转换过程，可重复运行 |
| 8 数据校验与读取测试 | `validate_data.py` | 输出每个数据集的行数、特征数、缺失情况，55 项全过 |
| 9 README 说明文件 | 本文件 | 含来源、日期、格式、转换、形状、缺失、问题与解决 |
| 10 数据来源与版本记录 | 本文件第八节 + `results/manifest.json` | 来源可查、版本明确、含 SHA256 |
| 11 异常情况处理 | `prep_utils` + `prepare_data.run()` | 单点失败不中断，重试与降级有实测记录 |
| 12 自查与单元校验 | `test_data.py` | 82 个用例全过，每数据集 ≥ 5 条断言 |
| 13 提交与复现 | `data/` + 脚本 + git | 按第三节步骤可完整复现 |

---

## 十、已知限制

1. **MNIST 的 CSV 体积很大**（104.5 MB + 17.4 MB）。这是"每人一行、785 列"这种
   扁平格式的必然结果，不是实现问题。若要传输方便可另存 `.npz`，但作业要求 CSV。
2. **波士顿房价的 `MEDV` 上限被截断在 50**。这是 UCI 原始数据的特性（房价超过
   5 万美元的样本统一记为 50），后续建模若把 `MEDV` 当连续目标需注意这个删失。
3. **`raw/mnist.npz` 是备用来源**，正常路径不会用到；保留它是为了在主源不可达时
   仍能复现，不算冗余。
4. **西瓜数据集是教材附录转录**，若老师要求必须从某个具体仓库获取，请替换
   `watermelon3_data.py` 中的常量并更新第八节来源，其余流程不用改。

---

## 十一、独立完成声明

> 本实践的数据收集、转换脚本、校验脚本、单元测试与本说明文档，
> 均由本人独立完成。除课程讲义与下列公开文档外，未参考他人提交的代码：
>
> - Python 官方文档 `csv` / `gzip` / `urllib` 模块
> - pandas 官方文档 `read_csv` / `to_csv`
> - MNIST 官方数据集格式说明（IDX 文件结构）
> - UCI Machine Learning Repository 的 `housing.names` 字段说明
>
> 数据来源均为公开数据集，已在第八节列明链接与获取日期。
>
> 学号：2025111142　姓名：陈锦煌　日期：2026-10-27
