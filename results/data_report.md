# 数据准备运行报告

- 生成时间：2026-10-27T15:28:20
- 数据获取日期：2026-10-27
- 运行环境：Python 3.13.14 / pandas 3.1.0 / numpy 2.5.3

## 总览

| 数据集 | 文件 | 行 | 列 | 缺失单元格 | 大小 | 状态 |
|---|---|---|---|---|---|---|
| 西瓜数据集 3.0 | `watermelon3.0.csv` | 17 | 10 | 0 | 1.1 KB | 成功 |
| 波士顿房价 | `boston_housing.csv` | 506 | 14 | 0 | 36.3 KB | 成功 |
| 泰坦尼克号 | `titanic_train.csv` | 891 | 12 | 866 | 60.5 KB | 成功 |
| MNIST | `mnist_train.csv` | 60000 | 785 | 0 | 104.5 MB | 成功 |

## 明细

### 西瓜数据集 3.0（`watermelon`）

- 来源：《机器学习》（周志华，清华大学出版社，2016）附录 西瓜数据集 3.0
- 实际使用地址：`（教材附录表格，无公开下载链接，故以代码内嵌常量形式落盘）`
- 版本：3.0（17 个样本；区别于 2.0 的 17 样本无密度/含糖率、4.0 的 17 样本含纹理）
- 获取日期：2026-10-27
- 形状：17 行 × 10 列
- 缺失值：无
- SHA256：`29d03effe207088a414894e11f1173a2d69a318cae6bbb00b69f046fe68f9427`
- 耗时：0.01 s

### 波士顿房价（`boston`）

- 来源：UCI Machine Learning Repository · Housing
- 实际使用地址：`https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data`
- 版本：UCI Housing（506 条，1993 年快照，未做任何删改）
- 获取日期：2026-10-27
- 形状：506 行 × 14 列
- 缺失值：无
- SHA256：`d0a8dba1d66346fb31288ad7f629f78c7a9709e5d8a2b0eff718aa352bad1b48`
- 耗时：0.02 s

### 泰坦尼克号（`titanic`）

- 来源：Kaggle · Titanic: Machine Learning from Disaster（社区镜像 datasciencedojo/datasets）
- 实际使用地址：`https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`
- 版本：Kaggle train.csv 原始快照（891 行 × 12 列）
- 获取日期：2026-10-27
- 形状：891 行 × 12 列
- 缺失值：Age 177、Cabin 687、Embarked 2（共 866 个单元格，保留为空字段）
- SHA256：`8331e5a2532f5fdb66153a8fc053df08a7dc9d2584f246c7fb1752f0a7139ca5`
- 耗时：0.01 s

### MNIST（`mnist`）

- 来源：MNIST 官方 IDX 文件（LeCun 等）· S3 镜像
- 实际使用地址：`https://ossci-datasets.s3.amazonaws.com/mnist/`
- 版本：原始 IDX v3（60000 / 10000 标准划分）
- 获取日期：2026-10-27
- 形状：60000 行 × 785 列
- 缺失值：无
- 标签分布：训练 [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
　　　　　　测试 [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
- 像素取值范围：0–255（未归一化）
- SHA256：`fb60bc58af4dac3554e394af262b3184479833d3cc540ff8783f274b73492d5d`
- 附加文件 `mnist_test.csv`：形状 [10000, 785]，SHA256 `51c292478d94ec3a01461bdfa82eb0885d262eb09e615679b2d69dedb6ad09e7`
- 耗时：3.85 s
