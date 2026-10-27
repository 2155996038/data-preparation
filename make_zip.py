"""打包提交物：``<本目录名>.zip``。

作业第 13 节要求提交"学号_姓名_数据准备.zip"。把目录改名成
``20231234_张三_数据准备`` 之后运行本脚本，生成的压缩包会自动叫
``20231234_张三_数据准备.zip``，不用手工改名、也不会漏文件。

自动排除以下内容：
    .git/                版本库元数据，交了也没意义还会撑大体积
    raw/                 22 MB 原始下载物，可由 prepare_data.py 重新获取
    __pycache__/ 等缓存   运行时产物
    *.pyc / *.tmp / *.part   临时文件

用法：
    python make_zip.py
"""
import os
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE.parent / (BASE.name + ".zip")

SKIP_DIRS = {".git", "raw", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
SKIP_SUFFIX = {".pyc", ".pyo", ".tmp", ".part"}

files = []
for root, dirs, names in os.walk(BASE):
    dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
    for name in sorted(names):
        if Path(name).suffix in SKIP_SUFFIX:
            continue
        p = Path(root) / name
        files.append(p.relative_to(BASE))

if OUT.exists():
    OUT.unlink()

with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED,
                     compresslevel=6) as zf:
    for rel in files:
        zf.write(BASE / rel, arcname=str(Path(BASE.name) / rel))

raw_size = sum((BASE / r).stat().st_size for r in files)
print(f"打包 {len(files)} 个文件 -> {OUT}")
print(f"原始大小 {raw_size / 1024 / 1024:.1f} MB，"
      f"压缩后 {OUT.stat().st_size / 1024 / 1024:.1f} MB")
print("清单：")
for rel in files:
    print(f"  {rel}  ({(BASE / rel).stat().st_size / 1024:.1f} KB)")
