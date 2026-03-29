# 依赖追踪裁剪分析报告

## 分析概要

- **原始依赖数量**: 194 个
- **核心必需依赖**: 约 12 个直接导入的包
- **精简后依赖数量**: 约 34 个（含必要的传递依赖）
- **可移除僵尸依赖**: 约 160 个

---

## 一、代码实际使用的核心依赖

### 深度学习框架
| 包名 | 用途 | 代码文件 |
|------|------|---------|
| torch | 神经网络训练 | reg_v1.py, cls.py, pretrain/*.py |
| transformers | 预训练模型加载 | reg_v1.py, cls.py, pretrain/*.py |
| datasets | 数据集加载 | reg_v1.py, cls.py, pretrain/*.py |
| tokenizers | 文本分词 | cls.py, pretrain/*.py |

### 数据处理
| 包名 | 用途 | 代码文件 |
|------|------|---------|
| pandas | 数据表处理 | reg_v1.py, cls.py, pretrain/*.py |
| numpy | 数值计算 | reg_v1.py, cls.py |
| matplotlib | 绘图 | reg_v1.py |
| scikit-learn | 评估指标、TSNE | reg_v1.py, cls.py |

### 超参数调优与日志
| 包名 | 用途 | 代码文件 |
|------|------|---------|
| ray | 分布式调参 | reg_v1.py |
| wandb | 实验跟踪 | cls.py |
| tensorboardX | 训练可视化 | reg_v1.py |
| tqdm | 进度条 | 所有文件 |

---

## 二、僵尸依赖清单（可安全移除）

以下 160 个依赖在代码中未发现任何 import 语句：

### 机器学习框架
- FLAML, bertviz, hyperopt, optuna, alembic, sentencepiece

### Web & API 相关
- fastapi, uvicorn, starlette, pydantic, grpcio, h11, websockets
- gradio, gradio_client, altair, dash, plotly

### 云服务
- boto3, botocore, s3transfer, google-auth, google-auth-oauthlib

### 实验跟踪
- mlflow, weights, tensorboard (本项目使用 tensorboardX)

### 图像处理
- opencv-python, Pillow, imageio, scikit-image

### Notebook 支持
- ipykernel, ipython, ipywidgets, jupyter_client, jupyter_core

### 开发工具
- ruff, sentry-sdk, loguru, typer, pyperclip

### 其他
- aiofiles, aiohttp, aiosignal, async-timeout, certifi, charset-normalizer
- cloudpickle, dill, distro, et-xmlfile, ffmpy, fonttools, frozenlist
- fsspec, future, gitdb, GitPython, greenlet, importlib-metadata
- jedi, jmespath, jsonschema, kiwisolver, markdown-it-py, mdurl
- mpmath, msgpack, multidict, nest-asyncio, networkx, oauthlib, openai
- orjson, packaging, parso, pathtools, pbr, pickleshare, platformdirs
- prettytable, prompt-toolkit, protobuf, psutil, pure-eval, py-cpuinfo
- py4j, pyarrow, pyasn1, pyasn1-modules, pydub, Pygments, pyparsing
- pyreadline3, python-dateutil, python-multipart, pytz, pywin32, PyYAML
- pyzmq, referencing, requests, requests-oauthlib, responses, rich
- rpds-py, rsa, semantic-version, seqeval, setproctitle, shellingham
- six, smmap, sniffio, SQLAlchemy, stack-data, sympy, tensorboard-data-server
- thop, torchinfo, torchviz, tornado, traitlets, typing_extensions, tzdata
- ultralytics, wcwidth, widgetsnbextension, win32-setctime, xlrd, xxhash
- yarl, zipp, absl-py, accelerate, annotated-types, anyio, appdirs, asttokens
- autopage, backcall, colorlog, comm, contourpy, cycler, debugpy, decorator
- exceptiongroup, executing, fastcore, jsonschema-specifications

---

## 三、精简后的 requirements（推荐）

```
# Core Deep Learning & NLP
torch>=1.13.0
transformers>=4.34.0
datasets>=2.14.0
tokenizers>=0.14.0
huggingface-hub>=0.20.0

# Data Processing
numpy>=1.26.0
pandas>=2.1.0
matplotlib>=3.8.0
scikit-learn>=1.3.0
scipy>=1.11.0

# Progress & Logging
tqdm>=4.66.0
tensorboardX>=2.6.0
wandb>=0.15.0

# Ray for Hyperparameter Tuning
ray>=2.7.0
click>=8.1.0
colorama>=0.4.0
stevedore>=5.0.0
cmd2>=2.4.0
cliff>=4.4.0
prettytable>=3.9.0
pyyaml>=6.0.0

# Utilities
fsspec>=2023.6.0
safetensors>=0.3.0
regex>=2023.8.0
filelock>=3.12.0
```

---

## 四、验证方法

使用以下命令验证精简后的依赖是否满足需求：

```bash
# 导出当前环境的所有依赖
pip freeze > original_requirements.txt

# 创建精简 requirements
# (见上方第三部分)

# 在新环境中安装并测试
pip install -r requirements_minimal.txt
python -c "import torch; import transformers; import datasets; ..."
```

---

## 五、注意事项

1. **ray 依赖链**: ray 本身需要多个传递依赖（click, colorama 等），这些已包含在精简列表中
2. **seaborn**: 虽然代码中被注释掉 `import seaborn`，但保留可能用于数据可视化
3. **protobuf**: transformers 旧版本可能需要，新版本已内置
4. **openpyxl**: 如果需要读取 `.xlsx` 文件，需添加 `openpyxl`

---

生成时间: 2026-03-29
