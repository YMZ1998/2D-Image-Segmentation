# Pixi 使用文档

适用于 Windows + Python/AI 项目，重点覆盖 Pixi 环境、Conda/PyPI 依赖、PyCharm、锁文件以及本项目 `2D-Image-Segmentation` 的常用操作。

## 1. Pixi 是什么

Pixi 是项目级环境与依赖管理工具，可以管理：

- Python 版本
- conda-forge 依赖
- PyPI 依赖
- 项目任务（tasks）
- 锁文件 `pixi.lock`
- 项目独立环境 `.pixi/`

推荐项目结构：

```text
2D-Image-Segmentation/
├─ apps/
├─ scripts/
├─ models/
├─ pyproject.toml
├─ pixi.lock
├─ .pixi/
└─ .gitignore
```

其中 `.pixi/` 不提交 Git，`pixi.lock` 建议提交 Git。

## 2. 安装

Windows PowerShell：

```powershell
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

检查：

```bat
pixi --version
```

## 3. 初始化项目

```bat
cd /d D:\Code\2D-Image-Segmentation
pixi init
```

Python 项目也可以：

```bat
pixi init --format pyproject
```

如果已有 `pyproject.toml`，先检查现有 `[project]` 和 `[tool.pixi]` 配置，避免重复初始化。

## 4. 添加 Python

例如 Python 3.10：

```bat
pixi add python=3.10
```

或：

```bat
pixi add "python>=3.10,<3.11"
```

检查：

```bat
pixi run python --version
pixi run python -c "import sys; print(sys.executable)"
```

Windows 下通常类似：

```text
D:\Code\2D-Image-Segmentation\.pixi\envs\default\python.exe
```

## 5. 添加 Conda 包

```bat
pixi add numpy scipy opencv
```

指定版本：

```bat
pixi add "numpy>=1.26,<2"
```

查看：

```bat
pixi list
pixi tree
```

## 6. 添加 PyPI 包

PyQt5：

```bat
pixi add --pypi PyQt5
```

Pillow：

```bat
pixi add --pypi Pillow
```

ONNX Runtime：

```bat
pixi add --pypi onnxruntime
```

固定版本：

```bat
pixi add --pypi "onnxruntime==1.23.2"
```

检查：

```bat
pixi run python -c "import onnxruntime; print(onnxruntime.__version__)"
```

一般原则：

| 依赖 | 推荐 |
|---|---|
| Python | `pixi add python=3.10` |
| numpy/scipy/opencv | `pixi add` |
| PyQt5 | `pixi add --pypi` |
| Pillow | `pixi add --pypi` |
| ONNX Runtime | `pixi add --pypi` |
| TensorRT | 根据 NVIDIA 官方包/环境单独处理 |

## 7. PyPI 镜像

例如清华镜像：

```toml
[tool.pixi.pypi-options]
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"
```

注意 NVIDIA/TensorRT 等包不一定能通过国内镜像正常获得。

## 8. 运行项目

推荐：

```bat
pixi run python apps/inference_viewer.py
```

检查当前 Python：

```bat
pixi run python -c "import sys; print(sys.executable)"
```

进入环境：

```bat
pixi shell
```

退出：

```bat
exit
```

`pixi run` 会在项目环境中执行命令，因此比直接输入 `python` 更不容易误用系统 Python。

## 9. Pixi Task

在 `pyproject.toml` 中：

```toml
[tool.pixi.tasks]
viewer = "python apps/inference_viewer.py"
train = "python train.py"
export-onnx = "python convert_onnx.py"
```

运行：

```bat
pixi run viewer
pixi run train
pixi run export-onnx
```

## 10. 安装环境

新电脑拉取代码后：

```bat
git clone <repository>
cd <repository>
pixi install
```

如果已有 `pixi.lock`，Pixi 会按锁文件恢复环境。

严格按锁文件：

```bat
pixi install --frozen
```

或：

```bat
pixi install --locked
```

## 11. pixi.toml / pyproject.toml / pixi.lock

可以理解为：

```text
pyproject.toml / pixi.toml
        ↓
   声明需要什么
        ↓
    pixi.lock
        ↓
  锁定具体版本
        ↓
     .pixi/
        ↓
   本机实际环境
```

建议 Git 提交：

```text
pyproject.toml
pixi.lock
.gitignore
```

`.gitignore`：

```gitignore
.pixi/
```

## 12. 查看和维护环境

查看：

```bat
pixi list
pixi tree
```

删除：

```bat
pixi remove numpy
pixi remove --pypi Pillow
```

更新：

```bat
pixi update
```

指定包：

```bat
pixi update numpy
```

预览：

```bat
pixi update --dry-run
```

生产项目不要随意全量更新依赖。

## 13. PyCharm 使用 Pixi

获取解释器路径：

```bat
pixi run python -c "import sys; print(sys.executable)"
```

例如：

```text
D:\Code\2D-Image-Segmentation\.pixi\envs\default\python.exe
```

PyCharm：

```text
Settings
  → Project
  → Python Interpreter
  → Add Interpreter
  → Add Local Interpreter
  → Existing
```

选择：

```text
.pixi\envs\default\python.exe
```

不要把 `pixi.exe` 当成 Python Interpreter。

## 14. requirements.txt 与 Pixi

传统方式：

```text
requirements.txt
      ↓
pip install -r requirements.txt
```

Pixi：

```text
pyproject.toml / pixi.toml
      ↓
pixi.lock
      ↓
pixi install
```

已有项目可以逐步迁移，不必一次全部修改。

## 15. Python 版本问题

如果配置：

```toml
requires-python = ">=3.14"
```

但项目实际使用 Python 3.10，可能导致依赖解析失败。

例如 Python 3.10 项目：

```toml
requires-python = ">=3.10,<3.11"
```

遇到：

```text
No solution found
has no wheels with a matching Python ABI tag
```

优先检查：

```bat
pixi run python --version
```

然后检查目标包是否提供当前 Python/Windows 对应 wheel。

## 16. 项目自身被当作 PyPI 包

如果出现：

```text
2d-image-segmentation @ file:///D:/Code/2D-Image-Segmentation
```

检查是否配置了类似：

```toml
[tool.pixi.pypi-dependencies]
2d-image-segmentation = { path = ".", editable = true }
```

如果项目只是 GUI/应用程序，而不是需要被其他项目安装的 Python library，通常不需要把自身作为 PyPI 依赖。

## 17. TensorRT 注意事项

TensorRT 同时涉及：

```text
TensorRT
CUDA
cuDNN
NVIDIA Driver
Python bindings
native libraries
```

因此不建议把 TensorRT 当普通 Python 包处理。

如果直接：

```bat
pixi add --pypi tensorrt
```

出现 wheel、构建或 `wheel_stub` 相关问题，可以让 Pixi 管理基础环境，再使用 pip：

```bat
pixi add --pypi pip
pixi run python -m pip install tensorrt==11.3.0.99
```

检查：

```bat
pixi run python -c "import tensorrt as trt; print(trt.__version__)"
```

同时建议记录：

```text
Python
PyTorch
CUDA
cuDNN
TensorRT
NVIDIA Driver
GPU
```

## 18. 常用检查命令

Python：

```bat
pixi run python --version
pixi run python -c "import sys; print(sys.executable)"
```

Pillow：

```bat
pixi run python -c "from PIL import Image; print(Image); print(Image.__file__)"
```

ONNX Runtime：

```bat
pixi run python -c "import onnxruntime; print(onnxruntime.__version__)"
```

PyTorch：

```bat
pixi run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

TensorRT：

```bat
pixi run python -c "import tensorrt as trt; print(trt.__version__)"
```

## 19. 本项目推荐工作流

```bat
cd /d D:\Code\2D-Image-Segmentation
```

安装：

```bat
pixi install
```

检查：

```bat
pixi run python --version
pixi run python -c "import sys; print(sys.executable)"
```

运行 GUI：

```bat
pixi run python apps/inference_viewer.py
```

以后可以配置：

```toml
[tool.pixi.tasks]
viewer = "python apps/inference_viewer.py"
```

然后：

```bat
pixi run viewer
```

## 20. 日常速查

| 操作 | 命令 |
|---|---|
| 初始化 | `pixi init` |
| 添加 Python | `pixi add python=3.10` |
| 添加 Conda 包 | `pixi add numpy` |
| 添加 PyPI 包 | `pixi add --pypi Pillow` |
| 安装 | `pixi install` |
| 查看包 | `pixi list` |
| 依赖树 | `pixi tree` |
| 更新 | `pixi update` |
| 删除 | `pixi remove numpy` |
| 进入环境 | `pixi shell` |
| 运行命令 | `pixi run xxx` |
| Python | `pixi run python` |
| Python 路径 | `pixi run python -c "import sys; print(sys.executable)"` |
| 锁定安装 | `pixi install --frozen` |
| Pixi 版本 | `pixi --version` |

## 21. 推荐原则

1. 项目依赖统一写入 Pixi 配置。
2. `pixi.lock` 提交 Git。
3. `.pixi/` 不提交 Git。
4. 项目脚本优先使用 `pixi run`。
5. 明确限制 Python 版本。
6. Conda 包和 PyPI 包按实际可用性选择。
7. CUDA/TensorRT 单独记录版本和 GPU 环境。
8. 稳定项目不要随意全量 `pixi update`。
9. PyCharm 使用 `.pixi\envs\default\python.exe`。
10. 新电脑优先执行 `pixi install`，不要手工重新搭建环境。

## 22. 官方文档

Pixi：

https://pixi.sh/

CLI：

https://pixi.sh/latest/reference/cli/

主要命令：

- `pixi init`
- `pixi add`
- `pixi install`
- `pixi update`
- `pixi remove`
- `pixi run`
- `pixi shell`
- `pixi list`
- `pixi tree`
