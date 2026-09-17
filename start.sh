#!/usr/bin/env bash
# 命令失败、未定义变量或管道失败时立即退出，避免带着错误环境继续运行。
set -euo pipefail
# 无论从哪里调用，始终以脚本所在目录作为项目根目录。
cd "$(dirname "$0")"
# 将依赖下载缓存留在项目内，便于隔离环境。
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
# 首次运行时创建 Python 3.11 虚拟环境；后续复用。
if [ ! -x .venv/bin/python ]; then
  uv venv .venv --python 3.11
fi
# 安装固定版本依赖；此脚本是可选 Web 入口。
uv pip install --python .venv/bin/python -r requirements.txt
# exec 将当前 shell 替换为应用进程，并透传端口、设备等参数。
exec .venv/bin/python app.py "$@"
