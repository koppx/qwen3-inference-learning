# Qwen3：纯 Python 调试入口

当前推荐使用 **`debug_official.py` + `learning.md`**。不需要启动 Web。

```bash
.venv/bin/python debug_official.py
```

- [完整调用链与断点指南](learning.md)
- [直接调用官方模型的 Python 脚本](debug_official.py)
- 解释器选择项目 `.venv/bin/python`；在 `model.generate(...)` 和官方 Decoder / Attention 中打断点。
- 模型权重不随仓库上传；纯调试所需依赖见 `requirements-debug.txt`。

## 首次克隆后运行

需要 Python 3.11 和 uv。在仓库根目录执行：

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements-debug.txt
```

将 `debug_official.py` 中的 `LOCAL_FILES_ONLY` 改为 `False`，然后运行：

```bash
.venv/bin/python debug_official.py
```

首次运行会下载官方 Qwen3-0.6B 权重到 `.cache/huggingface`；下载完成后可将 `LOCAL_FILES_ONLY` 改回 `True` 离线运行。`.venv`、模型缓存和 `outputs` 运行记录均不纳入版本管理。

---

以下保留之前的可选 Web 观察器说明。

# Qwen3 推理观察室

运行 **官方 Qwen/Qwen3-0.6B**，从输入、分词、`generate()` 一直观察到 Transformers `modeling_qwen3.py` 中的实际计算，再回看每一步对应的源码和张量形状。

## 启动

需要 `uv`（本机已安装）。在本目录执行：

```bash
bash start.sh
```

浏览器打开 http://127.0.0.1:7860 。首次点击“运行官方模型并记录”下载约 1.2 GB 权重，终端会显示下载进度。模型和环境放在项目目录下。自动选择 CUDA → Apple MPS → CPU；CPU 使用 FP32，其他设备使用 FP16。建议预留至少 4 GB 可用内存。

已安装环境时可直接启动，不再检查依赖：

```bash
.venv/bin/python app.py
# 指定设备 / 端口
.venv/bin/python app.py --device cpu --port 7861
```

固定使用 Transformers **4.56.2**，而不是不断变化的 GitHub main。页面显示的是实际安装、实际执行版本的源码，含本机路径与行号。

## 第一次怎样看

1. 保留默认第 0 层，生成 4～8 个 token，点击运行。
2. 在“输入与下一个 token”里查看聊天模板的特殊 token、输入 ID 和每步 Top 5。
3. 在调用表点击 `Qwen3ForCausalLM`，看生成循环如何进入模型。
4. 搜索 `model.layers.0`，逐步观察 RMSNorm、Attention、残差、MLP。
5. 搜索 `eager_attention_forward`，点击“即将执行”事件。高亮行尚未执行，右边是执行前的变量；下一步会显示更新后的变量。
6. 切到 Decode，比较 Q/K/V 的序列维度。启用 KV Cache 时，后续 forward 输入通常只有 1 个 token，但注意力看到的 K/V 历史在增长。
7. 取消 KV Cache 重新运行，观察输入长度逐次增加。关闭缓存后表格仍把第一轮之后称为 Decode（生成阶段），但实际每轮重算完整前缀。

张量的常见形状：`[batch, sequence, hidden]`；拆头之后为 `[batch, heads, sequence, head_dim]`。Qwen3-0.6B 有 28 个 Decoder，Q 有 16 个头，KV 有 8 个头。

## 文件入口

- `trace_model.py`：加载官方模型、注册观察 hooks、行跟踪、实际 `generate()` 调用、JSON 导出；核心入口是 `Inspector.run()`。
- `app.py`：本地 Gradio 页面、事件筛选、源码回放。
- `debug_official.py`：没有行跟踪器的最小官方推理，适合 IDE 断点。
- `test_trace.py`：用微型随机权重的官方 Qwen3 测试观察器不改变输出、缓存行为、异常清理。这不是实际模型效果验收。
- `outputs/trace-*.json`：本次完整执行记录，包含实际源码快照。

仅使用命令行：

```bash
.venv/bin/python trace_model.py --prompt '1加1等于几？' --tokens 8 --layer 0
.venv/bin/python trace_model.py --prompt '1加1等于几？' --tokens 8 --no-cache
```

## 真正用断点暂停

在 IDE 里选择 `.venv/bin/python`，运行 `debug_official.py`，给 `engine.model.generate(...)` 那行加断点。VS Code 调试配置设置 `"justMyCode": false`；PyCharm 允许步入库代码。

也可以直接在以下官方源文件的 `Qwen3Attention.forward`、`Qwen3DecoderLayer.forward`、`eager_attention_forward` 打断点：

```bash
.venv/bin/python -c 'import inspect; from transformers.models.qwen3 import modeling_qwen3; print(inspect.getsourcefile(modeling_qwen3))'
```

不要用 IDE 调试 `trace_model.py` 的逐行记录流程：调试器和本项目都会使用 `sys.settrace`；`debug_official.py` 专门避免这一冲突。

## 跟踪范围与取舍

- 模型权重、前向计算和生成循环来自官方 Transformers；本项目不手写替代模型，也不修改第三方源码。
- 所有 Decoder 都记录模块边界；选定层的子模块也记录边界。前两次 forward 深入选中层的 Python 源码，包括 RoPE、eager attention 和 Cache update。
- 使用 eager attention，便于看到 QKᵀ、mask、Softmax、V 的乘法。这里的教学时延不能当作优化后推理速度。
- 只保存张量 shape/dtype/device 与少量标量，不存完整激活，不追踪底层 Rust/C++/GPU 内核。
- 使用 `enable_thinking=False` 和贪心生成让调用易于复现。8 token 的结果可能是半句话，可增加到 32。
- 界面是运行完成后的回放，不是实时单步暂停；每次运行最多 256 输入 token / 32 输出 token，避免教学记录过大。
- 当前只支持单请求、小型稠密 Qwen3；不是 vLLM 服务，也不包含训练流程。

## 验证

```bash
uv pip install --python .venv/bin/python 'pytest>=8,<9'
.venv/bin/python -m pytest -q
```

## 排查

- 下载失败：检查 Hugging Face 网络可达性；重新执行会复用项目缓存。不要把下载失败当作推理成功。
- 依赖冲突：使用项目 `.venv/bin/python`，不要使用系统 Python。
- GPU/MPS 错误：使用 `--device cpu`。
- 内存不足：关闭其他大程序，并缩短输入。模型本身仍需要几 GB 内存。
- 端口占用：使用 `--port 7861`。

参考：[模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)、[对应版本源码](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3/modeling_qwen3.py)。
