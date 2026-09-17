"""Local learning UI: real execution first, source-linked replay afterwards."""
import argparse
import html
import json
import threading

# 先导入本地模块设置 HF_HOME，再导入 Gradio，避免缓存指向其他目录。
from trace_model import Inspector, ROOT, save_result
import gradio as gr

# 全进程复用模型；LOCK 串行化推理，避免不同请求的 hooks 互相干扰。
ENGINE = Inspector()
LOCK = threading.Lock()

# 仅负责页面外观；深色区域显式设置前景色，避免被 Gradio 默认样式覆盖。
CSS = """
.gradio-container {max-width: 1450px !important; margin: auto;}
.hero {padding: 28px 32px; background: #142c38; border-radius: 18px; color: #effbf7; margin-bottom: 18px;}
.hero h1 {color: #effbf7; margin: 5px 0 10px; font-size: 32px;}
.hero p {color: #c6d7dd; margin: 0;}
.eyebrow {color: #6de4c3; font: 12px monospace; letter-spacing: 3px;}
.flow {display:flex; gap: 8px; flex-wrap:wrap; margin: 20px 0 0;}
.hero .flow span {color:#effbf7 !important; border:1px solid #48606a; padding:6px 12px; border-radius:6px; font:12px monospace;}
.source {background:#10202c; color:#cce2ec; border-radius:12px; overflow:auto; max-height:550px; padding:14px 0; font:13px/1.75 monospace;}
.source .line {display:block; white-space:pre; padding:0 14px; color:#cce2ec !important;}
.source .active {background:#244e49; color:#fff !important; border-left:3px solid #5ff0c3;}
.source .num {color:#7b99a8; display:inline-block; width:48px; user-select:none;}
"""


# 将一个事件转换成标题、源码片段和变量 JSON；不会重新执行模型。
def detail(result, index):
    if not result or not result.get("events"):
        return "运行后选择一个事件。", "", {}
    # 将回放索引限制在有效范围，避免首尾翻页时越界。
    event = result["events"][max(0, min(int(index), len(result["events"]) - 1))]
    filename, line = event["file"], event["line"]
    heading = f"**事件 {event['id']} · {event['phase']} · 第 {event['step'] + 1} 次 forward · {event['kind']}**\n\n`{event['name']}`"
    if filename:
        heading += f"\n\n本机源码：`{filename}:{line}`"
    if event["kind"] == "即将执行":
        heading += "\n\n当前高亮行**即将执行**；右侧是执行该行之前的局部变量。点下一步观察结果。"
    source = result.get("sources", {}).get(filename, "")
    if source and line:
        lines = source.splitlines()
        start, end = max(0, line - 5), min(len(lines), line + 18)
        # 源码以文本方式转义后放入 HTML，防止代码中的符号被当成网页标签。
        body = "".join(f'<span class="line {"active" if i + 1 == line else ""}"><span class="num">{i + 1}</span>{html.escape(lines[i])}</span>' for i in range(start, end))
        rendered = f'<div class="source">{body}</div>'
    else:
        rendered = '<p>此事件记录输入、输出或模块边界；选择“即将执行”事件查看源码行。</p>'
    return heading, rendered, event["values"]


# 筛选只改变表格显示；保留全局事件 ID 以便定位完整记录。
def event_rows(result, phase="全部", keyword=""):
    if not result:
        return []
    return [[e["id"], e["step"] + 1, e["phase"], e["kind"], e["name"], e["line"] or ""]
            for e in result["events"] if (phase == "全部" or phase == e["phase"])
            and keyword.lower() in e["name"].lower()]


# Gradio 生成器回调：先输出加载状态，再输出完整推理结果。
def run(prompt, tokens, layer, cache):
    # 第一次 yield 清空上一轮的展示和过滤条件，避免新旧结果混淆。
    yield ("正在加载官方 Qwen3-0.6B。首次运行需要下载约 1.2 GB 权重，进度显示在终端。",
           None, "", [], [], [], gr.update(value=0, maximum=1), None, {}, "全部", "")
    try:
        # 锁保护共享模型与观察器，当前请求结束后才允许下一个请求进入。
        with LOCK:
            result = ENGINE.run(prompt, int(tokens), int(layer), cache)
        file = save_result(result)
        rows = event_rows(result)
        input_rows = [[i, t["id"], t["text"], t["raw"]] for i, t in enumerate(result["input_tokens"])]
        # 把每一步的五个候选展开成表格行；概率来自真实生成分数。
        generated_rows = [[t["step"] + 1, c["id"], c["text"], c["probability"]]
                          for t in result["top_tokens"] for c in t["candidates"]]
        meta = {k: result[k] for k in ("model", "parameters", "device", "dtype", "transformers", "torch", "use_cache", "selected_layer")}
        meta["config"] = result["config"]
        status = (f"已完成 · {len(result['input_tokens'])} 输入 token → {len(result['output_ids'])} 输出 token · "
                  f"{len(rows)} 个真实事件 · {result['elapsed_ms'] / 1000:.2f} 秒（含跟踪开销）\n\n"
                  "下面是本次运行的回放；前两次 forward 深入源码，其余步骤保留模块与 token 记录。")
        # 各返回项顺序必须与 build_app 中的 outputs 列表完全一致。
        yield (status, result, result["answer"], rows, input_rows, generated_rows,
               gr.update(value=0, maximum=max(1, len(rows) - 1)), file, meta, "全部", "")
    # 把后端错误转换成页面提示，同时保留原始异常链便于排查。
    except Exception as exc:
        raise gr.Error(f"运行失败：{type(exc).__name__}: {exc}。详细排查见 README。") from exc


# 声明组件与事件绑定；构建页面本身不会加载模型权重。
def build_app():
    with gr.Blocks(title="Qwen3 推理观察室", theme=gr.themes.Soft(primary_hue="teal"), css=CSS) as demo:
        # 每个浏览器会话保存自己的运行记录；共享的只有 ENGINE 中的模型。
        state = gr.State(None)
        gr.HTML('<div class="hero"><div class="eyebrow">QWEN3 / EXECUTION LAB</div><h1>让一次推理，变得看得见。</h1><p>运行官方 Qwen3-0.6B，沿着真实调用走进 modeling_qwen3.py。</p><div class="flow"><span>文本 → Tokenizer</span><span>generate()</span><span>Embedding</span><span>28 × Decoder</span><span>logits → token ↺</span></div></div>')
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                prompt = gr.Textbox(label="输入文本", value="用一句话解释什么是大模型。", lines=3)
                tokens = gr.Slider(1, 32, value=8, step=1, label="最多生成多少个 token")
                layer = gr.Slider(0, 27, value=0, step=1, label="深入观察第几层 Decoder（从 0 开始）")
                cache = gr.Checkbox(value=True, label="启用 KV Cache")
                button = gr.Button("运行官方模型并记录", variant="primary")
                gr.Markdown("建议先生成 **4～8 个 token**。关闭 KV Cache 后再运行，比较每次 forward 的序列长度。使用 eager attention 和贪心生成，方便观察与复现。")
            with gr.Column(scale=2):
                status = gr.Markdown("等待运行。模型将在第一次点击时加载。")
                answer = gr.Textbox(label="模型实际输出", lines=5, interactive=False)
                download = gr.File(label="下载完整运行记录 JSON（包含对应源码）")
        # 三个视角：源码事件、token 数据、配置与学习说明。
        with gr.Tabs():
            with gr.Tab("01 · 调用与源码回放"):
                with gr.Row():
                    phase = gr.Dropdown(["全部", "准备", "Prefill", "Decode"], value="全部", label="阶段")
                    keyword = gr.Textbox(label="过滤函数 / 模块", placeholder="例如 q_proj、eager_attention、mlp、update")
                table = gr.Dataframe(headers=["事件 ID", "forward 次数", "阶段", "事件", "模块 / 函数", "源码行"],
                                     datatype=["number", "number", "str", "str", "str", "str"],
                                     interactive=False, label="真实调用顺序 · 点击任意行查看详情", max_height=300)
                with gr.Row():
                    previous = gr.Button("← 上一步", scale=0)
                    index = gr.Slider(0, 1, value=0, step=1, label="事件回放位置（全局 ID）")
                    following = gr.Button("下一步 →", scale=0)
                heading = gr.Markdown("运行后选择一个事件。")
                with gr.Row():
                    source = gr.HTML()
                    values = gr.JSON(label="真实局部变量 / 张量形状")
            with gr.Tab("02 · 输入与下一个 token"):
                gr.Markdown("输入包含聊天模板的特殊 token。单个 token 解码可能显示半个字符，raw 列保留 tokenizer 原始表示。候选概率来自实际生成分数的 Softmax；本项目选择概率最高的 token。")
                input_table = gr.Dataframe(headers=["位置", "ID", "解码文本", "raw token"], interactive=False, label="输入分词")
                top_table = gr.Dataframe(headers=["生成步骤", "候选 ID", "候选文本", "概率"], interactive=False, label="每步 Top 5 候选")
            with gr.Tab("03 · 模型与阅读指南"):
                gr.Markdown("""### 推荐观察顺序
1. 搜索 `Qwen3ForCausalLM`：看 generate 如何进入模型，以及最后输出 logits。
2. 搜索 `embed_tokens`：整数 token ID 如何变成 `[batch, sequence, hidden]`。
3. 搜索 `model.layers.0`：看 RMSNorm → Attention → 残差 → MLP → 残差。
4. 搜索 `q_proj` / `k_proj` / `v_proj`：比较 GQA 的投影维度。
5. 搜索 `apply_rotary_pos_emb`：查看旋转前后的 Q/K 形状。
6. 搜索 `eager_attention_forward`：逐行看 QKᵀ、mask、Softmax、乘 V。
7. 搜索 `update`：查看 KV Cache 的更新；切换 Decode 对比序列长度。

**观察范围：** 官方 Python 模型、生成循环的关键调用，以及选中层的 Python 源码行；不追踪 tokenizer 的 Rust 内部、PyTorch C++ 或 GPU 内核。张量记录 shape/dtype/device，不复制完整激活值。事件时间含跟踪开销，不是性能基准。

**回放不是暂停模型：** 模型先真实运行，然后逐事件阅读记录。想用 IDE 断点实时暂停，见 README。
""")
                metadata = gr.JSON(label="实际模型配置与运行环境")
        # 输出组件顺序对应 run 每次 yield 返回的元组。
        outputs = [status, state, answer, table, input_table, top_table, index, download, metadata, phase, keyword]
        # 按钮触发真实推理；过滤、选行、翻页只读取已经保存的 state。
        button.click(run, [prompt, tokens, layer, cache], outputs, concurrency_limit=1)
        phase.change(event_rows, [state, phase, keyword], table)
        keyword.change(event_rows, [state, phase, keyword], table)
        index.change(detail, [state, index], [heading, source, values])
        state.change(detail, [state, index], [heading, source, values])

        # 表格行号会随过滤改变；从首列取全局事件 ID，而非直接使用行号。
        def selected(rows, evt: gr.SelectData):
            return int(rows.iloc[evt.index[0], 0])

        table.select(selected, [table], [index])
        # 翻页索引限制在合法区间；尚无结果时维持位置 0。
        previous.click(lambda i: max(0, int(i) - 1), index, index)
        following.click(lambda r, i: min(len(r["events"]) - 1, int(i) + 1) if r else 0, [state, index], index)
    return demo


# 可选 Web 启动入口；纯 Python 调试使用 debug_official.py 即可。
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    ENGINE.device = args.device
    # 只绑定本机回环地址，不创建公开分享链接；允许下载 outputs 中的记录。
    build_app().queue().launch(server_name="127.0.0.1", server_port=args.port, share=False,
                              allowed_paths=[str(ROOT / "outputs")])
