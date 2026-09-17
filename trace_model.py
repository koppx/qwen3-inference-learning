"""Observe the official model; never reimplement or replace its forward math."""
import inspect
import json
import os
from pathlib import Path
import sys
import time

# 缓存环境变量必须在导入 Hugging Face 相关库前设置，确保路径生效。
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache

MODEL_ID = "Qwen/Qwen3-0.6B"


# 把运行对象转成可序列化摘要，用于展示；不保留张量引用或完整激活。
def describe(value, depth=0):
    """JSON metadata only: never retain tensors or copy full activations to CPU."""
    # 只读取 shape/dtype/device，不触发整个张量从 GPU 复制到 CPU。
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    # 这里展示第 0 层长度；各层在 forward 中先后更新，长度暂时可能不同。
    if isinstance(value, Cache):
        return {"type": type(value).__name__, "cached_tokens_layer_0": int(value.get_seq_length())}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value if not isinstance(value, str) else value[:300]
    # 限制递归深度与容器长度，避免记录整个模型对象或过大的嵌套结构。
    if depth < 2 and isinstance(value, dict):
        return {str(k): describe(v, depth + 1) for k, v in list(value.items())[:25]}
    # 限制递归深度与容器长度，避免记录整个模型对象或过大的嵌套结构。
    if depth < 2 and isinstance(value, (list, tuple)):
        return [describe(v, depth + 1) for v in value[:8]]
    return type(value).__name__


# 上下文管理器：进入时安装观察函数，退出时恢复现场，不改写模型计算。
class Recorder:
    # 保存实例状态；具体字段用于下方的方法，不在构造时开始推理。
    def __init__(self, model, layer=0, detailed_steps=2):
        self.model = model
        self.layer = layer
        self.detailed_steps = detailed_steps
        self.events = []
        self.sources = {}
        self.handles = []
        # -1 代表输入准备；外层模型每进入一次 forward 就加 1。
        self.step = -1
        self.inside_selected = False
        self.started = time.perf_counter()
        self.old_trace = None
        # 用对象身份映射到模块路径，例如 model.layers.0.self_attn.q_proj。
        self.names = {id(m): n or "Qwen3ForCausalLM" for n, m in model.named_modules()}

    # 统一记录事件与相对时间；源文件只保存一次，供离线回放定位。
    def add(self, kind, name, values=None, filename=None, line=None):
        if filename and filename not in self.sources:
            self.sources[filename] = Path(filename).read_text()
        self.events.append({"id": len(self.events), "step": self.step,
                            "phase": "准备" if self.step < 0 else ("Prefill" if self.step == 0 else "Decode"),
                            "kind": kind, "name": name, "values": values or {},
                            "file": filename, "line": line,
                            "elapsed_ms": round((time.perf_counter() - self.started) * 1000, 2)})

    # forward 前的 hook：记录输入；返回 None，因此不会替换输入参数。
    def pre(self, name, module, args, kwargs):
        if module is self.model:
            self.step += 1
        if name == f"model.layers.{self.layer}":
            self.inside_selected = True
        # 去掉装饰器包装以定位原函数源码，而非显示包装函数的位置。
        fn = inspect.unwrap(module.forward)
        filename = inspect.getsourcefile(fn)
        line = inspect.getsourcelines(fn)[1] if filename else None
        self.add("进入模块", name, {"args": describe(args), "kwargs": describe(kwargs)}, filename, line)

    # forward 后记录返回值摘要；返回 None，因此不会替换模型输出。
    def post(self, name, module, args, kwargs, output):
        self.add("离开模块", name, {"output": describe(output)})
        if name == f"model.layers.{self.layer}":
            self.inside_selected = False

    # Python 跟踪回调：frame 提供当前函数、源码行和局部变量。
    def trace(self, frame, event, arg):
        filename = frame.f_code.co_filename
        function = frame.f_code.co_name
        # 只关注模型、关键生成函数和选中层的 Cache update，跳过无关库。
        is_qwen = filename.endswith("/models/qwen3/modeling_qwen3.py")
        is_generation = filename.endswith("/generation/utils.py") and function in {
            "generate", "_sample", "prepare_inputs_for_generation", "_update_model_kwargs_for_generation"}
        is_cache = filename.endswith("/cache_utils.py") and function == "update" and self.inside_selected
        if not (is_qwen or is_generation or is_cache):
            return None
        # 生成循环只记录调用/返回；模型内部才深入到行级别。
        if event in ("call", "return") and is_generation:
            self.add("函数调用" if event == "call" else "函数返回", function,
                     {k: describe(v) for k, v in frame.f_locals.items() if k != "self" and not k.startswith("_")},
                     filename, frame.f_lineno)
        # 只在前 detailed_steps 次 forward 深入选中层，控制记录量。
        if self.step < self.detailed_steps and self.inside_selected and (is_qwen or is_cache):
            if event in ("line", "return"):
                owner = frame.f_locals.get("self")
                name = self.names.get(id(owner), function)
                # line 事件发生在该行执行前，变量显示的是上一条已完成语句的结果。
                self.add("即将执行" if event == "line" else "函数返回", f"{name} · {function}",
                         {k: describe(v) for k, v in frame.f_locals.items() if k != "self" and not k.startswith("_")},
                         filename, frame.f_lineno)
        # 返回自身，继续接收当前函数后续行事件。
        return self.trace

    # with recorder 进入时启用记录；应避免与 IDE 的 sys.settrace 调试器并用。
    def __enter__(self):
        # 保存当前线程原有的跟踪器，结束后恢复它。
        self.old_trace = sys.gettrace()
        # Whole-model boundaries, every Decoder, and every child of the selected Decoder.
        prefix = f"model.layers.{self.layer}"
        for name, module in self.model.named_modules():
            if (name in ("", "model", "model.embed_tokens", "model.rotary_emb", "model.norm", "lm_head")
                    or (name.startswith("model.layers.") and name.count(".") == 2)
                    or name.startswith(prefix + ".")):
                # 保存 hook 句柄供清理；lambda 默认参数 n=name 固定当前循环的模块名。
                self.handles.append(module.register_forward_pre_hook(
                    lambda m, a, kw, n=name: self.pre(n or "Qwen3ForCausalLM", m, a, kw), with_kwargs=True))
                self.handles.append(module.register_forward_hook(
                    lambda m, a, kw, out, n=name: self.post(n or "Qwen3ForCausalLM", m, a, kw, out),
                    # 同时捕获具名参数；异常时也尽量执行退出 hook。
                    with_kwargs=True, always_call=True))
        sys.settrace(self.trace)
        return self

    # 即使推理抛异常也会执行清理，防止下次请求重复注册记录器。
    def __exit__(self, *exc):
        sys.settrace(self.old_trace)
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.inside_selected = False


# 实现 generate 的 streamer 接口，捕获每一步真正选出的 token ID。
class TokenRecorder:
    """The official generate() streamer receives the actual selected token IDs."""
    # 保存实例状态；具体字段用于下方的方法，不在构造时开始推理。
    def __init__(self, recorder, tokenizer):
        self.recorder, self.tokenizer = recorder, tokenizer
        self.first = True
        self.ids = []

    # generate 先推送整段 prompt，随后每轮推送新增 token。
    def put(self, value):
        if self.first:
            self.first = False  # generate() first streams the entire prompt.
            return
        # 本工具仅单请求；展平成 ID 列表后累积，便于解码当前完整回答。
        ids = value.reshape(-1).tolist()
        self.ids.extend(ids)
        self.recorder.add("生成 token", "generate → streamer.put", {
            "token_ids": ids, "token_text": self.tokenizer.decode(ids),
            "answer_so_far": self.tokenizer.decode(self.ids, skip_special_tokens=True)})

    # streamer 协议的结束通知；当前实现不需要额外刷新缓冲区。
    def end(self):
        pass


# 组织模型加载、单请求推理与结果导出；Web/CLI 复用同一个入口。
class Inspector:
    # 保存实例状态；具体字段用于下方的方法，不在构造时开始推理。
    def __init__(self, device="auto"):
        self.device = device
        self.model = None
        self.tokenizer = None

    # 延迟加载模型，并在后续请求中复用权重，避免重复分配内存。
    def load(self):
        if self.model is not None:
            return
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float32 if device == "cpu" else torch.float16
        # Reuse a complete download immediately, without waiting for Hub HEAD requests.
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, local_files_only=True, dtype=dtype, attn_implementation="eager")
        # 本地缓存缺失时转为允许联网加载；完整缓存可直接离线运行。
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=dtype, attn_implementation="eager")
        model = model.to(device).eval()
        # Neutralize sampling defaults stored with the checkpoint for this greedy tutorial.
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
        model.generation_config.top_k = 50
        self.tokenizer, self.model = tokenizer, model

    # 处理一个请求：校验 → 模板 → 分词 → 官方 generate → 汇总记录。
    def run(self, prompt, max_new_tokens=8, layer=0, use_cache=True):
        if not prompt.strip():
            raise ValueError("请先输入一个问题。")
        # 限制教学输出长度，避免产生过多事件或长时间占用设备。
        if not 1 <= int(max_new_tokens) <= 32:
            raise ValueError("生成长度必须在 1～32 之间。")
        self.load()
        if not 0 <= int(layer) < self.model.config.num_hidden_layers:
            raise ValueError("Decoder 层号超出范围。")
        # 每次请求使用独立的事件列表；模型权重仍由 Inspector 复用。
        recorder = Recorder(self.model, int(layer))
        recorder.add("输入", "用户文本", {"text": prompt})
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        recorder.add("模板", "tokenizer.apply_chat_template", {"text": text, "enable_thinking": False})
        # token IDs 是 int64；to(model.device) 让输入与权重位于同一设备。
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        ids = inputs.input_ids[0].tolist()
        # 注意力矩阵随长度增长；这里对教学请求设置输入上限。
        if len(ids) > 256:
            raise ValueError(f"当前输入含 {len(ids)} 个 token；教学模式最多 256 个，请缩短输入。")
        recorder.add("分词", "tokenizer → input_ids / attention_mask", {
            "input_ids": ids, "tensors": describe(inputs.data)})
        streamer = TokenRecorder(recorder, self.tokenizer)
        # 同时关闭求导并启用观察；离开 with 后自动撤掉所有 hooks。
        with torch.inference_mode(), recorder:
            output = self.model.generate(
                **inputs, max_new_tokens=int(max_new_tokens), do_sample=False,
                use_cache=bool(use_cache), streamer=streamer,
                # 请求结构化输出和每步分数，用于事后展示候选 token。
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id)
        # 生成结果含 prompt；从输入长度后切片才是新生成的回答。
        generated = output.sequences[0, len(ids):].tolist()
        top_tokens = []
        # 每个 score 对应一步 [batch,vocab] 的处理后分数，不是注意力矩阵。
        for step, score in enumerate(output.scores):
            # 展示用途：转成概率后取 Top 5；这不会重新选择或改变生成结果。
            probabilities, choices = score[0].float().softmax(-1).topk(5)
            top_tokens.append({"step": step, "candidates": [
                {"id": idx, "text": self.tokenizer.decode([idx]), "probability": round(p, 6)}
                for idx, p in zip(choices.tolist(), probabilities.tolist())]})
        # 汇总 JSON 友好的配置、文本、token 和事件；不保存模型权重或激活。
        result = {"model": MODEL_ID, "transformers": transformers.__version__,
                  "torch": torch.__version__, "device": str(self.model.device),
                  "dtype": str(self.model.dtype), "parameters": sum(p.numel() for p in self.model.parameters()),
                  "config": json.loads(self.model.config.to_json_string(use_diff=False)),
                  "selected_layer": int(layer), "use_cache": bool(use_cache),
                  "detailed_steps": recorder.detailed_steps, "prompt": prompt, "chat_template": text,
                  "input_tokens": [{"id": i, "text": self.tokenizer.decode([i]),
                                    "raw": self.tokenizer.convert_ids_to_tokens(i)} for i in ids],
                  "output_ids": generated, "answer": self.tokenizer.decode(generated, skip_special_tokens=True),
                  "top_tokens": top_tokens, "events": recorder.events, "sources": recorder.sources,
                  "elapsed_ms": round((time.perf_counter() - recorder.started) * 1000, 2)}
        return result


# 将一次完整记录写入 outputs，随机文件名避免覆盖先前请求。
def save_result(result):
    import uuid
    directory = ROOT / "outputs"
    directory.mkdir(exist_ok=True)
    target = directory / f"trace-{uuid.uuid4().hex[:12]}.json"
    # ensure_ascii=False 保留中文；缩进便于直接阅读 JSON。
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return str(target)


# CLI 入口；没有 Web 也可以生成 trace JSON。
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the official Qwen3 and record its execution")
    parser.add_argument("--prompt", default="用一句话解释什么是大模型。")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    result = Inspector(args.device).run(args.prompt, args.tokens, args.layer, not args.no_cache)
    print(result["answer"])
    print(f"{len(result['events'])} events → {save_result(result)}")
