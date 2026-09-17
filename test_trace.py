"""Small random official Qwen3 tests; real downloaded-model acceptance is separate."""
import sys
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from trace_model import Recorder


# 构造微型随机权重的官方 Qwen3，只测试记录机制，不用于评估模型回答能力。
def model():
    # 固定初始化随机数，便于复现跟踪前后的输出对比。
    torch.manual_seed(5)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8,
                        eos_token_id=None, pad_token_id=0)
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


# 确认观察器不影响生成 token，并能捕获 Attention/RoPE 和输入形状。
def test_observation_preserves_output_and_captures_internals():
    m = model()
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        expected = m.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False)
        with Recorder(m) as recorder:
            actual = m.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False)
    # 最关键断言：同权重同输入，启用记录后输出必须保持一致。
    assert torch.equal(expected, actual)
    assert recorder.step == 2
    names = [e["name"] for e in recorder.events]
    assert any("eager_attention_forward" in n for n in names)
    assert any("apply_rotary_pos_emb" in n for n in names)
    assert any(e["values"].get("attn_weights", {}).get("shape") == [1, 4, 3, 3]
               for e in recorder.events if isinstance(e["values"].get("attn_weights"), dict))
    # 只取外层模型进入事件，避免将 28 层内部调用误认为多次生成。
    calls = [e for e in recorder.events if e["kind"] == "进入模块" and e["name"] == "Qwen3ForCausalLM"]
    assert [e["values"]["kwargs"]["input_ids"]["shape"][1] for e in calls] == [3, 1, 1]
    assert not recorder.handles
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in m.modules())


# 关闭缓存后应重算完整前缀，因此输入序列长度逐轮递增。
def test_no_cache_recomputes_sequence():
    m = model()
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode(), Recorder(m, layer=1) as recorder:
        m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=3, do_sample=False, use_cache=False)
    # 只取外层模型进入事件，避免将 28 层内部调用误认为多次生成。
    calls = [e for e in recorder.events if e["kind"] == "进入模块" and e["name"] == "Qwen3ForCausalLM"]
    assert [e["values"]["kwargs"]["input_ids"]["shape"][1] for e in calls] == [3, 4, 5]
    assert any("model.layers.1.self_attn.q_proj" == e["name"] for e in recorder.events)


# 模拟异常中断，确认 hooks 和 Python 跟踪状态都会被还原。
def test_cleanup_on_exception():
    m = model()
    old = sys.gettrace()
    try:
        with Recorder(m):
            raise RuntimeError("test")
    except RuntimeError:
        pass
    assert sys.gettrace() is old
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in m.modules())


# 使用实际模型导出的记录，验证 JSON 展示和源码事件定位。
def test_ui_serializes_real_trace():
    import json
    from pathlib import Path
    import gradio as gr
    from app import detail, event_rows
    records = list((Path(__file__).parent / "outputs").glob("trace-*.json"))
    # 没有真实权重运行记录时明确跳过，不把随机模型记录冒充真实结果。
    if not records:
        import pytest
        pytest.skip("Run real-model CLI acceptance first")
    result = json.loads(records[-1].read_text())
    gr.JSON().postprocess(result["config"])
    rows = event_rows(result, "Prefill", "eager_attention_forward")
    assert rows
    heading, source, values = detail(result, rows[0][0])
    assert "source" in source and "modeling_qwen3.py" in heading
    gr.JSON().postprocess(values)
