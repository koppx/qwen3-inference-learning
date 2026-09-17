"""直接调试官方 Qwen3；无 Web、无 hooks、无 sys.settrace。"""
import os
from pathlib import Path



ROOT = Path(__file__).resolve().parent  # 从脚本所在目录定位缓存，避免 IDE 工作目录不同导致找不到权重。
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))  # 必须在导入 transformers / huggingface_hub 之前设置缓存路径。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


import torch  # PyTorch 执行张量计算。
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase  # Transformers 提供分词器、模型结构和生成循环；基类用于类型注解。


MODEL_ID = "Qwen/Qwen3-0.6B"  # 选择 0.6B 稠密模型；更换模型时需重新核对配置和张量维度。
PROMPT = "1加1等于几？"
MAX_NEW_TOKENS = 16
USE_CACHE = True       # 改成 False，观察每次 forward 的输入长度。
DEVICE = "auto"        # 可改成 cpu / mps / cuda。
LOCAL_FILES_ONLY = True  # 本机权重已下载；新机器首次运行改成 False。



def main():  # 调试主入口：初始化一次模型，然后完整处理一个请求。

    device = DEVICE
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"  # 按 CUDA → Apple MPS → CPU 的顺序选择设备。

    dtype = torch.float32 if device == "cpu" else torch.float16  # CPU 使用 FP32；GPU 使用 FP16 减少参数与激活内存。



    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=LOCAL_FILES_ONLY)  # 加载词表和规则；类型注解帮助编辑器跳转到方法定义。

    model = AutoModelForCausalLM.from_pretrained(  # ① 根据 config 创建 Qwen3ForCausalLM，并将训练好的权重填入各层。
        MODEL_ID, local_files_only=LOCAL_FILES_ONLY,

        dtype=dtype, attn_implementation="eager",  # 显式执行 QKᵀ、mask、Softmax、乘 V，便于逐行断点观察。

    ).to(device).eval()  # to 移动模型参数；eval 切换推理行为，不等同于关闭梯度记录。

    model.generation_config.temperature = 1.0  # 清理检查点中的采样设置；下面实际使用 do_sample=False 的贪心分支。
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    print(f"模型: {MODEL_ID}, 设备: {device}, dtype: {dtype}")

    print(model)  # 打印模块树：Embedding、28 个 Decoder、最终 RMSNorm 和 lm_head。



    messages = [{"role": "user", "content": PROMPT}]  # role 标明说话者，content 是请求内容；此时仍是普通 Python 数据。
    text = tokenizer.apply_chat_template(  # ② 用户文本包装成聊天模板；这里还只是字符串。

        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,  # tokenize=False 返回字符串；generation prompt 添加 assistant 开头。
    )
    print("聊天模板:", repr(text))



    inputs = tokenizer(text, return_tensors="pt").to(device)  # ③ 字符串转为 BatchEncoding；整数 input_ids 和 attention_mask 均为 [1,18]。

    prompt_length = inputs.input_ids.shape[1]  # 记住 prompt 长度；generate 返回的序列还包含原始输入。
    print("input_ids:", inputs.input_ids.tolist())
    print("输入形状:", tuple(inputs.input_ids.shape))


    with torch.inference_mode():  # 关闭自动求导记录；本次只推理，不计算梯度或更新参数。


        output_ids = model.generate(  # ④ 在此打断点；generate 管理循环，forward 计算 logits，argmax 选新 token；内部断点见 learning.md。

            **inputs, max_new_tokens=MAX_NEW_TOKENS,  # ** 展开 input_ids、attention_mask；max_new_tokens 只计新增 token。

            do_sample=False, use_cache=USE_CACHE,  # 取最高分 token；缓存开启后，后续 forward 通常只接收 1 个新 token。
        )



    new_ids = output_ids[0, prompt_length:]  # ⑤ 第 0 个 batch 去掉原始输入，只保留新增 token；结果是一维整数张量。

    answer = tokenizer.decode(new_ids, skip_special_tokens=True)  # 把新增 ID 转回文字，并过滤 EOS 等特殊 token。
    print("新生成的 token ID:", new_ids.tolist())
    print("回答:", answer)



if __name__ == "__main__":  # 直接执行时运行 main；被其他文件 import 时不自动加载大模型。
    main()
