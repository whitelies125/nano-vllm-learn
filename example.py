import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # 指定模型文件所在目录
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # 从 path 加载 tokenizer 和 LLM 模型
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # 设置采样参数
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    # 准备 prompts
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template( # 将 prompts 转换为模型的对话格式
            [{"role": "user", "content": prompt}],
            tokenize=False, # 为 false 输出字符串; 为 true 输出 token id
            add_generation_prompt=True, # 为 true 则输入最后添加上提示 assistant 开始的 token; 为 false 则不添加
        )
        for prompt in prompts
    ]
    # 推理，生成输出
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
