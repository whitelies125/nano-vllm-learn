非常简洁的一个文件。
#### step 1 指定文件所在目录

```python
    # 指定模型文件所在目录
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
```

#### step 2 从指定目录加载 tokenizer 和模型

```python
    # 从指定 path 加载 tokenizer 和 LLM 模型
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
```

python 的 transformers 库主要由 Hugging Face 开发和维护，它提供了一些方便的函数。
`AutoTokenizer.from_pretrained(path)` 是从路径 path 加载分词器。

Qwen3-0.6B 的仓库在：[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/main)

可以看到其中目录结构为：

```shell
huggingface/
└── Qwen3-0.6B/
    ├── config.json
    ├── model.safetensors
    ├── generation_config.json
    │
    ├── tokenizer.json           # 分词器定义，包含词表和分词流程
    ├── tokenizer_config.json    # 分词器配置、特殊 token、聊天模板等
    ├── vocab.json               # token 与 ID 的对应词表
    ├── merges.txt               # BPE 分词的合并规则
    │
    ├── README.md
    ├── LICENSE
    └── .gitattributes
```

其中涉及分词器的文件有 tokennizer.json，tokenizer_config.json，vocab.json，merges.txt。

因此，通过 transformers 库提供的 from_pretrained 函数，提供模型 path 后，函数就会读取相关的文件，创建一个 tokenizer 实例。

`llm = LLM(..)` 涉及的内容比较多，后面再详细讲解，不过也是传入了 path，依旧会去模型目录读取文件。

#### step 3 设置采样参数

```python
    # 设置采样参数
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
```

采用参数翻到 sampling_params.py 文件。
共有 3 个参数，含义如下：

```python
class SamplingParams:
    temperature: float = 1.0 # SoftMax 公式中的参数 T
    # 值越低（越接近0），更偏向于选择高分 token，结果越确定；
    # 值越高（1或更高），低分 token 被选择地概率越高，则回答更多样。
    max_tokens: int = 64 # 每条请求最多生成 max_tokens 个新 token, 到达上限时停止生成 token
    ignore_eos: bool = False # 是否忽略 EOS(End Of Sequeue).
    # EOS 是模型用来表示生成结束的特殊 token.
    # 如果不忽略，那么遇到 EOS token 则会停止当前请求的生成
    # 如果忽略，则不会停止，而是直到 max_tokens 再停止

```

#### step 4 准备 prompts

```python
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
```

这里使用 transformers 库的 apply_chat_template 函数，生成聊天格式的 prompts

这一部分的说明可以参考：[nano-vllm 源码阅读](https://kinnari-notes.vercel.app/%E6%9D%82/nano-vllm#prompts)
讲解得很详细。
#### step 5 推理，生成输出

```python
    # 推理，生成输出
    outputs = llm.generate(prompts, sampling_params)
```

OK，到这里，example.py 完成。
下一步，深入 llm.generate 看看。

2026.10.29