llm.generate 函数在 nanovllm/engine/llm_engine.py 中。
### generate

依旧先看个大概逻辑：
#### step 1 将 prompt, sampling_params 成对加入请求

```python
        # 将 {prompt, sampling_params} 成对加入请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
```

这里的参数是成对的，说明其实不同 prompt 也可以使用不同的 sampling_params。
#### step 2 循环处理

```python
        # 请求未结束时，继续循环
        while not self.is_finished():
            t = perf_counter()
            # 进行一步 step
            output, num_tokens = self.step()
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
```
#### step 3 返回输出

```python
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 生成的 token id 解码为文本字符串
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
```


### add_request

```python
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            # 如果传入的 prompt 是字符串 str 类型，则编码为 token id
            prompt = self.tokenizer.encode(prompt)
        # 创建 Sequence
        seq = Sequence(prompt, sampling_params)
        # 将 Sequence 加入调度器中
        self.scheduler.add(seq)
```

#### Sequence

这个类包含了一个请求的完整信息（毕竟  self.scheduler.add(seq) 添加的就是一个 Sequence）。

```python
class Sequence:
    block_size = 256
    counter = count() # python 标准库提供的整数计数器

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter) # 递增 id
        self.status = SequenceStatus.WAITING  # 当前状态
        self.token_ids = copy(token_ids) # 完整的 token id 序列，后续会追加写入
        self.last_token = token_ids[-1] # 当前 token id 序列的最后一个 token id，后续随追加写入更新
        self.num_tokens = len(self.token_ids) # 当前 token id 序列长度，后续随追加写入更新
        self.num_prompt_tokens = len(token_ids) # 创建时传入的 prompt 的 token id 数，之后不变，为常量
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        # 保存采样参数
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        
    # 将 token id 追加到 Sequence 保存的序列
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
```

### is_finished

简短一行，其实是来自调度器 scheduler 是否结束：

```python
    def is_finished(self):
        return self.scheduler.is_finished()
```

### step

```python
    def step(self):
        # 本轮要处理的请求列表 seqs， 为 prefill 预填充阶段还是 decode 解码阶段
        seqs, is_prefill = self.scheduler.schedule()
        # 只是用来更新 pbar 进度条的，不重要，可忽视不看
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        # 执行本轮模型计算并采用，得到各 seq 新的 token id
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 后处理，主要是更新 seq 的状态，因为上一步生成了新 token 嘛，所以得更新一下，比如追加写入啥的
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # 只返回已完成的请求 seq 及其累计生成的 token id
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens
```

一步一步来看：

```python
	seqs, is_prefill = self.scheduler.schedule()
```

返回了 scheduler 决定要处理的 seqs 列表，和是否为 prefill 阶段还是 decode 阶段。

关于 prefill 和 decode 阶段，后文再细说。
现在可以粗糙的理解为，对 seq 的处理是先进行 prefill 阶段，完成后，后续的处理是 decode 阶段。

```python
	# 只是用来更新 pbar 进度条的，不重要，可忽视不看
	num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
```

这个单纯更新进度条的，可以忽略。
这里代码约定的是，返回非负数表示 prefill 阶段预计要处理的 token 数量，返回小于等于 0 则表示 decode 阶段要处理的 token 数。
因为 decode 阶段每个请求 seq 只处理一个 token，所以这里返回 len(seq) 简化代码了。

```python
        # 执行本轮模型计算并采用，得到各 seq 新的 token id
        token_ids = self.model_runner.call("run", seqs, is_prefill)
```
进行模型计算。

```python
        # 后处理，主要是更新 seq 的状态，因为上一步生成了新 token 嘛，所以得更新一下，比如追加写入啥的
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

后处理。
主要是更新 seq 的状态，生成了新的 token 追加写入到 seq 中之类的。或者 seq 已完成，标记为 is_finished

```python
        # 只返回已完成的请求 seq 及其累计生成的 token id
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens
```

这个就是返回已完成的 seq 和生成的 token。

ok，接下来就是细化 step，往 scheduler 里看了。

2026.10.29