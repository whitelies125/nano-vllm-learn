import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            # 如果传入的 prompt 是字符串 str 类型，则编码为 token id
            prompt = self.tokenizer.encode(prompt)
        # 创建 Sequence
        seq = Sequence(prompt, sampling_params)
        # 把 Sequence 加入调度器中
        self.scheduler.add(seq)

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

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True, # 控制是否显示进度条
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm) # 进度条总任务数= total=len(prompts)
        if not isinstance(sampling_params, list):
            # 这里只是把单个 sampling_params 处理成列表 list
            sampling_params = [sampling_params] * len(prompts)
        # 将 {prompt, sampling_params} 成对加入请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        # 请求未结束时，继续循环
        while not self.is_finished():
            t = perf_counter()
            # 进行一步 step
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1) # 进度条已完成任务数+1
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 生成的 token id 解码为文本字符串
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
