from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0 # SoftMax 公式中的参数 T
    # 值越低（越接近0），更偏向于选择高分 token，结果越确定；
    # 值越高（1或更高），低分 token 被选择地概率越高，则回答更多样。
    max_tokens: int = 64 # 每条请求最多生成 max_tokens 个新 token, 到达上限时停止生成 token
    ignore_eos: bool = False # 是否忽略 EOS(End Of Sequeue).
    # EOS 是模型用来表示生成结束的特殊 token.
    # 如果不忽略，那么遇到 EOS token 则会停止当前请求的生成
    # 如果忽略，则不会停止，而是直到 max_tokens 再停止

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
