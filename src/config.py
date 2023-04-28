from dataclasses import dataclass, field


@dataclass
class MusicRVQAEConfig:
    # transformer
    dropout: int = 0.1
    num_heads: int = 8
    intermediate_size: int = 2048
    is_gelu_approx: bool = False
    layer_norm_eps: float = 1e-6
    attention_norm_type: str = "postnorm"

    # feature extractor
    filter_sizes: list = field(
        default_factory=lambda: [504, 504, 504, 504]
    )
    kernel_sizes: list = field(default_factory=lambda: [3, 3, 3, 3])
    strides: list = field(default_factory=lambda: [2, 2, 2, 2])


@dataclass
class MusicRVQLMConfig:
    # rvq
    hidden_size: int = 504
    codebook_size = 504 * 2
    commitment_cost = 0.25
    num_quantizers = 16
