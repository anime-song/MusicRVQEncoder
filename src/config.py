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
    # encoder
    dropout: int = 0.1
    hidden_size: int = 504
    num_heads: int = 8
    num_layers: int = 6
    intermediate_size: int = 2048
    is_gelu_approx: bool = False
    layer_norm_eps: float = 1e-6
    attention_norm_type: str = "postnorm"

    # feature extractor
    embedding_dim: int = 504
    codebook_size: int = 512
    commitment_cost: float = 0.
    num_quantizers: int = 8
    ema_decay: float = 0.99
    threshold_ema_dead_code: int = 2
