from dataclasses import dataclass, field


@dataclass
class MusicRVQEncoderConfig:
    dropout: int = 0.1
    hidden_size: int = 504
    num_heads: int = 8
    num_layers: int = 4
    intermediate_size: int = 2048
    is_gelu_approx: bool = False
    layer_norm_eps: float = 1e-6

    # feature extractor
    filter_sizes: list = field(
        default_factory=lambda: [504, 504]
    )
    kernel_sizes: list = field(default_factory=lambda: [3, 3])
    strides: list = field(default_factory=lambda: [2, 2])
    codebook_sizes: list = field(
        default_factory=lambda: [512, 512]
    )
    commitment_costs: list = field(
        default_factory=lambda: [0.25, 0.25]
    )
    num_quantizers_list: list = field(
        default_factory=lambda: [8, 8]
    )

    attention_norm_type: str = "postnorm"
