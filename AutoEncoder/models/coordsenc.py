from typing import Callable, Tuple

import torch
from torch import Tensor


class CoordsEncoder:
    def __init__(
        self,
        input_dims: int = 3,
        include_input: bool = True,
        max_freq_log2: int = 9,
        num_freqs: int = 10,
        log_sampling: bool = True,
        periodic_fns: Tuple[Callable, Callable] = (torch.sin, torch.cos),
    ) -> None:
        self.input_dims = input_dims
        self.include_input = include_input
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        self.create_encoding_fn()

    def create_encoding_fn(self) -> None:
        encoding_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            encoding_fns.append(lambda x: x)
            out_dim += d

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(
                0.0, self.max_freq_log2, steps=self.num_freqs
            )
        else:
            freq_bands = torch.linspace(
                2.0**0.0, 2.0**self.max_freq_log2, steps=self.num_freqs
            )

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                encoding_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.encoding_fns = encoding_fns
        self.out_dim = out_dim

    def encode(self, inputs: Tensor) -> Tensor:
        return torch.cat([fn(inputs) for fn in self.encoding_fns], -1)
