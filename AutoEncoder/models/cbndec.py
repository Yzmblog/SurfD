from einops import repeat
from torch import Tensor, nn

class DecoderConditionalBatchNorm(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dim_condition_embedding: int,
        dim_hidden_layers: int,
        num_hidden_layers: int,
        dim_out: int,
        refine = False
    ):
        super().__init__()

        self.fc_p = nn.Conv1d(input_dim, dim_hidden_layers, 1)

        self.num_blocks = num_hidden_layers
        self.blocks = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.blocks.append(
                ConditionalResnetBlock1d(
                    dim_condition_embedding,
                    dim_hidden_layers,
                )
            )

        self.bn = ConditionalBatchNorm1d(dim_condition_embedding, dim_hidden_layers)

        self.fc_out = nn.Conv1d(dim_hidden_layers, dim_out, 1)
        if refine:
            nn.init.zeros_(self.fc_out.weight)
        self.actvn = nn.ReLU()

    def forward(self, points: Tensor, conditions: Tensor) -> Tensor:
        p = points.transpose(1, 2)
        c = conditions.transpose(1, 2)
        net = self.fc_p(p)

        for i in range(self.num_blocks):
            net = self.blocks[i](net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))

        out = out.squeeze(1)

        return out


class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, c_dim: int, f_dim: int) -> None:
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim

        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        self.bn = nn.BatchNorm1d(f_dim, affine=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)  # type: ignore
        nn.init.zeros_(self.conv_beta.bias)  # type: ignore

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        assert x.shape[0] == c.shape[0]  # batch size
        assert c.shape[1] == self.c_dim  # embedding dim
        assert x.shape[2] == c.shape[2]  # num points

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta


        return out


class ConditionalResnetBlock1d(nn.Module):
    def __init__(self, c_dim: int, size_in: int) -> None:
        super().__init__()
        self.size_in = size_in
        self.bn_0 = ConditionalBatchNorm1d(c_dim, size_in)
        self.bn_1 = ConditionalBatchNorm1d(c_dim, size_in)

        self.fc_0 = nn.Conv1d(size_in, size_in, 1)
        self.fc_1 = nn.Conv1d(size_in, size_in, 1)

        self.actvn = nn.ReLU()

        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        return x + dx


class CbnDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        out_dim: int = 1,
        refine = False
    ) -> None:
        super().__init__()
        self.decoder = DecoderConditionalBatchNorm(
            input_dim,
            latent_dim,
            hidden_dim,
            num_hidden_layers,
            out_dim,
            refine
        )


    def forward(self, coords_emb, latent_codes) -> Tensor:
        # coords -> (B, N, 3)
        # latent_codes -> (B, D) or (B, N, D)

        if len(latent_codes.shape) == 2:
            latent_codes = repeat(latent_codes, "b d -> b r d", r=coords_emb.shape[1])
        out = self.decoder(coords_emb, latent_codes)
        return out


