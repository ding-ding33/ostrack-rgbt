from typing import Tuple

import torch
from torch import nn
import timm
import math


'''
def forward_block(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x))) * self.s
    return x


def forward_block_attn(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x
'''


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class Bi_direct_adapter(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)  
        self.adapter_up = nn.Linear(dim, 768)  
        self.adapter_mid = nn.Linear(dim, dim)

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)   
        #x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down)
        #x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  
        #print("return adap x", x_up.size())
        return x_up

"""


class Convpass(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)

        self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        #print(x.shape)
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        #print(x_down.shape)

        x_patch = x_down[:, 64:].reshape(B, 16, 16, self.dim).permute(0, 3, 1, 2)   ############
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 16 * 16, self.dim)


        #x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up
"""


class LightweightIQA_MLP(nn.Module):
    """轻量级图像质量评估模块。

    对 token 序列做全局平均池化后，通过两层 MLP + Sigmoid 输出 0~1 的质量分数。
    """

    def __init__(self, in_channels: int, reduction: int = 16,
                 act_layer: type = nn.GELU) -> None:
        super().__init__()
        hidden_channels = max(in_channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, 1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # [B, N, C] → [B, C, N] → pool → [B, C, 1] → [B, C]
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)
        hidden = self.act(self.fc1(pooled))
        score = torch.sigmoid(self.fc2(hidden))
        return score


class VIB_Feature_Purifier(nn.Module):
    """Variational information bottleneck purifier for token features.

    通过重参数化采样在训练阶段引入受控噪声；ReZero 残差使得第 0 步等价于恒等映射，
    便于在不破坏预训练权重的前提下逐步学习特征净化能力。
    """

    def __init__(self, channels: int, reduction: int = 8, beta: float = 0.001) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 1)

        self.encoder = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden_channels, channels)
        self.logvar_head = nn.Linear(hidden_channels, channels)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = beta

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """输入 token 序列并返回净化后的特征和 KL 损失。"""
        hidden = self.encoder(x)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_raw = mu + eps * std
        else:
            z_raw = mu

        out = x + self.alpha * z_raw

        if self.training:
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - torch.exp(logvar))
            kl_loss = self.beta * kl
        else:
            kl_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        return out, kl_loss


class CrossAttnRepairLayer(nn.Module):
    """单层交叉注意力修复：Q 来自待修复模态，K/V 来自完好模态。"""

    def __init__(self, dim: int = 768, num_heads: int = 8, mlp_ratio: float = 4.0,
                 drop: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm_ffn = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        B, N, C = q_tokens.shape

        q = self.q_proj(self.norm_q(q_tokens))
        k = self.k_proj(self.norm_kv(kv_tokens))
        v = self.v_proj(self.norm_kv(kv_tokens))

        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        q_tokens = q_tokens + out
        q_tokens = q_tokens + self.mlp(self.norm_ffn(q_tokens))
        return q_tokens


class CrossAttnRepair(nn.Module):
    """多层交叉注意力修复模块，ReZero 残差使得第 0 步等价于恒等映射。"""

    def __init__(self, dim: int = 768, num_heads: int = 8, num_layers: int = 2,
                 mlp_ratio: float = 4.0, drop: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttnRepairLayer(dim, num_heads, mlp_ratio, drop)
            for _ in range(num_layers)
        ])
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        out = q_tokens
        for layer in self.layers:
            out = layer(out, kv_tokens)
        return q_tokens + self.alpha * (out - q_tokens)