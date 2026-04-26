import torch
import torch.nn as nn
import torch.nn.functional as F


class CoAttentionSingle(nn.Module):
    """Single-vector co-attention fusion for two modalities.

    Each modality is represented by one vector of shape [B, D].
    The module performs bidirectional cross-modal attention
    (x attends to y, and y attends to x), followed by FFN blocks,
    then concatenates the updated vectors and projects them back to [B, D].
    """

    def __init__(self, args, num_heads=8, dropout=0.1):
        super(CoAttentionSingle, self).__init__()
        self.embedding_dim = args.embedding_dim
        if self.embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.token_type_embeddings = nn.Embedding(2, self.embedding_dim)
        self.x_to_y_attn = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.y_to_x_attn = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.x_norm1 = nn.LayerNorm(self.embedding_dim)
        self.y_norm1 = nn.LayerNorm(self.embedding_dim)
        self.x_norm2 = nn.LayerNorm(self.embedding_dim)
        self.y_norm2 = nn.LayerNorm(self.embedding_dim)

        self.x_ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim * 4, self.embedding_dim),
        )
        self.y_ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim * 4, self.embedding_dim),
        )

        self.fuse_proj = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    def forward(self, x, y):
        if x.dim() != 2 or y.dim() != 2:
            raise ValueError(f"x and y must be 2D tensors [B, D], got {x.shape} and {y.shape}")
        if x.size(0) != y.size(0) or x.size(1) != y.size(1):
            raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")

        batch_size = x.size(0)
        device = x.device

        x = x.unsqueeze(1)
        y = y.unsqueeze(1)

        x = x + self.token_type_embeddings(torch.zeros(batch_size, 1, dtype=torch.long, device=device))
        y = y + self.token_type_embeddings(torch.ones(batch_size, 1, dtype=torch.long, device=device))

        x_attn, _ = self.x_to_y_attn(query=x, key=y, value=y, need_weights=False)
        y_attn, _ = self.y_to_x_attn(query=y, key=x, value=x, need_weights=False)

        x_attn = torch.nan_to_num(x_attn, nan=0.0, posinf=0.0, neginf=0.0)
        y_attn = torch.nan_to_num(y_attn, nan=0.0, posinf=0.0, neginf=0.0)

        x = self.x_norm1(x + x_attn)
        y = self.y_norm1(y + y_attn)

        x = self.x_norm2(x + self.x_ffn(x))
        y = self.y_norm2(y + self.y_ffn(y))

        fused = self.fuse_proj(torch.cat([x, y], dim=-1))
        return fused.squeeze(1)


class CrossAttentionSingle(nn.Module):
    """Single-vector cross-attention fusion for two modalities.

    This module implements a unidirectional cross-attention update:
    x attends to y, where x is the query modality and y provides key/value.
    The updated x is then fused with the original y and projected to [B, D].
    """

    def __init__(self, args, num_heads=8, dropout=0.1):
        super(CrossAttentionSingle, self).__init__()
        self.embedding_dim = args.embedding_dim
        if self.embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.token_type_embeddings = nn.Embedding(2, self.embedding_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.x_norm1 = nn.LayerNorm(self.embedding_dim)
        self.x_norm2 = nn.LayerNorm(self.embedding_dim)
        self.x_ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim * 4, self.embedding_dim),
        )

        self.fuse_proj = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    def forward(self, x, y):
        if x.dim() != 2 or y.dim() != 2:
            raise ValueError(f"x and y must be 2D tensors [B, D], got {x.shape} and {y.shape}")
        if x.size(0) != y.size(0) or x.size(1) != y.size(1):
            raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")

        batch_size = x.size(0)
        device = x.device

        x = x.unsqueeze(1)
        y = y.unsqueeze(1)

        x = x + self.token_type_embeddings(torch.zeros(batch_size, 1, dtype=torch.long, device=device))
        y = y + self.token_type_embeddings(torch.ones(batch_size, 1, dtype=torch.long, device=device))

        x_attn, _ = self.cross_attn(query=x, key=y, value=y, need_weights=False)
        x_attn = torch.nan_to_num(x_attn, nan=0.0, posinf=0.0, neginf=0.0)

        x = self.x_norm1(x + x_attn)
        x = self.x_norm2(x + self.x_ffn(x))

        fused = self.fuse_proj(torch.cat([x, y], dim=-1))
        return fused.squeeze(1)


class CrossAttentionSeq(nn.Module):
    def __init__(self, args, num_heads=8, dropout=0.1):
        super(CrossAttentionSeq, self).__init__()
        self.embedding_dim = args.embedding_dim
        if self.embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.token_type_embeddings = nn.Embedding(2, self.embedding_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_norm1 = nn.LayerNorm(self.embedding_dim)
        self.text_norm2 = nn.LayerNorm(self.embedding_dim)
        self.text_ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim * 4, self.embedding_dim),
        )
        self.fuse_proj = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    def forward(self, text_seq, text_mask, video_vec):
        if text_seq.dim() != 3:
            raise ValueError(f"text_seq must be [B, L, D], got {text_seq.shape}")
        if text_mask.dim() == 3 and text_mask.size(-1) == 1:
            text_mask = text_mask.squeeze(-1)
        if text_mask.dim() != 2:
            raise ValueError(f"text_mask must be [B, L], got {text_mask.shape}")
        if video_vec.dim() != 2:
            raise ValueError(f"video_vec must be [B, D], got {video_vec.shape}")
        if text_seq.size(0) != video_vec.size(0) or text_seq.size(2) != video_vec.size(1):
            raise ValueError(
                f"text_seq and video_vec size mismatch: {text_seq.shape} vs {video_vec.shape}"
            )

        batch_size, seq_len, _ = text_seq.shape
        device = text_seq.device
        text_mask = text_mask.long()

        text_seq = text_seq + self.token_type_embeddings(
            torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        )
        video_seq = video_vec.unsqueeze(1) + self.token_type_embeddings(
            torch.ones(batch_size, 1, dtype=torch.long, device=device)
        )

        attn_out, _ = self.cross_attn(
            query=text_seq,
            key=video_seq,
            value=video_seq,
            need_weights=False,
        )
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)
        text_seq = self.text_norm1(text_seq + attn_out)
        text_seq = text_seq * text_mask.unsqueeze(-1).to(text_seq.dtype)
        text_seq = self.text_norm2(text_seq + self.text_ffn(text_seq))
        text_seq = text_seq * text_mask.unsqueeze(-1).to(text_seq.dtype)

        video_expand = video_vec.unsqueeze(1).expand(-1, seq_len, -1)
        fused_tokens = self.fuse_proj(torch.cat([text_seq, video_expand], dim=-1))
        fused_tokens = fused_tokens * text_mask.unsqueeze(-1).to(fused_tokens.dtype)

        denom = text_mask.sum(dim=1, keepdim=True).clamp_min(1).to(fused_tokens.dtype)
        fused_vec = fused_tokens.sum(dim=1) / denom
        return fused_vec

class MoEFusion(nn.Module):
    """Mixture-of-Experts fusion for two or three modalities.

    Dense gate: softmax over experts and weighted sum of expert outputs.
    Experts are simple 2-layer MLPs on concatenated modality embeddings.
    """

    def __init__(self, args, num_experts=6, num_modalities=2, hidden_dim=None, dropout=0.0):
        super().__init__()
        if num_modalities not in (2, 3):
            raise ValueError("MoEFusion only supports num_modalities=2 or 3")

        self.emb_dim = args.embedding_dim
        self.num_experts = int(num_experts)
        self.num_modalities = num_modalities

        if hidden_dim is None:
            hidden_dim = self.emb_dim * 2

        self.in_dim = self.emb_dim * self.num_modalities
        self.gate = nn.Linear(self.in_dim, self.num_experts)

        experts = []
        for _ in range(self.num_experts):
            experts.append(
                nn.Sequential(
                    nn.Linear(self.in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, self.emb_dim),
                )
            )
        self.experts = nn.ModuleList(experts)

    def forward(self, x, y, z=None, return_gate: bool = False):
        if z is None and self.num_modalities == 3:
            raise ValueError("MoEFusion(num_modalities=3) requires x, y, z inputs")
        if z is not None and self.num_modalities == 2:
            raise ValueError("MoEFusion(num_modalities=2) expects only x and y")

        if z is None:
            inp = torch.cat((x, y), dim=1)
        else:
            inp = torch.cat((x, y, z), dim=1)

        gate_logits = self.gate(inp)
        gate_prob = torch.softmax(gate_logits, dim=-1)

        expert_outs = [expert(inp) for expert in self.experts]
        stacked = torch.stack(expert_outs, dim=1)
        fused = (gate_prob.unsqueeze(-1) * stacked).sum(dim=1)

        if not return_gate:
            return fused

        stats = {
            "gate_prob": gate_prob,
            "gate_entropy": (-gate_prob.clamp_min(1e-9).log() * gate_prob).sum(dim=-1).mean(),
            "gate_max": gate_prob.max(dim=-1).values.mean(),
        }
        return fused, stats


class SumFusion(nn.Module):
    def __init__(self, args):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.fc_y = nn.Linear(args.embedding_dim, args.embedding_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return output

class ConcatFusion(nn.Module):
    def __init__(self, args):
        super(ConcatFusion, self).__init__()
        self.fc_1_3 = nn.Linear(args.embedding_dim * 3, args.embedding_dim)
        self.fc_1_2 = nn.Linear(args.embedding_dim * 2, args.embedding_dim)
        self.fc_2 = nn.Linear(args.embedding_dim, args.embedding_dim)

    def forward(self, x, y, z=None):
        if z is None:
            output = torch.cat((x, y), dim=1)
            output = self.fc_2(self.fc_1_2(output))
        else:
            output = torch.cat((x, y, z), dim=1)
            output = self.fc_2(self.fc_1_3(output))
        return output

class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, args, x_film=True):
        super(FiLM, self).__init__()

        self.dim = args.embedding_dim
        self.fc = nn.Linear(args.embedding_dim, 2 * args.embedding_dim)
        self.fc_out = nn.Linear(args.embedding_dim, args.embedding_dim)

        self.x_film = x_film

    def forward(self, x, y):
        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = to_be_film + self.fc_out(output)

        return output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, args, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.fc_y = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.fc_out = nn.Linear(args.embedding_dim, args.embedding_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return output

class CoAttnFusion(nn.Module):
    def __init__(self, args, num_heads=8, dropout=0.1):
        super(CoAttnFusion, self).__init__()
        self.embedding_dim = args.embedding_dim
        if self.embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads

        self.token_type_embeddings = nn.Embedding(2, self.embedding_dim)

        self.x_to_y_attn = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.y_to_x_attn = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.x_norm1 = nn.LayerNorm(self.embedding_dim)
        self.y_norm1 = nn.LayerNorm(self.embedding_dim)
        self.x_norm2 = nn.LayerNorm(self.embedding_dim)
        self.y_norm2 = nn.LayerNorm(self.embedding_dim)

        self.x_ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim * 4, self.embedding_dim),
        )
        self.y_ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim * 4, self.embedding_dim),
        )

        self.fuse_proj = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    def _normalize_mask(self, mask, feats):
        if mask is None:
            return torch.ones(feats.size(0), feats.size(1), device=feats.device, dtype=torch.long)
        if mask.dim() == 3 and mask.size(-1) == 1:
            mask = mask.squeeze(-1)
        if mask.dim() != 2:
            raise ValueError(f"mask must have shape [B, L] or [B, L, 1], got {mask.shape}")
        if mask.size(0) != feats.size(0) or mask.size(1) != feats.size(1):
            raise ValueError(
                f"mask shape {mask.shape} must match feature shape {feats.shape[:2]}"
            )
        return mask.long()

    def _apply_mask(self, feats, mask):
        return feats * mask.unsqueeze(-1).to(feats.dtype)

    def forward(self, x, x_mask, y, y_mask):
        if x.dim() != 3 or y.dim() != 3:
            raise ValueError(f"x and y must be 3D tensors [B, L, D], got {x.shape} and {y.shape}")
        if x.size(0) != y.size(0) or x.size(2) != y.size(2):
            raise ValueError(
                f"x and y must have matching batch and hidden sizes, got {x.shape} and {y.shape}"
            )

        x_mask = self._normalize_mask(x_mask, x)
        y_mask = self._normalize_mask(y_mask, y)

        device = x.device
        x = x + self.token_type_embeddings(torch.zeros_like(x_mask, dtype=torch.long, device=device))
        y = y + self.token_type_embeddings(torch.ones_like(y_mask, dtype=torch.long, device=device))

        x_key_padding_mask = x_mask == 0
        y_key_padding_mask = y_mask == 0

        x_attn, _ = self.x_to_y_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=y_key_padding_mask,
            need_weights=False,
        )
        y_attn, _ = self.y_to_x_attn(
            query=y,
            key=x,
            value=x,
            key_padding_mask=x_key_padding_mask,
            need_weights=False,
        )

        x_attn = torch.nan_to_num(x_attn, nan=0.0, posinf=0.0, neginf=0.0)
        y_attn = torch.nan_to_num(y_attn, nan=0.0, posinf=0.0, neginf=0.0)
        x_attn = self._apply_mask(x_attn, x_mask)
        y_attn = self._apply_mask(y_attn, y_mask)

        x = self.x_norm1(x + x_attn)
        y = self.y_norm1(y + y_attn)
        x = self._apply_mask(x, x_mask)
        y = self._apply_mask(y, y_mask)

        x = self.x_norm2(x + self.x_ffn(x))
        y = self.y_norm2(y + self.y_ffn(y))
        x = self._apply_mask(x, x_mask)
        y = self._apply_mask(y, y_mask)

        fused = self.fuse_proj(torch.cat([x, y], dim=-1))
        fused_mask = torch.logical_or(x_mask.bool(), y_mask.bool()).long()
        fused = self._apply_mask(fused, fused_mask)
        return fused