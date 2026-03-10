import torch
import torch.nn as nn


class MoEFusion(nn.Module):
    """Mixture-of-Experts fusion for two modalities.

    Dense gate: softmax over all experts and weighted sum of expert outputs.
    Experts are simple 2-layer MLPs that operate on concatenated features.

    Forward returns fused embedding by default; if return_gate=True it also
    returns a dict with routing stats for logging.
    """

    def __init__(self, args, num_experts: int = 4, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.emb_dim = args.embedding_dim
        self.num_experts = int(num_experts)
        if hidden_dim is None:
            hidden_dim = self.emb_dim * 2

        in_dim = self.emb_dim * 2
        self.gate = nn.Linear(in_dim, self.num_experts)

        experts = []
        for _ in range(self.num_experts):
            experts.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, self.emb_dim),
                )
            )
        self.experts = nn.ModuleList(experts)

    def forward(self, x, y, return_gate: bool = False):
        inp = torch.cat((x, y), dim=1)
        gate_logits = self.gate(inp)
        gate_prob = torch.softmax(gate_logits, dim=-1)  # (B, E)

        expert_outs = []
        for expert in self.experts:
            expert_outs.append(expert(inp))
        stacked = torch.stack(expert_outs, dim=1)  # (B, E, D)

        fused = (gate_prob.unsqueeze(-1) * stacked).sum(dim=1)  # (B, D)

        if not return_gate:
            return fused

        # simple stats for monitoring; no extra loss by default
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
        self.fc_1 = nn.Linear(args.embedding_dim * 2, args.embedding_dim)
        self.fc_2 = nn.Linear(args.embedding_dim, args.embedding_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_2(self.fc_1(output))
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
        output = self.fc_out(output)

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

