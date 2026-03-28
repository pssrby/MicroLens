import torch
import torch.nn as nn

class MoEFusion(nn.Module):
    """Mixture-of-Experts fusion for two or three modalities.

    Dense gate: softmax over experts and weighted sum of expert outputs.
    Experts are simple 2-layer MLPs on concatenated modality embeddings.
    """

    def __init__(self, args, num_experts=4, num_modalities=2, hidden_dim=None, dropout=0.0):
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

