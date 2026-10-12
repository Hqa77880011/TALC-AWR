import copy

import torch
from torch import nn
from torch.nn import functional as F


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).square().mean(1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + 1e-6) * self.weight + self.bias


class SharedBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = ChannelNorm(channels)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.gate_a = nn.Conv2d(channels, 2 * channels, 1)
        self.gate_b = nn.Conv2d(channels, 2 * channels, 1)
        self.project = nn.Conv2d(2 * channels, channels, 1)
        self.alpha = nn.Parameter(torch.ones(()))

    def forward(self, x):
        u = self.depthwise(self.norm(x))
        return x + self.alpha * self.project(self.gate_a(u) * self.gate_b(u).sigmoid())


class Adapter(nn.Module):
    def __init__(self, channels, rank):
        super().__init__()
        self.down = nn.Conv2d(channels, rank, 1)
        self.depthwise = nn.Conv2d(rank, rank, 3, padding=1, groups=rank)
        self.up = nn.Conv2d(rank, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.depthwise(F.gelu(self.down(x))))


class Restorer(nn.Module):
    def __init__(self, architecture, kind="talc", tasks=0, policy="learned", shared=False):
        super().__init__()
        self.architecture = copy.deepcopy(architecture)
        self.kind = kind
        self.policy = policy
        self.shared_adapters = shared
        self.tasks = 0
        self.default_top_k = 2
        widths = list(architecture["widths"])
        enc_depth = list(architecture["encoder_blocks"])
        dec_depth = list(architecture["decoder_blocks"])
        if kind in {"plain", "capacity"}:
            enc_depth[3] += 1
            dec_depth[1] += 2
        if kind == "capacity":
            widths[2:] = [144, 280]
        self.widths = widths
        self.stem = nn.Conv2d(3, widths[0], 3, padding=1)
        self.head = nn.Conv2d(widths[0], 3, 3, padding=1)
        self.encoder = nn.ModuleList(
            [nn.ModuleList([SharedBlock(c) for _ in range(n)]) for c, n in zip(widths, enc_depth)]
        )
        self.decoder = nn.ModuleList(
            [nn.ModuleList([SharedBlock(c) for _ in range(n)]) for c, n in zip(widths, dec_depth)]
        )
        self.down = nn.ModuleList(
            [nn.Conv2d(a, b, 3, stride=2, padding=1) for a, b in zip(widths, widths[1:])]
        )
        self.up = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(b, 4 * a, 1), nn.PixelShuffle(2))
                for a, b in zip(widths, widths[1:])
            ]
        )
        self.fuse = nn.ModuleList([nn.Conv2d(2 * c, c, 1) for c in widths[:3]])
        self.sites = {}
        if kind == "talc":
            for prefix, key in [("e", "encoder_sites"), ("d", "decoder_sites")]:
                for level, indices in enumerate(architecture[key]):
                    for index in indices:
                        self.sites[f"{prefix}{level}_{index - 1}"] = level
        self.last_sites = {}
        for name, level in self.sites.items():
            self.last_sites[level] = name
        self.experts = nn.ModuleList()
        self.router = None
        if kind == "talc" and policy != "uniform":
            hidden = architecture["router_hidden"]
            self.router = nn.Sequential(nn.Linear(widths[0] + widths[1], hidden), nn.GELU())
        for _ in range(tasks):
            self.add_task()

    def add_task(self):
        self.tasks += 1
        if self.kind != "talc":
            return
        if not self.shared_adapters or not self.experts:
            self.experts.append(
                nn.ModuleDict(
                    {
                        name: Adapter(self.widths[level], self.architecture["rank_base"] * 2**level)
                        for name, level in self.sites.items()
                    }
                )
            )
        if self.router is not None:
            hidden = self.architecture["router_hidden"]
            old = self.router[2] if len(self.router) == 3 else None
            output = nn.Linear(hidden, self.tasks).to(self.stem.weight.device)
            if old is not None:
                with torch.no_grad():
                    output.weight[: self.tasks - 1].copy_(old.weight)
                    output.bias[: self.tasks - 1].copy_(old.bias)
                self.router[2] = output
            else:
                self.router.append(output)

    def probe(self, x):
        x = self.stem(x)
        for block in self.encoder[0]:
            x = block(x)
        a = x.mean((2, 3))
        x = self.down[0](x)
        for block in self.encoder[1]:
            x = block(x)
        return torch.cat([a, x.mean((2, 3))], 1)

    def configure_trainable(self, unfreeze_old=False):
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        if self.kind != "talc" or self.tasks == 1:
            return
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in self.modules():
            if isinstance(module, ChannelNorm):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        for level in self.decoder:
            for block in level:
                for module in [block.gate_a, block.gate_b, block.project]:
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
        for module in [self.head, self.router]:
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        experts = self.experts if unfreeze_old or self.shared_adapters else [self.experts[-1]]
        for expert in experts:
            for parameter in expert.parameters():
                parameter.requires_grad_(True)

    def _route(self, x, top_k, forced):
        if self.router is None:
            probabilities = x.new_full((len(x), self.tasks), 1 / self.tasks, dtype=torch.float32)
            return probabilities, probabilities, None
        logits = self.router(self.probe(x)).float()
        probabilities = logits.softmax(1)
        if forced is None:
            indices = logits.topk(min(top_k, self.tasks), 1).indices
        else:
            indices = forced.to(device=x.device, dtype=torch.long)
            if indices.ndim == 1:
                indices = indices.expand(len(x), -1)
            if indices.shape[0] != len(x) or indices.min() < 0 or indices.max() >= self.tasks:
                raise ValueError("Forced expert indices do not match the task bank")
        weights = logits.gather(1, indices).softmax(1)
        sparse = torch.zeros_like(logits).scatter(1, indices, weights)
        return sparse, probabilities, logits

    def _block(self, block, x, name, weights, orth):
        result = block(x)
        if name not in self.sites:
            return result, x.new_zeros(())
        penalty = x.new_zeros((), dtype=torch.float32)
        if self.shared_adapters:
            return result + self.experts[0][name](x), penalty
        for task, expert in enumerate(self.experts):
            indices = torch.nonzero(weights[:, task] > 0, as_tuple=True)[0]
            if len(indices):
                residual = expert[name](x.index_select(0, indices))
                scale = weights.index_select(0, indices)[:, task, None, None, None]
                result = result.index_add(0, indices, residual * scale.to(residual.dtype))
        level = self.sites[name]
        if orth and self.tasks > 1 and self.last_sites[level] == name:
            columns = []
            for task, expert in enumerate(self.experts):
                residual = expert[name](x).float().flatten(1)
                if task < self.tasks - 1:
                    residual = residual.detach()
                columns.append(F.normalize(residual, dim=1, eps=1e-6))
            q = torch.stack(columns, 2)
            with torch.autocast(device_type=x.device.type, enabled=False):
                gram = q.transpose(1, 2) @ q
            mask = ~torch.eye(self.tasks, device=x.device, dtype=torch.bool)
            penalty = gram[:, mask].square().sum(1).mean() / (self.tasks * (self.tasks - 1))
        return result, penalty

    def forward(self, x, top_k=None, forced=None, orth=False):
        top_k = self.default_top_k if top_k is None else top_k
        height, width = x.shape[-2:]
        pad_h, pad_w = (-height) % 8, (-width) % 8
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        weights = probabilities = logits = None
        if self.kind == "talc":
            if not self.tasks:
                raise ValueError("Add at least one task before restoration")
            weights, probabilities, logits = self._route(x, top_k, forced)
        features = [None] * 4
        skips = []
        penalty = x.new_zeros((), dtype=torch.float32)
        z = self.stem(x)
        for level, blocks in enumerate(self.encoder):
            for index, block in enumerate(blocks):
                z, term = self._block(block, z, f"e{level}_{index}", weights, orth)
                penalty = penalty + term
            if level < 3:
                skips.append(z)
                z = self.down[level](z)
        features[3] = z
        for level in [2, 1, 0]:
            z = self.fuse[level](torch.cat([self.up[level](z), skips[level]], 1))
            for index, block in enumerate(self.decoder[level]):
                z, term = self._block(block, z, f"d{level}_{index}", weights, orth)
                penalty = penalty + term
            features[level] = z
        restored = (x + self.head(z)).clamp(0, 1)[..., :height, :width]
        return {
            "image": restored,
            "features": features,
            "logits": logits,
            "probabilities": probabilities,
            "orthogonality": penalty,
        }


def build_model(config, variant, tasks):
    kind = "plain"
    if variant.startswith("talc"):
        kind = "talc"
    elif variant == "derpp_capacity":
        kind = "capacity"
    model = Restorer(
        config["model"],
        kind,
        tasks,
        policy="uniform" if variant == "talc_uniform" else "learned",
        shared=variant == "talc_shared",
    )
    model.default_top_k = 1 if variant == "talc_top1" else 2
    return model
