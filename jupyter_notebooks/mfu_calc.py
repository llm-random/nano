"""Internals of mfu_calculator.ipynb: parameter count, FLOPs and the MFU / time / steps solver."""

from dataclasses import dataclass

PEAK_BF16 = {
    "h100": 989e12,
    "gh200": 989e12,
    "a100": 312e12,
}  # dense bf16, as in src/core/mfu.py


@dataclass
class Setup:
    n_layers: int
    dmodel: int
    dff: int  # per expert
    q_heads: int
    kv_heads: int
    ff_matrices: int  # 3 for gated FF (SwiGLU, GatedFeedForward), 2 for GELU
    n_experts: int
    topk: int  # experts active per token
    vocab: int
    sl: int
    bs: int  # global batch, all GPUs together
    gpu: str  # key of PEAK_BF16
    n_gpus: int
    include_attention_scores: bool  # False reproduces the old 6 * N notebooks

    def _layer_params(self, experts):
        dhead = self.dmodel // self.q_heads
        attention = (
            2 * self.dmodel**2 + 2 * self.dmodel * self.kv_heads * dhead
        )  # q, o + k, v
        router = self.dmodel * self.n_experts if self.n_experts > 1 else 0
        return attention + experts * self.ff_matrices * self.dmodel * self.dff + router

    @property
    def total_params(self):
        return (
            self.n_layers * self._layer_params(self.n_experts)
            + 2 * self.vocab * self.dmodel
        )

    @property
    def active_params(self):
        # the head counts, the embedding lookup doesn't
        return self.n_layers * self._layer_params(self.topk) + self.vocab * self.dmodel

    @property
    def score_flops(self):
        if not self.include_attention_scores:
            return 0
        return 12 * self.n_layers * self.sl * self.dmodel

    @property
    def flops_per_token(self):
        return (
            6 * self.active_params + self.score_flops
        )  # convention of src/core/mfu.py

    @property
    def flops_per_step(self):
        return self.flops_per_token * self.sl * self.bs

    @property
    def peak(self):
        return PEAK_BF16[self.gpu] * self.n_gpus

    def describe(self):
        print(
            f"{self.active_params / 1e6:.1f}M active / {self.total_params / 1e6:.1f}M total params | "
            f"{self.flops_per_token / 1e9:.3f} GFLOPs/token "
            f"({self.score_flops / self.flops_per_token:.1%} attention) | "
            f"{self.n_gpus}x {self.gpu} = {self.peak / 1e15:.2f} PFLOP/s peak"
        )

    def solve(self, mfu=None, time_h=None, steps=None):
        """Pass exactly two of mfu (0-1), time_h (hours), steps; prints the third first."""
        given = dict(mfu=mfu, time_h=time_h, steps=steps)
        missing = [name for name, value in given.items() if value is None]
        if len(missing) != 1:
            raise ValueError("pass exactly two of mfu, time_h, steps")

        if mfu is None:
            mfu = self.flops_per_step * steps / (time_h * 3600 * self.peak)
        elif time_h is None:
            time_h = self.flops_per_step * steps / (mfu * self.peak) / 3600
        else:
            steps = mfu * self.peak * time_h * 3600 / self.flops_per_step

        tokens = steps * self.sl * self.bs
        parts = {
            "mfu": f"MFU {mfu:.2%}",
            "time_h": f"time {time_h:.2f} h = {time_h / 24:.2f} d",
            "steps": f"{steps:,.0f} steps",
        }
        inputs = ", ".join(text for name, text in parts.items() if name != missing[0])
        print(
            f"{parts[missing[0]]:<22} <- {inputs} | "
            f"{tokens / 1e9:.2f}B tokens ({tokens / self.active_params:.1f}TTP), "
            f"{tokens / (time_h * 3600):,.0f} tok/s"
        )
