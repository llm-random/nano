"""AME -- Attention Memory Embedding: token-id-indexed Q/K banks instead of W_Q, W_K.

The method, in one line: every head-layer carries two banks `E_Q, E_K` of shape
`[vocab, d_head]`; `q_i = E_Q[t_i]` and `k_j = E_K[t_j]` are *gathers by token id*,
not projections of the residual stream. Values stay an ordinary `W_V @ x`. Scores
therefore depend on **token ids and distance only**:

    s_ij = tau * T[t_i, t_j] - m_h * (i - j),      T[a, b] = norm(E_Q[a]) . norm(E_K[b])

which is the whole design: it makes the score table precomputable after training
(the decode-time payoff, not built here) and it is also the source of every
weakness -- the weights never read a hidden state, so retrieval is never
content-addressed. See `documents/notes/ame_implementation.md` for the full
rationale and the research record it comes from.

Training is **unmodified O(n^2) attention with two gathers**. There is no score
table, no row cache, no V-cache and no CPU offload in this file; those are
decode-time machinery for a later stage and must not be built first.

Five things here are load-bearing and easy to get subtly wrong:

1. **The ALiBi distance bias is not optional.** Without it the layer is
   permutation-invariant over the prefix -- a bag of token types. It is the only
   positional signal AME has; there is no RoPE anywhere.
2. **QK-norm goes on the bank rows, *before* the KV-shift mix** -- never on the
   mixed key. Normalizing the mix would make a key's norm a function of the
   bigram `(t_{j-1}, t_j)`, which destroys the property that a score is a linear
   combination of two entries of one row of `T`, i.e. the entire decode-time win.
3. **Bank rows initialize inside the sigma plateau: per-element sigma = 1e-4
   by default, with the QK-norm.** Under it a row's scale never reaches the
   scores, but it sets how far one Adam step turns the row, so sigma acts as the
   banks' inverse learning rate. At h12 every sigma from 1e-5 to 0.0178 ties;
   sigma = 1 (the default until 2026-11-25) is 0.097 worse, and the init fails
   at 1e-7 and below and at 0.0316 and above
   (documents/notes/archive/ame_bank_sigma_grid.md). Without the QK-norm the scores
   scale with sigma^2, so the default there stays 1.0: 1e-4, like the embedding
   habit of 0.02 (scores of std ~4e-4), would start attention uniform.
4. **The shift must not wrap.** `_shift1` pads and slices; `torch.roll` would mix
   the last token of a packed row into position 0 of the next one, which corrupts
   a run silently and surfaces only as "the shift didn't help".
5. **alpha and beta are different mechanisms** and are pinned separately, because
   a steep-slope head is already a previous-token head and supplies beta for free.
   Pinning them together confounds the two.

KV shifting (`alpha`, `beta`) is prior art -- Xu, Cheng, Wang & Chen, arXiv
2411.19574 -- not this project's contribution. Any dense baseline AME is compared
against must have it too, or AME collects someone else's gain.

**Bank gates** (`bank_gate`, documents/notes/archive/ame_bank_gates_plan.md). A bank row
gets a gradient only in steps where its token occurs (AdamW's momentum carries
it further for a few dozen steps after), yet under the hard QK-norm a rarely
trained row acts at full strength. Each gate changes how much a row's identity
reaches the scores; "none" is the model above, bit for bit:

    confidence  (B1)  q = sigmoid(c_Q[t]) * norm(E_Q[t]), the same for k; c is
                      one trust logit per token and head, stored as H extra
                      bank columns and started at logit(confidence_init),
                      0.05 unless set
    context     (B2)  q = sigmoid(W_g x_i + b) * norm(E_Q[t]), one scalar per
                      query position and head; W_g = 0, b = 3 at init
    backoff     (B3)  q = norm((base_Q + E_Q[t]) / sqrt 2), the same for k; one
                      shared vector per head, started at zero
    stem        (ST)  q = silu(W_g x_i) * norm(E_Q[t]), per channel, as STEM's
                      SwiGLU gate scales its table row

`context` and `stem` read the hidden state, so their scores are no longer a
function of ids and distance alone; the instrumentation that assumes they are
(`score_matrix`, `attention_diagnostics`) refuses them rather than feed x = 0.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.init_rng import private_generator, private_rand
from src.core.model import Linear, llm_random_weight_init

# The values `bank_gate` takes. One enum rather than a boolean per mechanism, so
# that "one mechanism per run" is structural.
BANK_GATES = ("none", "confidence", "context", "backoff", "stem")
# B1's default `confidence_init`: every trust scalar starts at 0.05, so every
# score starts at 0.05 * 0.05 = 0.0025 of its ungated size: attention begins as
# the ALiBi prior plus the sink, until a token's own updates earn it trust.
# Stage 2's b1open_std001 starts it at 0.95 instead, so scores start at 0.9025
# of their size (documents/notes/archive/ame_bank_gates_stage2_plan.md).
CONFIDENCE_INIT = 0.05
# B2: W_g = 0 and b = 3 open every gate to sigmoid(3) = 0.953, the strongest of
# Highway networks' carry settings (a transform-gate bias of -1 to -3,
# Srivastava, Greff & Schmidhuber 2015). Not tuned.
CONTEXT_GATE_INIT_BIAS = 3.0
# The banks' default init sigma depends on the QK-norm
# (documents/notes/archive/review_issues.md, N1). Under it a row's scale never reaches the
# scores and only sets how far one Adam step turns the row: 1e-4 sits inside the
# plateau the sigma grid found (1e-5 to 0.0178, documents/notes/archive/ame_bank_sigma_grid.md).
# Without it the scores scale with sigma^2 and tau is frozen at 1/sqrt(dhead), so
# 1e-4 would start attention uniform; sigma ~ 1 puts the scores at std ~1 there.
BANK_INIT_STD = 1.0e-4
BANK_INIT_STD_WITHOUT_QK_NORM = 1.0

# Horizon of the steepest head, in tokens: a token at distance d competes only
# while m*d <~ tau, so the ladder is specified in horizons (tau/m) and converted
# to slopes once tau is known. 16 tokens is a near-local head.
DEFAULT_MIN_HORIZON = 16.0
# Horizon of the flattest head, as a multiple of the training context. The
# standard ALiBi ladder (2^-0.5 ... 2^-8) would cap the flattest head at ~3k
# tokens whatever the context is, which silently limits a long-context method.
DEFAULT_HORIZON_FACTOR = 4.0


def _shift1(z):
    """`z[:, j] <- z[:, j-1]`, position 0 zeroed. Shape `[B, n, ...]`, any rank.

    Pad-and-slice, deliberately not `torch.roll`: roll wraps, and under the
    document packing this repo uses that would mix the end of one packed row
    into the start of the next.

    F.pad takes its pairs from the last dimension backwards, so the pad list is
    built from the rank. A hard-coded `(0, 0, 0, 0, 1, 0)` reaches dim 1 only
    for a 4-D input: given a 3-D one it pads the *batch* dimension instead and
    returns a wrong shape with no error at all -- `(2, 4, 3)` comes back as
    `(3, 3, 3)`. Only `[B, n, H, D]` is passed today, but the raw-`v` register
    of ame.md 4.1 is the next caller and would hit exactly that.
    """
    pad = [0, 0] * (z.dim() - 2) + [1, 0]
    return F.pad(z, pad)[:, :-1]


def _with_trust_columns(bank, heads, init=CONFIDENCE_INIT):
    """B1: `bank`, `heads` columns wider, the new columns at logit(`init`).

    Built from the bank as already drawn, so the first `H*d` columns are exactly
    the ungated model's draw and the global RNG is not touched: every arm then
    starts every shared weight from the same numbers, and differs from "none"
    only in its mechanism. `_weight=` keeps nn.Embedding from re-drawing.

    The logit is the expression the module constant used to be, so the default
    0.05 writes the same fp32 value as stage 1's B1, bit for bit.
    """
    vocab, width = bank.weight.shape
    wide = bank.weight.new_empty(vocab, width + heads)
    if not wide.is_meta:
        with torch.no_grad():
            wide[:, :width] = bank.weight
            wide[:, width:] = math.log(init / (1.0 - init))
    return nn.Embedding(vocab, width + heads, _weight=wide)


def _rng_neutral_linear(in_features, out_features, bias, device):
    """An nn.Linear whose construction draws nothing from the global RNG.

    nn.Linear's own reset_parameters draws a Kaiming init; on the meta device it
    draws nothing, and `to_empty` then allocates on the real device. The caller
    fills the weights. Keeping the global stream untouched is what keeps every
    later parameter (this layer's v_proj, the next blocks) identical across the
    gate arms.
    """
    layer = nn.Linear(in_features, out_features, bias=bias, device="meta")
    return layer.to_empty(device=device)


def _full(tensor):
    """A possibly-sharded parameter as its global value.

    FSDP2 hands back DTensors, and the instrumentation wants the whole head
    ladder rather than this rank's slice of it. `full_tensor()` is a collective,
    so every caller must run on every rank -- which `Trainer.log_metrics` does
    (the rank gate lives inside the metric logger, not around the call).
    """
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor


def _diagnostic_rows(seq_len, max_rows, device=None):
    """Which query rows the attention diagnostics look at.

    At most `max_rows` of them, spread evenly across the sequence so the sample
    spans every position regime rather than clustering at the start. The cap is
    what keeps a quadratic diagnostic off the logging path: at h20 the full
    matrix is `[B, H, 2048, 2049]` and 128 rows is about 1/16th of one row-block
    of it.

    Duplicates are dropped, so a `max_rows` above `seq_len` is simply every row.
    """
    count = min(max_rows, seq_len)
    rows = torch.linspace(0, seq_len - 1, count, device=device).long()
    return torch.unique(rows)


_COMPILED_FLEX = None


def _flex_attention(q, k, v, **kwargs):
    """`flex_attention`, compiled -- which is not optional, it is the kernel.

    Uncompiled, `flex_attention` dispatches to `sdpa_dense` ->
    `_math_attention_inner`, which materializes the whole `[B, H, Q, KV]` score
    tensor, applies `score_mod` elementwise under `vmap`, prunes nothing from the
    block mask and re-materializes in the backward. Measured on CPU that is
    1.9-3.4x slower than this layer's own naive reference path, so it is worse
    than no kernel at all. torch 2.9 warns; the pinned 2.7.1 does not, which is
    how the first h20 run spent an hour on the unfused path in silence.

    Compiled here rather than left to a config flag on purpose: the failure was
    exactly that every GPU test compiled the layer itself while no config did,
    so the tests proved a path the runs never took.

    When the caller is *already* tracing -- `compile_blocks_full`,
    `compile_model` -- the surrounding graph lowers `flex_attention` itself, so
    hand it the bare function instead of nesting one torch.compile inside
    another.
    """
    from torch.nn.attention.flex_attention import flex_attention

    # Already tracing: the surrounding graph lowers this call itself.
    # CPU: a compiled flex_attention whose inputs require grad cannot be lowered
    # at all (the same reason `backend="auto"` picks reference there). CPU only
    # reaches this path from the backend-equivalence tests, where the question
    # is whether the two paths agree, not how fast the slow one is.
    if torch.compiler.is_compiling() or not q.is_cuda:
        return flex_attention(q, k, v, **kwargs)

    global _COMPILED_FLEX
    if _COMPILED_FLEX is None:
        # dynamic=False: seq_len is fixed per run, and a dynamic shape would
        # re-specialize the kernel on the first eval batch of a different width.
        _COMPILED_FLEX = torch.compile(flex_attention, dynamic=False)
    return _COMPILED_FLEX(q, k, v, **kwargs)


def horizon_ladder(n_heads, tau, seq_len, min_horizon, horizon_factor):
    """Per-head ALiBi slopes as a geometric ladder, steepest head first.

    Specified through horizons rather than raw slopes: head h competes out to
    roughly `tau / m_h` tokens, so a ladder from `min_horizon` to
    `horizon_factor * seq_len` is readable as "this head sees 16 tokens, that one
    sees four contexts". Returns slopes `[n_heads]`, float32.
    """
    max_horizon = horizon_factor * seq_len
    if n_heads == 1:
        return torch.tensor([tau / max_horizon], dtype=torch.float32)
    exponents = torch.linspace(
        math.log2(min_horizon), math.log2(max_horizon), n_heads, dtype=torch.float32
    )
    return tau / torch.pow(2.0, exponents)


class AMEAttention(nn.Module):
    """One AME attention layer: id-indexed Q/K banks, ordinary values.

    Declares `wants_token_ids`, which is how `src.core.model.Residual` knows to
    hand it the batch's token ids alongside the normalized residual stream.

    Shape contract is the registry's: `(B, n, dmodel) -> (B, n, dmodel)`. AME is
    MHA-shaped -- every query head has its own bank slice and its own value head
    -- so the config's `kv_heads` does not apply and the builder ignores it, the
    same way `mha` does.

    `rope_base` / `rope_scale_freqs` are accepted to satisfy the shared
    constructor contract and are **ignored on purpose**: the distance bias is the
    only positional signal in this layer.
    """

    wants_token_ids = True

    # 2-D by shape, but not matmul weights: src.core.mfu would otherwise charge
    # 6*P of matmul FLOPs for per-head scalars and one key vector. The B3 base
    # vectors are added to the bank rows, not multiplied into anything. Names a
    # layer does not have (no sink, no backoff gate) are skipped.
    non_matmul_parameters = ("alpha", "beta", "k_sink", "v_sink", "q_base", "k_base")

    # Scales and per-head scalars, which decoupled weight decay must not touch.
    # AdamW's decay moves a parameter whose gradient is zero, so at wd=0.1 it
    # drives alpha_2 and beta_2 toward 0 on its own -- hardest in the head with
    # the weakest gradient, i.e. the head that is indifferent to the shift. That
    # is exactly the signal ame_experiments.md 8.1 reads as "the model declines
    # the mechanism", so leaving decay on makes the project's first experiment
    # measure the optimizer. tau is here for a second reason: `slopes` is frozen
    # at tau_init, and a head's horizon is tau/m, so a decaying tau silently
    # contracts every head's reach while the ladder stays put.
    #
    # Deliberately a separate list from `non_matmul_parameters`: that one answers
    # "is this a matmul weight" for FLOP accounting, this one answers "should
    # decay touch it". They overlap today and are not the same question.
    #
    # The banks are here too, decided 2026-11-03
    # (archive/ame_review_and_fixes.md 2.6). With
    # QK-norm a bank row's length does not reach the score at all -- only its
    # direction does -- so decay on the banks changes no output. What it does do
    # is shrink ||row|| every step, which inflates the effective learning rate on
    # row direction and destroys Stage 1's "rare-type rows stay at init"
    # diagnostic: a row that is never gradient-updated does not stay at init, it
    # decays toward zero while keeping its random direction. The cost is that
    # AME's banks are then regularized differently from the dense baseline's
    # W_Q/W_K, which baselines.md asks to be matched -- say so when reporting.
    #
    # The bank gates (ame_bank_gates_plan.md): B1's trust logits are bank
    # columns, so the bank entry already covers them. B2's gate bias is here
    # because decay would pull every gate toward sigmoid(0) = 0.5; its weight
    # decays like any Linear weight, and so does ST's gate (no bias). B3's base
    # vectors are scales of the same kind as the sink. Names a layer does not
    # have -- or, for ST's bias-free gate, `gate.bias` resolving to None -- are
    # skipped by parameter_groups.
    no_weight_decay_parameters = (
        "tau",
        "alpha",
        "beta",
        "slopes",
        "k_sink",
        "v_sink",
        "q_bank.weight",
        "k_bank.weight",
        "gate.bias",
        "q_base",
        "k_base",
    )

    # The memory banks, named for the optimizer: src.core.model.parameter_groups
    # gives them a group of their own at trainer.bank_lr_ratio x the base lr
    # (src.core.memory_parallel's controller applies the same ratio when it
    # steps them instead).
    bank_parameters = ("q_bank.weight", "k_bank.weight")

    def __init__(
        self,
        dmodel,
        q_heads,
        kv_heads,
        seq_len,
        rope_base=None,
        rope_scale_freqs=None,
        init_scale=1,
        *,
        vocab_size,
        dhead=None,
        qk_norm=True,
        bank_init_std=None,
        bank_gate="none",
        confidence_init=CONFIDENCE_INIT,
        shift=True,
        pin_alpha=False,
        pin_beta=False,
        sink=True,
        min_horizon=DEFAULT_MIN_HORIZON,
        horizon_factor=DEFAULT_HORIZON_FACTOR,
        slopes=None,
        learnable_slopes=False,
        backend="auto",
    ):
        super().__init__()
        if backend not in ("auto", "flex", "reference"):
            raise ValueError(
                f"unknown ame backend {backend!r}; expected auto, flex or reference"
            )
        if bank_gate not in BANK_GATES:
            raise ValueError(
                f"unknown ame bank_gate {bank_gate!r}; expected one of "
                f"{', '.join(BANK_GATES)}"
            )
        if not 0.0 < confidence_init < 1.0:
            raise ValueError(
                f"ame confidence_init is B1's starting trust, a sigmoid value; it "
                f"must lie strictly between 0 and 1, got {confidence_init!r}"
            )
        if bank_gate != "confidence" and confidence_init != CONFIDENCE_INIT:
            # Not silently ignored: a run that sets it believes it changed B1.
            raise ValueError(
                f"ame confidence_init={confidence_init!r} applies only to "
                f"bank_gate=confidence, got bank_gate={bank_gate!r}"
            )
        self.bank_gate = bank_gate
        self.confidence_init = confidence_init
        self.dmodel = dmodel
        self.q_heads = q_heads
        # Kept so the registry's introspection (and get_model_dimensions) finds
        # the same attribute names it finds on RoPEAttention.
        self.kv_heads = q_heads
        self.dhead = dhead if dhead is not None else dmodel // q_heads
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.qk_norm = qk_norm
        self.use_sink = sink
        self.backend = backend

        datt = self.q_heads * self.dhead

        # The banks. nn.Embedding because that is exactly what they are -- a
        # gather by id -- and because FSDP and the MFU accounting then treat them
        # as the vocab-sized lookups they are rather than as matmul weights.
        self.q_bank = nn.Embedding(vocab_size, datt)
        self.k_bank = nn.Embedding(vocab_size, datt)
        # Set by src.core.memory_parallel to a module returning BOTH banks'
        # rows from one gather. None in every ordinary run, including every
        # measurement in ame_implementation.md, so the default path is the one
        # those numbers were taken on.
        self.qk_bank = None
        if bank_init_std is None:
            bank_init_std = BANK_INIT_STD if qk_norm else BANK_INIT_STD_WITHOUT_QK_NORM
        self.bank_init_std = bank_init_std
        with torch.no_grad():
            for bank in (self.q_bank, self.k_bank):
                if not bank.weight.is_meta:
                    bank.weight.normal_(mean=0.0, std=bank_init_std)
        if bank_gate == "confidence":
            # B1: H trust logits appended to every row (width H*d + H), all at
            # logit(confidence_init). Widened after the draw above, so the row
            # part is the ungated draw.
            self.q_bank = _with_trust_columns(self.q_bank, q_heads, confidence_init)
            self.k_bank = _with_trust_columns(self.k_bank, q_heads, confidence_init)

        # Values and output stay exactly as in MHA.
        linear = partial(
            Linear,
            partial_init_fn=partial(llm_random_weight_init, scale=init_scale),
        )
        self.v_proj = linear(dmodel, datt)
        self.o_proj = linear(datt, dmodel)

        # Learned per-head temperature. Replaces 1/sqrt(dhead): two random unit
        # vectors in d dims have dot-product std 1/sqrt(d), so tau = sqrt(dhead)
        # puts the scores at std ~1 at init. Without QK-norm there is nothing to
        # scale and the usual 1/sqrt(dhead) applies.
        tau_init = math.sqrt(self.dhead) if qk_norm else 1.0 / math.sqrt(self.dhead)
        self.tau = nn.Parameter(
            torch.full((q_heads,), tau_init, dtype=torch.float32),
            requires_grad=qk_norm,
        )

        # KV shifting: per-head (c1, c2) mixing a position with its predecessor.
        # Pinned coefficients become buffers, not parameters, so an ablation arm
        # cannot drift off its pin and the parameter count reports the truth.
        self._init_mix("alpha", pinned=pin_alpha or not shift)
        self._init_mix("beta", pinned=pin_beta or not shift)

        # ALiBi slopes. Fixed geometric by default; learnable is an ablation, and
        # it is the reason the flex backend exists -- flash's alibi_slopes
        # argument takes a non-differentiable tensor.
        if slopes is None:
            # tau from its known init value, not read off the tensor: the
            # model may be built on the meta device, where there is no value.
            ladder = horizon_ladder(
                q_heads,
                tau_init if qk_norm else 1.0,
                seq_len,
                min_horizon,
                horizon_factor,
            )
        else:
            ladder = torch.as_tensor(list(slopes), dtype=torch.float32)
            if ladder.shape != (q_heads,):
                raise ValueError(
                    f"ame slopes must have one entry per head: got {tuple(ladder.shape)} "
                    f"for {q_heads} heads"
                )
        if learnable_slopes:
            self.slopes = nn.Parameter(ladder)
        else:
            self.register_buffer("slopes", ladder, persistent=True)

        # A position-independent learned sink key, so a head that learns
        # alpha_1 -> 0 (a fully shifted head, logit 0 at position 0) has
        # somewhere to put its mass, and so the distance bias cannot evict the
        # sink the model would otherwise learn on BOS. Zero-init is neutral: the
        # logit is 0, in the middle of a score distribution of std ~1.
        if sink:
            self.k_sink = nn.Parameter(torch.zeros(q_heads, self.dhead))
            self.v_sink = nn.Parameter(torch.zeros(q_heads, self.dhead))
        self._block_mask_cache = {}
        # Last, so every parameter above keeps its place in the state dict, and
        # none of them draws from the global RNG (see the helpers' docstrings).
        self._init_gate(bank_gate, init_scale)

    def _init_gate(self, bank_gate, init_scale):
        """The parameters of the B2, B3 and ST gates. B1's live in the banks."""
        device = self.v_proj.weight.device
        datt = self.q_heads * self.dhead
        if bank_gate == "context":
            # B2: one sigmoid gate per head, read off the query's hidden state.
            self.gate = _rng_neutral_linear(self.dmodel, self.q_heads, True, device)
            if not self.gate.weight.is_meta:
                with torch.no_grad():
                    self.gate.weight.zero_()
                    self.gate.bias.fill_(CONTEXT_GATE_INIT_BIAS)
        elif bank_gate == "backoff":
            # B3: one shared vector per head, mixed into every row before the
            # norm. Zero at init: the same forward as an ungated run.
            self.q_base = nn.Parameter(torch.zeros(self.q_heads, self.dhead))
            self.k_base = nn.Parameter(torch.zeros(self.q_heads, self.dhead))
        elif bank_gate == "stem":
            # ST: a dmodel -> H*d gate under silu, initialized by the rule the
            # SwiGLU gate uses, drawn from a private generator.
            self.gate = _rng_neutral_linear(self.dmodel, datt, False, device)
            if not self.gate.weight.is_meta:
                # On the CPU, where the private generator lives, even when the
                # model is built under a CUDA default device; then copied over.
                draw = torch.empty(datt, self.dmodel, device="cpu")
                llm_random_weight_init(self.dmodel, init_scale)(
                    draw, generator=private_generator("stem")
                )
                with torch.no_grad():
                    self.gate.weight.copy_(draw)

    @property
    def scores_read_hidden_state(self):
        """True for the gates that read x (B2, ST): their scores are no longer a
        function of token ids and distance alone."""
        return self.bank_gate in ("context", "stem")

    def _init_mix(self, name, pinned):
        """alpha/beta as `[H, 2]`: `(c1, c2)` per head, kept in fp32.

        Unpinned: `c1 ~ U(0, 1)`, `c2 = 1 - c1`, which makes both degenerate
        settings reachable -- `(1, 0)` is the unshifted head and `(0, 1)` the
        fully shifted one -- so "which heads are shifted" is not a hyperparameter.
        Pinned: a frozen `(1, 0)` buffer, i.e. no shift at all.

        `c1` comes from a private generator (src.core.init_rng), so the shift
        switch and the pins move no other weight's init.
        """
        if pinned:
            pin = torch.zeros(self.q_heads, 2, dtype=torch.float32)
            pin[:, 0] = 1.0
            self.register_buffer(name, pin, persistent=True)
            return
        c1 = private_rand(self.q_heads, 1, tag=f"mix.{name}")
        setattr(self, name, nn.Parameter(torch.cat([c1, 1.0 - c1], dim=-1)))

    @property
    def alpha_is_pinned(self):
        return not isinstance(self.alpha, nn.Parameter)

    @property
    def beta_is_pinned(self):
        return not isinstance(self.beta, nn.Parameter)

    def horizons(self):
        """`tau / m` per head, in tokens -- the instrumentation readout that
        catches a slope ladder silently capping the context."""
        tau = _full(self.tau).detach().float()
        scale = tau if self.qk_norm else torch.ones_like(tau)
        return (scale / _full(self.slopes).detach().float()).cpu()

    @torch.no_grad()
    def instrumentation(self, token_ids=None):
        """Per-layer scalars for the metric logger, as plain floats.

        Two readouts the method cannot be judged without, and which nothing
        logged before this:

        - **alpha_2 / beta_2**, the coefficients on the previous token. A
          collapse to zero is `ame_experiments.md` 8.1's kill criterion -- the
          model declining KV shifting -- and it is only interpretable if you can
          see the trajectory rather than the endpoint.
        - **the horizon ladder**, `tau / m` per head. The slopes are frozen at
          `tau_init` while tau is learned, so the reach of every head moves with
          tau and the ladder does not. A flattest head that has quietly fallen
          below the training context is invisible in the loss.

        `src.core.trainer` picks this up by name off any module that defines it,
        so a stack with no AME layers logs nothing and pays nothing.
        """
        readout = {}
        for name in ("alpha", "beta"):
            # Column 1 is the coefficient on the shifted (previous-token) copy.
            shifted = _full(getattr(self, name)).detach().float()[:, 1]
            readout[f"{name}_2/mean"] = shifted.mean().item()
            readout[f"{name}_2/min"] = shifted.min().item()
            readout[f"{name}_2/max"] = shifted.max().item()
            # The collapse criterion as one number: signed means cancel across
            # heads, so a stack that has split into +0.5 and -0.5 heads would
            # read as zero. This does not.
            readout[f"{name}_2/absmean"] = shifted.abs().mean().item()

        tau = _full(self.tau).detach().float()
        readout["tau/mean"] = tau.mean().item()
        readout["tau/min"] = tau.min().item()

        horizons = self.horizons()
        readout["horizon/min"] = horizons.min().item()
        readout["horizon/max"] = horizons.max().item()

        readout.update(self._gate_readout())

        # Entropy and top-k mass need a batch; the trainer hands over the last
        # micro-batch's ids. They are the other half of 7's list, and unlike
        # everything above they describe what attention *did* rather than what
        # its parameters are. A gate that reads x gets nothing here (see
        # attention_diagnostics).
        #
        # Not under FSDP2 either. This runs outside FSDP2's forward hooks, where
        # the parameters are sharded DTensors: the diagnostics' embedding of
        # plain int64 ids into a DTensor bank raises, and the trainer then turns
        # off this layer's whole readout -- which is why no FSDP2 run ever logged
        # alpha or tau (stage_2c_report.md, "Readouts"). The readouts above
        # gather with `_full` and keep working; only entropy and top-k mass are
        # lost on a sharded run.
        sharded = any(hasattr(p, "full_tensor") for p in self.parameters())
        if token_ids is not None and not sharded:
            readout.update(self.attention_diagnostics(token_ids))
        return readout

    def _trust_logits(self, which):
        """B1's `[vocab, H]` trust logits of the q or k bank, or None when this
        rank holds only a column shard of it (mem_parallel_size > 1).

        Read from the fp32 master when the memory-parallel controller owns the
        bank -- the forward's copy is bf16 -- and from the weight otherwise.
        """
        bank = getattr(self, f"{which}_bank")
        weight = getattr(bank, "master", None)
        if weight is None:
            weight = bank.weight
        datt = self.q_heads * self.dhead
        if weight.shape[-1] != datt + self.q_heads:
            return None
        return _full(weight).detach()[:, datt:].float()

    @torch.no_grad()
    def _gate_readout(self):
        """Parameter-only readouts of the bank gate, cheap enough for every
        heavy-metric step. B2's and ST's gates depend on x, so for them this is
        the gate's weights, not its values."""
        readout = {}
        if self.bank_gate == "confidence":
            for which in ("q", "k"):
                logits = self._trust_logits(which)
                if logits is None:
                    continue
                trust = torch.sigmoid(logits).flatten()
                levels = torch.tensor([0.1, 0.5, 0.9], device=trust.device)
                p10, p50, p90 = torch.quantile(trust, levels).tolist()
                readout[f"trust_{which}/mean"] = trust.mean().item()
                readout[f"trust_{which}/p10"] = p10
                readout[f"trust_{which}/p50"] = p50
                readout[f"trust_{which}/p90"] = p90
                readout[f"trust_{which}/above_half"] = (
                    (trust > 0.5).float().mean().item()
                )
        elif self.bank_gate == "backoff":
            for which in ("q", "k"):
                norms = _full(getattr(self, f"{which}_base")).detach().float()
                norms = norms.norm(dim=-1)
                readout[f"base_{which}_norm/mean"] = norms.mean().item()
                for head, value in enumerate(norms.tolist()):
                    readout[f"base_{which}_norm/h{head}"] = value
        elif self.bank_gate in ("context", "stem"):
            weight = _full(self.gate.weight).detach().float()
            readout["gate_weight/rms"] = weight.pow(2).mean().sqrt().item()
            if self.bank_gate == "context":
                bias = _full(self.gate.bias).detach().float()
                readout["gate_bias/mean"] = bias.mean().item()
                readout["gate_bias/min"] = bias.min().item()
                readout["gate_bias/max"] = bias.max().item()
        return readout

    def _query_gate(self, x, q):
        """B2's and ST's factor on the normalized query, in q's dtype.

        B2: `sigmoid(W_g x + b)`, one scalar per position and head, computed in
        fp32. ST: `silu(W_g x)`, one value per position and channel.
        """
        batch, seq_len = x.shape[:2]
        if self.bank_gate == "context":
            gate = torch.sigmoid(self.gate(x).float())
            return gate.to(q.dtype).unsqueeze(-1)
        gate = F.silu(self.gate(x))
        return gate.view(batch, seq_len, self.q_heads, self.dhead).to(q.dtype)

    def _mixed_qkv(self, x, token_ids):
        """Everything up to the kernel: gather, normalize, mix, scale.

        Returns q, k, v as `[B, H, n, D]`, with the score scale already folded
        into q so the kernel runs at scale 1.
        """
        batch, seq_len = token_ids.shape
        heads, dhead = self.q_heads, self.dhead

        # One call when a memory-parallel controller has fused the pair. The two
        # banks are gathered on the SAME ids and their results are used
        # identically, so under column sharding they are one gather, one
        # all-to-all and one permute instead of two of each -- and those kernels
        # are launch-bound, not bandwidth-bound, so halving the count is most of
        # the cost (documents/notes/archive/ame_optimization_plan.md, D1/D2). Absent the
        # controller `qk_bank` is None and this is the original two-lookup path,
        # unchanged.
        if self.qk_bank is not None:
            q, k = self.qk_bank(token_ids)
        else:
            q = self.q_bank(token_ids)
            k = self.k_bank(token_ids)
        if self.bank_gate == "confidence":
            # B1: a row's last H columns are its trust logits, one per head.
            datt = heads * dhead
            q, q_trust = q[..., :datt], q[..., datt:]
            k, k_trust = k[..., :datt], k[..., datt:]
        q = q.view(batch, seq_len, heads, dhead)
        k = k.view(batch, seq_len, heads, dhead)
        v = self.v_proj(x).view(batch, seq_len, heads, dhead)

        if self.bank_gate == "backoff":
            # B3, Gemma 3n's per-layer-embedding mix: every row is averaged with
            # one shared, densely trained vector before the norm. The norm makes
            # the 1/sqrt(2) a no-op; it stays so this reads as Gemma's formula.
            # base is cast to the rows' dtype so the prep runs in the precision
            # every other arm's does.
            q = (q + self.q_base.to(q.dtype)) / math.sqrt(2.0)
            k = (k + self.k_base.to(k.dtype)) / math.sqrt(2.0)

        if self.qk_norm:
            # On the bank ROWS, before the mix. Not on the mixed key. Cast to the
            # compute dtype (v's): CUDA autocast runs the norm in fp32, and fp32
            # banks gather fp32 rows, so the gates, the mix and tau would run in
            # fp32 until the cast below (documents/notes/archive/review_issues.md, N7).
            q = F.normalize(q, dim=-1).to(v.dtype)
            k = F.normalize(k, dim=-1).to(v.dtype)

        if self.bank_gate == "confidence":
            # After the norm, before the mix: the KV shift mixes trust-scaled
            # keys. The sigmoid in fp32, then cast.
            q = q * torch.sigmoid(q_trust.float()).to(q.dtype).unsqueeze(-1)
            k = k * torch.sigmoid(k_trust.float()).to(k.dtype).unsqueeze(-1)
        elif self.scores_read_hidden_state:
            # B2 and ST: the query only, after the norm and before tau.
            q = q * self._query_gate(x, q)

        alpha = self.alpha.to(k.dtype)
        beta = self.beta.to(v.dtype)
        k = alpha[:, 0].view(1, 1, -1, 1) * k + alpha[:, 1].view(1, 1, -1, 1) * _shift1(
            k
        )
        v = beta[:, 0].view(1, 1, -1, 1) * v + beta[:, 1].view(1, 1, -1, 1) * _shift1(v)

        # tau folded into q: scaling one side is identical to scaling the score,
        # and it leaves the kernel's own scale at 1 for both backends.
        q = q * self.tau.to(q.dtype).view(1, 1, -1, 1)

        # One compute dtype for the kernel: q and k take `v`'s, and `v` (an
        # nn.Linear output) is the honest signal for what the surrounding
        # autocast context computes in. What q and k carry before this cast
        # depends on where the banks live:
        # - Without the memory-parallel controller the banks are nn.Embedding
        #   parameters. On one GPU or under DDP they stay fp32, and autocast
        #   casts nn.Linear but not an embedding gather, so the gather, the norm
        #   and the mix run in fp32 up to here while `v` is bf16. Flex rejects
        #   that mismatch outright and the reference path would silently run
        #   the score half in fp32. Under FSDP2 the MixedPrecisionPolicy
        #   (param_dtype bf16) casts the banks like every other parameter, so
        #   the gathered rows are already bf16.
        # - With the controller (`trainer.mem_parallel_size` set, even to 1 on a
        #   single GPU) the forward reads a copy of the banks held in the
        #   controller's compute dtype -- bf16 when CUDA is available and the
        #   stack is FSDP2 or autocast_bf16, the model's dtype otherwise (main.py,
        #   build_memory_parallel_controller) -- refreshed from an fp32 master
        #   after every optimizer step. On a bf16 stack the gathered rows are
        #   then already bf16; only the master, its gradient and the optimizer
        #   state are fp32.
        q, k = q.to(v.dtype), k.to(v.dtype)

        return (z.transpose(1, 2) for z in (q, k, v))

    def forward(self, x, token_ids):
        if token_ids is None:
            # Reachable by calling the encoder or a block directly: token_ids
            # defaults to None the whole way down so that layers which do not
            # want it need no signature change. Without this the failure is an
            # AttributeError on None.shape, four frames in, naming nothing.
            raise ValueError(
                "ame attention needs token ids: its Q/K banks are indexed by "
                "token id, not by the residual stream. The caller must thread "
                "input_ids through (see LLM.forward -> TransformerEncoder)."
            )
        batch, seq_len = token_ids.shape
        q, k, v = self._mixed_qkv(x, token_ids)
        # "auto" = flex on CUDA, reference on CPU. Not a preference: in torch
        # 2.9 a compiled flex_attention whose inputs require grad fails to lower
        # on CPU (it fails for a bare flex_attention call with no score_mod and
        # no block_mask, so it is the backend, not this layer), while the
        # reference path compiles to one graph with no breaks. On CUDA the
        # ordering reverses -- flex is the whole reason the distance bias is
        # affordable at a real context length.
        use_flex = q.is_cuda if self.backend == "auto" else self.backend == "flex"
        attend = self._attend_flex if use_flex else self._attend_reference
        out = attend(q, k, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.q_heads * self.dhead)
        return self.o_proj(out)

    # ---- kernels -------------------------------------------------------------
    #
    # Two paths computing the same function. `reference` materializes the n x n
    # scores and is the oracle the tests check `flex` against; `flex` applies the
    # distance bias through score_mod without materializing anything, which is
    # what makes the bias affordable at a real context length (and keeps the
    # slopes differentiable, unlike flash's alibi_slopes argument).

    def _with_sink(self, k, v):
        """Prepend the learned sink key/value as kv column 0."""
        if not self.use_sink:
            return k, v, 0
        batch = k.shape[0]
        k_sink = self.k_sink.to(k.dtype).view(1, self.q_heads, 1, self.dhead)
        v_sink = self.v_sink.to(v.dtype).view(1, self.q_heads, 1, self.dhead)
        k = torch.cat([k_sink.expand(batch, -1, -1, -1), k], dim=2)
        v = torch.cat([v_sink.expand(batch, -1, -1, -1), v], dim=2)
        return k, v, 1

    def score_bias(self, seq_len, device, dtype, offset):
        """Additive `[H, n, n + offset]` bias: `-m_h * (i - j)` plus the causal
        mask, with the sink column (when present) unbiased and always visible."""
        i = torch.arange(seq_len, device=device).view(1, -1, 1)
        j = torch.arange(seq_len + offset, device=device).view(1, 1, -1) - offset
        slopes = self.slopes.to(device=device, dtype=torch.float32).view(-1, 1, 1)
        bias = -slopes * (i - j).float()
        allowed = j <= i
        if offset:
            # kv column 0 is the sink: no distance, visible to every query.
            bias[:, :, :offset] = 0.0
            allowed = allowed.clone()
            allowed[:, :, :offset] = True
        return bias.masked_fill(~allowed, float("-inf")).to(dtype)

    def _attend_reference(self, q, k, v, return_scores=False):
        k, v, offset = self._with_sink(k, v)
        scores = torch.einsum("bhid,bhjd->bhij", q, k).float()
        scores = scores + self.score_bias(
            q.shape[2], q.device, torch.float32, offset
        ).unsqueeze(0)
        weights = scores.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", weights.to(v.dtype), v)
        return (out, scores) if return_scores else out

    def _block_mask(self, seq_len, offset, device):
        """Build (and cache) the causal-plus-sink block mask.

        Left *inside* the compiled graph, and that was measured rather than
        assumed. `@torch._dynamo.disable` here looks attractive -- it drops the
        layer from 3 graph breaks to 2, because tracing `create_block_mask`
        costs three `call_id`-on-a-nested-function breaks -- but it makes the
        BlockMask a graph *input*, and its sequence length then arrives as a
        SymInt once dynamo marks that dimension dynamic. Inductor cannot lower
        `flex_attention` with it (`args[8]: (s4,)`) and two GPU tests go red.
        2.1 ms against 2.0 ms is not worth a kernel that will not lower.

        If you try this again, test it with `pytest -m gpu` and not with a
        script: compiling the layer at a couple of sequence lengths in a fresh
        process does *not* reproduce the failure, because the dynamic promotion
        comes from the suite compiling many layers across widths in one
        process (`archive/ame_review_and_fixes.md` 1.2, jobs 32368 and 32372).

        The mask is cached per (seq_len, offset, device), so after the first
        call this is a dict lookup.
        """
        from torch.nn.attention.flex_attention import create_block_mask

        key = (seq_len, offset, str(device))
        cached = self._block_mask_cache.get(key)
        if cached is not None:
            return cached

        def mask_mod(b, h, q_idx, kv_idx):
            # Causal, shifted by the sink column; the sink is visible to all.
            return (kv_idx < offset) | (kv_idx - offset <= q_idx)

        mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len + offset,
            device=device,
        )
        self._block_mask_cache[key] = mask
        return mask

    def _attend_flex(self, q, k, v):
        """Flex over the sequence, with the sink merged in afterwards.

        **The sink used to be a kv column and it cost 876 ms per step.** Measured
        at the h20 shape (`scripts/bench_attention.py`, one H100, 20 layers x 8
        micro-steps):

            flex, ALiBi + sink as kv column 0    1331.6 ms/step
            flex, ALiBi, no sink                  455.8 ms/step
            flex, no score_mod at all             397.5 ms/step
            SDPA flash, causal, no bias           447.5 ms/step

        So the distance bias costs 58 ms and flex with it already matches the
        flash floor; prepending one key to make room for the sink costs **more
        than the whole rest of attention**. Two reasons, both structural rather
        than arithmetic: it makes `KV_LEN` 2049, which is one past flex's
        128-wide block alignment and turns every block ragged, and it puts a
        `torch.where` inside `score_mod`, which flex evaluates once per score --
        n^2 per head, to special-case a single column.

        Neither is necessary. Attention over `{sink} u sequence` is exactly the
        softmax-weighted merge of attention over each part, and flex will return
        the log-sum-exp needed to do it:

            numerator   = out_seq * exp(lse_seq) + exp(q . k_sink) * v_sink
            denominator = exp(lse_seq)           + exp(q . k_sink)

        shifted by the running max for stability. That is O(B*H*n*D) elementwise
        work -- no n^2 term -- against 876 ms of kernel. The sink keeps its exact
        semantics: no distance bias, visible to every query, `tau` already folded
        into `q` by `_mixed_qkv`.
        """
        if not self.use_sink:
            return self._attend_flex_sequence(q, k, v)

        out_seq, lse_seq = self._attend_flex_sequence(q, k, v, return_lse=True)

        # The sink's logit: one score per query, no bias, scale 1 (tau is in q).
        #
        # Written as a broadcast multiply and a reduction over D rather than as
        # `einsum("bhnd,hd->bhn", q, k_sink)`, which is the same value for 2x
        # the time. The einsum is a matrix-VECTOR product per head -- it reduces
        # 64 elements down to one scalar per (b, h, n) -- which is the shape a
        # tensor core is worst at, and it blocks the surrounding fusion because
        # it lands in a GEMM library kernel. Measured per optimizer step at the
        # h20 shape (`scripts/bench_attention.py`, job 32776, merge in
        # isolation with out_seq/lse_seq as leaves):
        #
        #     einsum,  eager      88.3 fwd + 131.5 bwd = 219.9 ms
        #     mul+sum, eager     121.6 fwd + 159.7 bwd = 281.3 ms
        #     einsum,  compiled   41.8 fwd +  61.8 bwd = 103.6 ms
        #     mul+sum, compiled   14.2 fwd +  36.7 bwd =  50.9 ms
        #
        # Note the inversion: eager, the einsum WINS, because a reduction over
        # the last axis of a [B, H, n, D] tensor really does cost more than a
        # library GEMM when nothing fuses it. Compiled -- which is how this
        # actually runs, inside `compile_blocks_full` -- inductor folds the
        # multiply, the reduction, the sigmoid and the final FMA into the
        # neighbouring pointwise kernels, and the einsum cannot be folded into
        # anything. **Optimising this eagerly would have picked the wrong form.**
        #
        # fp32 for the product, not bf16: the summands are a bf16 q against a
        # bf16 k_sink, and accumulating 64 of them in bf16 would be strictly
        # worse than the fp32 accumulation cuBLAS gave the einsum for free.
        # inductor fuses the upcast into the reduction, so no [B, H, n, D] fp32
        # temporary is ever materialised.
        k_sink = self.k_sink.to(q.dtype).view(1, self.q_heads, 1, self.dhead)
        sink_score = (q.to(torch.float32) * k_sink.to(torch.float32)).sum(-1)

        # The merge, written as a LERP rather than as the ratio it comes from.
        #
        #     out = (out_seq * e_seq + v_sink * e_sink) / (e_seq + e_sink)
        #         = out_seq + alpha * (v_sink - out_seq),  alpha = sigmoid(s - lse)
        #
        # Identical arithmetic, very different cost. The ratio form needs the
        # full `[B, H, n, D]` output in fp32 and builds five tensors that size;
        # measured at the h20 shape that was 156 ms forward and 403 ms backward
        # per step, which ate most of what taking the sink out of the kernel had
        # just saved. This form touches `[B, H, n]` for the weight -- 1/64th the
        # elements -- and leaves the one big tensor in its own dtype for a
        # single fused multiply-add.
        #
        # `sigmoid` is computed in fp32 because `s - lse` is an unbounded logit
        # difference; `alpha` itself lives in [0, 1] and is cast down, which is
        # the precision the surrounding bf16 attention already works in.
        # `sink_score` is already fp32 from the reduction above; only `lse_seq`
        # needs the cast, and flex returns it fp32 anyway on every path we use.
        alpha = torch.sigmoid(sink_score - lse_seq.to(torch.float32)).to(out_seq.dtype)
        v_sink = self.v_sink.to(out_seq.dtype).view(1, self.q_heads, 1, self.dhead)
        # Written as the fused multiply-add rather than torch.lerp: lerp wants
        # its three arguments the same shape, so `v_sink` would be expanded to
        # the full [B, H, n, D] first. This broadcasts instead, and inside
        # `compile_blocks_full` inductor fuses the whole tail -- sigmoid, sub,
        # mul, add -- into the kernel that produced `out_seq`.
        return out_seq + alpha.unsqueeze(-1) * (v_sink - out_seq)

    def _attend_flex_sequence(self, q, k, v, return_lse=False):
        """flex over the sequence alone: plain causal mask, ALiBi score_mod."""
        offset = 0
        slopes = self.slopes.to(device=q.device, dtype=torch.float32)

        def score_mod(score, b, h, q_idx, kv_idx):
            # The bias in fp32, always. In bf16 a distance near 2048 quantizes
            # to multiples of 16, which for a steep head (m ~ 2^-1) is several
            # logits of error on exactly the near-diagonal entries that decide
            # the softmax -- while tau is only ~11. The final logit is cast
            # back, where bf16's resolution is the same one standard attention
            # already lives with.
            distance = (q_idx - kv_idx).to(torch.float32)
            return (score.to(torch.float32) - slopes[h] * distance).to(score.dtype)

        # The block mask is built outside the kernel call and cached, so the
        # compiled path never traces create_block_mask. offset is 0 here, so
        # KV_LEN == Q_LEN == seq_len -- square, but square is not the same as
        # aligned. flex blocks are 128 wide, and a seq_len that is not a
        # multiple of 128 leaves a ragged last row-block that costs far more
        # than its size: measured at B=16 H=20 D=64 (job 32688), seq_len 2047
        # takes 1156.9 ms per optimizer step against 677.0 at 2048. The caller
        # owns that -- `_preprocess_input` shifts for next-token targets, so a
        # batch 2049 wide is what puts seq_len at 2048.
        block_mask = self._block_mask(q.shape[2], offset, q.device)
        return _flex_attention(
            q,
            k,
            v,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=1.0,
            return_lse=return_lse,
        )

    # ---- instrumentation -----------------------------------------------------

    @torch.no_grad()
    def score_matrix(self, token_ids):
        """The `[B, H, n, n (+1)]` score matrix, for diagnostics: attention
        entropy, top-k mass, whether a sink row emerged. Reference path only --
        it materializes the scores on purpose.

        It feeds `x = 0`, which is exact only while the scores ignore x. Under a
        gate that reads x it would be silently wrong -- B2's gate would read
        sigmoid(b) everywhere and ST's silu(0) = 0 would erase every query -- so
        those gates raise instead.
        """
        if self.scores_read_hidden_state:
            raise ValueError(
                f"score_matrix runs on x = 0, but bank_gate={self.bank_gate!r} "
                "reads the hidden state, so its scores need real activations "
                "(documents/scripts/bank_gates_eval.py computes them)"
            )
        x = torch.zeros(
            *token_ids.shape,
            self.dmodel,
            device=token_ids.device,
            dtype=self.v_proj.weight.dtype,
        )
        q, k, v = self._mixed_qkv(x, token_ids)
        _, scores = self._attend_reference(q, k, v, return_scores=True)
        return scores

    @torch.no_grad()
    def attention_diagnostics(self, token_ids, top_k=8, max_rows=128, max_batch=1):
        """Attention entropy and top-k mass, from a sample of query rows.

        `ame_experiments.md` 7 asks for both from the first run. They are
        functions of the attention *weights*, which the fused kernel never
        forms, so this recomputes the scores for a sample of rows rather than
        asking the kernel for something it does not keep.

        It is cheap because AME's scores depend on ids and distance only: no
        activations are needed, and `max_rows` rows cost `[1, H, max_rows, n]`
        instead of the full `[B, H, n, n]` -- about 21 MB rather than 336 MB at
        h20, once per heavy-metric interval.

        Entropy is reported twice. The raw mean is in nats; it is dominated by
        position, because query row `i` can only see `i + 1` keys and an early
        row is near-zero whatever the head is doing. The **normalized** figure
        divides by `ln(visible keys)` per row, so 1.0 is uniform attention over
        what the row can reach and 0.0 is a head collapsed onto one key --
        comparable across rows, heads and context lengths, and the one to read.

        Empty for a gate that reads x (B2, ST): computed from `x = 0` their
        readout would be wrong without saying so. Their attention statistics
        come from the offline evaluation, which has real activations.
        """
        if self.scores_read_hidden_state:
            return {}
        ids = token_ids[:max_batch]
        batch, seq_len = ids.shape
        x = torch.zeros(
            batch,
            seq_len,
            self.dmodel,
            device=ids.device,
            dtype=self.v_proj.weight.dtype,
        )
        q, k, _ = self._mixed_qkv(x, ids)
        k, _, offset = self._with_sink(k, k)

        rows = _diagnostic_rows(seq_len, max_rows, ids.device)

        scores = torch.einsum("bhid,bhjd->bhij", q[:, :, rows, :], k).float()
        scores = scores + self.score_bias(seq_len, ids.device, torch.float32, offset)[
            :, rows, :
        ].unsqueeze(0)
        weights = scores.softmax(dim=-1)

        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
        # Keys row i can actually see: itself, everything before it, and the
        # sink column when there is one.
        visible = (rows + 1 + offset).to(entropy.dtype)
        ceiling = visible.log().clamp_min(1e-6).view(1, 1, -1)
        k_eff = min(top_k, weights.shape[-1])

        return {
            "attention_entropy/nats": entropy.mean().item(),
            "attention_entropy/normalized": (entropy / ceiling).mean().item(),
            f"top{k_eff}_mass/mean": weights.topk(k_eff, dim=-1)
            .values.sum(-1)
            .mean()
            .item(),
        }

    def attention_score_flops_per_token(self, seq_len):
        """QK^T and AV, the PaLM convention: 2*s*datt each per token forward,
        tripled for the backward pass. Identical to standard attention -- AME's
        saving is the deleted W_Q/W_K matmuls, not the score path. The sink is
        one more key column, so it adds one to the key count."""
        keys = seq_len + (1 if self.use_sink else 0)
        return 12 * keys * self.q_heads * self.dhead

    def extra_repr(self):
        return (
            f"vocab_size={self.vocab_size}, q_heads={self.q_heads}, dhead={self.dhead}, "
            f"qk_norm={self.qk_norm}, sink={self.use_sink}, "
            f"alpha_pinned={self.alpha_is_pinned}, beta_pinned={self.beta_is_pinned}, "
            f"bank_gate={self.bank_gate}"
            + (
                f", confidence_init={self.confidence_init}"
                if self.bank_gate == "confidence"
                else ""
            )
        )
