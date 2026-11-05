"""Hand-managed, column-sharded Q/K banks -- Tiers 2 and 3 of the AME plan.

AME replaces `W_Q`/`W_K` with two `nn.Embedding` banks indexed by token id. At
h20 that is 2.576B of the model's 3.163B parameters, and FSDP2 treats them like
any other weight: it all-gathers every bank on the way into each micro-step's
forward and again on the way into its backward, materialises a dense gradient
for each, reduce-scatters it, and hands the whole thing to AdamW. The forward
only ever reads `micro_batch * seq_len` of the 50304 rows, so almost all of that
traffic is for rows nothing touched.

This module takes the banks off FSDP2 entirely and runs them by hand:

  O1  `fully_shard(..., ignored_params=...)` keeps FSDP2 out. Each bank keeps a
      compute parameter in the autocast dtype, a private fp32 master, and a
      persistent fp32 `main_grad` the backward scatters into with `index_add_`.
      No dense per-micro-step gradient is ever allocated, and `main_grad` is not
      zeroed between micro-steps, so accumulation is free.
  O2  One `all_reduce(SUM)` of `main_grad` per optimizer step, then
      `div_(world_size)` -- FSDP2 reduces gradients as a MEAN over ranks, and
      the banks must agree with the backbone about what a gradient means.
  O3  fp32 master -> compute parameter copy after each step.
  O4  The clip. See `local_grad_norm_sq`: the norm is computed JOINTLY with the
      backbone's, not separately, because a separate clip is a different
      function and the section 1.4 gate compares against a baseline that clips
      once over everything.
  O5  Three hook points, called from `src.core.trainer.Trainer.train`:
      `materialize()` at step begin, `reduce()` after the accumulation loop, and
      `step()` beside the backbone's optimizer step.
  O6  Two process groups. The *memory-parallel* group is a contiguous block of
      `mem_parallel_size` ranks that between them own one whole bank, each rank
      holding a column slice `[vocab, width / mp]`. The *memory-DP* subgroup is
      strided with size `world / mp` and holds the ranks that own the SAME
      slice; `main_grad` is all-reduced only there.
  O7  The forward all-gathers token ids inside the mp-group, gathers every
      rank's tokens from the locally owned columns, and all-to-alls the result
      so each rank ends up with its own tokens at full width. The backward
      mirrors it. Attention downstream sees exactly the tensor it saw before.
  O8  **The divisor is the GLOBAL world size, not the size of the group the
      all-reduce ran in.** The id all-gather in the forward already summed `mp`
      ranks' token contributions into each slice's local gradient, and the
      all-reduce then sums the remaining `world / mp`, so the terms total
      `mp * (world / mp) = world`. Getting this wrong scales every bank gradient
      by a constant and is invisible in the loss for hundreds of steps -- which
      is why `tests/test_memory_parallel.py` compares `mp=1` against `mp=4`
      directly.

A column slice is a HEAD subset (`width = q_heads * d_head`, and heads are
independent until `o_proj`), so this split is natural for attention even though
the all-to-all restores full width before the layer does anything with it.

`mem_parallel_size=1` is Tier 2 on its own: no collectives in the bank forward,
every rank holding a whole bank. It is mathematically identical to `mp>1` and
exists as the correctness reference. At h20 it does not FIT -- an unsharded bank
costs 46.4 GB/rank of state against 11.6 GB sharded -- so on real hardware the
useful settings are `mp == world_size`.
"""

import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _BankGather(torch.autograd.Function):
    """ids -> `[*ids.shape, width]`, with the backward scattering into `main_grad`.

    Returns `None` as the gradient of `weight`, so `weight.grad` stays None
    forever. That is what keeps the dense per-micro-step gradient from ever
    being allocated, and it is also what makes the bank invisible to
    `Trainer.clip_gradient` (which filters on `p.grad is not None`) and to any
    optimizer that was handed the parameter by accident.
    """

    @staticmethod
    def forward(ctx, weight, ids, bank):
        mp_size = bank.mp_size
        ids_flat = ids.reshape(-1).contiguous()
        n = ids_flat.numel()

        if mp_size == 1:
            gathered_ids = ids_flat
            out = F.embedding(gathered_ids, weight)
        else:
            # O7, forward half. Every rank needs every rank's ids, because the
            # columns it owns are the only source for those columns.
            gathered_ids = torch.empty(
                mp_size * n, dtype=ids_flat.dtype, device=ids_flat.device
            )
            dist.all_gather_into_tensor(gathered_ids, ids_flat, group=bank.mp_group)
            local = F.embedding(gathered_ids, weight)  # [mp*n, width/mp]
            shuffled = torch.empty_like(local)
            # Chunk i of `local` is rank i's tokens on MY columns and goes to
            # rank i; chunk j of the result is MY tokens on rank j's columns.
            dist.all_to_all_single(shuffled, local.contiguous(), group=bank.mp_group)
            # Rank j owns columns [j*w, (j+1)*w), so concatenating the received
            # chunks in rank order restores the original column order exactly.
            out = shuffled.view(mp_size, n, -1).permute(1, 0, 2).reshape(n, -1)

        ctx.bank = bank
        ctx.n = n
        ctx.save_for_backward(gathered_ids)
        return out.view(*ids.shape, out.shape[-1])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        bank = ctx.bank
        (gathered_ids,) = ctx.saved_tensors
        mp_size, n = bank.mp_size, ctx.n
        grad = grad_out.reshape(n, -1)

        if mp_size == 1:
            local = grad
        else:
            # O7, backward half: the forward's all-to-all run with the splits
            # swapped. Slice j of the width belongs to rank j, so it is sent
            # there; what comes back is every rank's gradient for MY columns,
            # row-aligned with `gathered_ids`.
            width = grad.shape[1] // mp_size
            send = (
                grad.view(n, mp_size, width)
                .permute(1, 0, 2)
                .reshape(mp_size * n, width)
                .contiguous()
            )
            local = torch.empty_like(send)
            dist.all_to_all_single(local, send, group=bank.mp_group)

        bank.accumulate(gathered_ids, local)
        return None, None, None


class ShardedBank(nn.Module):
    """A column slice of one Q/K bank, with its own fp32 master and gradient.

    Drop-in for the `nn.Embedding` it replaces: `bank(ids)` still returns
    `[batch, seq, width]` at full width, so `AMEAttention._mixed_qkv` -- the
    measured hot path -- is untouched.
    """

    # src.core.mfu charges 6*P for every 2-D parameter unless the module says
    # otherwise, and identifies lookups by `isinstance(module, nn.Embedding)`.
    # This is not one, so it must opt out by name or the MFU denominator would
    # gain 6 * 2.576B FLOPs/token the banks never spend -- and every number in
    # the plan's ledger would stop being comparable to the ones above it.
    non_matmul_parameters = ("weight",)

    def __init__(self, vocab_size, width, controller):
        super().__init__()
        mp_size = controller.mp_size
        if width % mp_size:
            raise ValueError(
                f"bank width {width} is not divisible by mem_parallel_size {mp_size}"
            )
        self.vocab_size = vocab_size
        self.width = width
        self.mp_size = mp_size
        # The master and the gradient accumulator are "fp32" in the sense that
        # matters -- higher precision than the compute parameter, and at least
        # fp32. Derived rather than hardcoded so a float64 gate really runs in
        # float64: an fp32 master under an fp64 model loses ~1e-7 per step,
        # which is six orders of magnitude above the noise the section 1.4 gate
        # is trying to distinguish a real bug from.
        self.master_dtype = (
            controller.compute_dtype
            if controller.compute_dtype.itemsize > 4
            else torch.float32
        )
        self.mp_group = controller.mp_group
        self.slice_width = width // mp_size
        self.column_offset = controller.mp_rank * self.slice_width
        self.weight = nn.Parameter(
            torch.empty(vocab_size, self.slice_width, dtype=controller.compute_dtype)
        )
        # Plain attributes, NOT buffers: FSDP2's MixedPrecisionPolicy casts a
        # module's buffers to the compute dtype, which would quietly turn the
        # fp32 master and the fp32 gradient accumulator into bf16 and make the
        # whole exercise pointless.
        self.master = None
        self.main_grad = None

    @torch.no_grad()
    def load_from(self, embedding):
        """Take this rank's columns out of the full, already-initialised bank.

        Every rank builds the identical model under one seed before this runs,
        so slicing by `column_offset` gives a consistent global parameter with
        no communication -- and `mp=1` and `mp=4` start from the same numbers,
        which is what makes their loss trajectories comparable.
        """
        full = embedding.weight.detach()
        columns = full[:, self.column_offset : self.column_offset + self.slice_width]
        self.weight.copy_(columns.to(self.weight.dtype))
        self.master = columns.to(self.master_dtype).clone()

    def materialize(self, device):
        """O5 hook 1: make sure the master and the accumulator sit on the
        parameter's device, allocating the accumulator on first use."""
        if self.master is None:
            self.master = self.weight.detach().to(self.master_dtype).clone()
        if self.master.device != device:
            self.master = self.master.to(device)
        if self.main_grad is None:
            self.main_grad = torch.zeros_like(self.master)
        elif self.main_grad.device != device:
            self.main_grad = self.main_grad.to(device)

    def accumulate(self, ids, grad):
        """Scatter one micro-step's gradient into the persistent fp32 buffer.

        `index_add_` rather than a dense `embedding_dense_backward` + add: the
        dense form allocates `[vocab, slice_width]` per micro-step per bank and
        then adds it to a buffer of the same shape, which is the allocation this
        tier exists to delete.
        """
        self.main_grad.index_add_(0, ids, grad.to(self.main_grad.dtype))

    # Eager, always. Dynamo cannot trace this: it is a custom autograd.Function
    # that stashes a live module on `ctx` and calls three process-group
    # collectives, and a silent partial trace here would be a wrong gradient
    # rather than a slow one. The cost is a graph break at the top of each
    # attention layer, which leaves the expensive part -- the mix, the kernel
    # and the feed-forward -- inside one compiled region. If compile_blocks_full
    # is ever measured as worse with this in it, that is the number to compare,
    # not a reason to let dynamo in.
    @torch.compiler.disable
    def forward(self, ids):
        return _BankGather.apply(self.weight, ids, self)

    # ---- checkpointing -------------------------------------------------
    #
    # The bank is deliberately absent from the module's state_dict. At
    # mem_parallel_size > 1 every rank's `weight` has the SAME fully-qualified
    # name and a DIFFERENT quarter of the columns, and it is a plain tensor, not
    # a DTensor -- so a distributed save would write one rank's slice under that
    # name and silently drop the rest. The controller writes a rank-local
    # sidecar instead (`MemoryParallelController.save`), which also carries the
    # fp32 master and the bank optimizer's moments; the compute parameter is
    # derived from the master on load and is not worth storing twice.

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        return

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Loud, not silent: a checkpoint written WITHOUT memory-parallel banks
        # carries `<prefix>weight`, and loading it here would otherwise look
        # like a success while leaving the banks at their initial values.
        for key in list(state_dict):
            if key.startswith(prefix):
                unexpected_keys.append(key)

    def extra_repr(self):
        return (
            f"{self.vocab_size}, {self.width}, mem_parallel={self.mp_size}, "
            f"columns=[{self.column_offset}, {self.column_offset + self.slice_width})"
        )


class FusedBankPair(nn.Module):
    """One gather, one all-to-all and one permute for a layer's q and k banks.

    `AMEAttention` looks both banks up on the SAME token ids, back to back, and
    uses the two results identically. Under column sharding each lookup costs a
    local gather, an all-to-all and a transposing copy, and all three are
    **launch-bound** -- the gather measured 90.8 ms for 168 MB of traffic, 5.6x
    its bandwidth bound, over 328 launches. Doing the pair in one call halves
    every one of those counts while moving exactly the same bytes.

    It owns no parameters. The two `ShardedBank` modules remain the owners of
    the weights, masters, gradients and optimizer state; this only batches the
    data path, so nothing about sharding, reduction or checkpointing changes.
    """

    def __init__(self, q_bank, k_bank):
        super().__init__()
        if q_bank.mp_size != k_bank.mp_size or q_bank.width != k_bank.width:
            raise ValueError("a fused pair needs two identically shaped banks")
        # Plain attributes, not submodules: the banks are already registered on
        # the attention layer under their own names, and registering them twice
        # would duplicate them in `model.parameters()`.
        object.__setattr__(self, "q", q_bank)
        object.__setattr__(self, "k", k_bank)

    def forward(self, ids):
        return _FusedBankGather.apply(self.q.weight, self.k.weight, ids, self)


class _FusedBankGather(torch.autograd.Function):
    """`_BankGather` for two banks at once. The two weights are concatenated
    along the column axis, so one collective carries both."""

    @staticmethod
    def forward(ctx, q_weight, k_weight, ids, pair):
        bank = pair.q
        mp_size = bank.mp_size
        ids_flat = ids.reshape(-1).contiguous()
        n = ids_flat.numel()
        width = q_weight.shape[1]

        if mp_size == 1:
            gathered_ids = ids_flat
            q = F.embedding(gathered_ids, q_weight)
            k = F.embedding(gathered_ids, k_weight)
            out_q, out_k = q, k
        else:
            gathered_ids = torch.empty(
                mp_size * n, dtype=ids_flat.dtype, device=ids_flat.device
            )
            dist.all_gather_into_tensor(gathered_ids, ids_flat, group=bank.mp_group)
            # One gather of the pair: [mp*n, 2*width/mp], q columns then k.
            local = torch.cat(
                (
                    F.embedding(gathered_ids, q_weight),
                    F.embedding(gathered_ids, k_weight),
                ),
                dim=1,
            )
            shuffled = torch.empty_like(local)
            dist.all_to_all_single(shuffled, local.contiguous(), group=bank.mp_group)
            # One permute restores full width for both halves at once.
            full = shuffled.view(mp_size, n, 2, width).permute(1, 2, 0, 3)
            full = full.reshape(n, 2, mp_size * width)
            out_q, out_k = full[:, 0], full[:, 1]

        ctx.pair = pair
        ctx.n = n
        ctx.width = width
        ctx.save_for_backward(gathered_ids)
        tail = out_q.shape[-1]
        return out_q.reshape(*ids.shape, tail), out_k.reshape(*ids.shape, tail)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_q, grad_k):
        pair = ctx.pair
        (gathered_ids,) = ctx.saved_tensors
        mp_size, n, width = pair.q.mp_size, ctx.n, ctx.width

        if mp_size == 1:
            pair.q.accumulate(gathered_ids, grad_q.reshape(n, -1))
            pair.k.accumulate(gathered_ids, grad_k.reshape(n, -1))
            return None, None, None, None

        # Mirror of the forward: stack the pair, one all-to-all, then split.
        stacked = torch.stack(
            (grad_q.reshape(n, mp_size, width), grad_k.reshape(n, mp_size, width)),
            dim=1,
        )  # [n, 2, mp, width]
        send = stacked.permute(2, 0, 1, 3).reshape(mp_size * n, 2 * width).contiguous()
        local = torch.empty_like(send)
        dist.all_to_all_single(local, send, group=pair.q.mp_group)
        pair.q.accumulate(gathered_ids, local[:, :width])
        pair.k.accumulate(gathered_ids, local[:, width:])
        return None, None, None, None


def _find_banks(model):
    """Every (module, attribute) pair holding an id-indexed bank.

    Duck-typed on the attribute names rather than on `isinstance(AMEAttention)`
    so a variant layer that keeps the same contract is picked up too, and so
    this module does not import the attention layer it rewrites.
    """
    found = []
    for module in model.modules():
        for name in ("q_bank", "k_bank"):
            bank = getattr(module, name, None)
            if isinstance(bank, nn.Embedding):
                found.append((module, name, bank))
    return found


class MemoryParallelController:
    """Owns every Q/K bank: its sharding, its gradient, and its optimizer.

    Built in `main.py` *before* `fully_shard`, so FSDP2 sees `ShardedBank`
    parameters it has been told to ignore rather than `nn.Embedding` weights it
    would shard a second, incompatible way.
    """

    def __init__(
        self, model, mem_parallel_size, compute_dtype=torch.bfloat16, fuse_qk=True
    ):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.compute_dtype = compute_dtype

        mp_size = int(mem_parallel_size)
        if mp_size < 1:
            raise ValueError(f"mem_parallel_size must be >= 1, got {mp_size}")
        if self.world_size % mp_size:
            raise ValueError(
                f"mem_parallel_size {mp_size} does not divide world_size "
                f"{self.world_size}"
            )
        self.mp_size = mp_size
        self.mp_rank = self.rank % mp_size
        self.dp_size = self.world_size // mp_size
        self.mp_group, self.dp_group = self._build_groups()

        self.banks = []
        for module, name, embedding in _find_banks(model):
            bank = ShardedBank(embedding.num_embeddings, embedding.embedding_dim, self)
            bank.to(embedding.weight.device)
            bank.load_from(embedding)
            setattr(module, name, bank)
            self.banks.append(bank)
        if not self.banks:
            raise ValueError(
                "mem_parallel_size was set but the model has no q_bank/k_bank "
                "embeddings -- this controller only knows how to run AME banks"
            )

        # D2: batch each layer's q and k into one gather/all-to-all/permute.
        # Purely a data-path change -- the ShardedBank modules above stay the
        # owners of every parameter, gradient and optimizer slot.
        self.fused_pairs = []
        if fuse_qk:
            for module in model.modules():
                q = getattr(module, "q_bank", None)
                k = getattr(module, "k_bank", None)
                if isinstance(q, ShardedBank) and isinstance(k, ShardedBank):
                    if not hasattr(module, "qk_bank"):
                        raise ValueError(
                            f"{type(module).__name__} holds sharded banks but has "
                            "no `qk_bank` hook; fuse_qk=False runs the unfused path"
                        )
                    pair = FusedBankPair(q, k)
                    module.qk_bank = pair
                    self.fused_pairs.append(pair)

        self.optimizer = None
        total = sum(b.weight.numel() for b in self.banks)
        logger.info(
            f"[memory-parallel] {len(self.banks)} banks "
            f"({len(self.fused_pairs)} fused q/k pairs), "
            f"mem_parallel_size={mp_size}, "
            f"mp_rank={self.mp_rank}, dp_size={self.dp_size}, "
            f"{total / 1e9:.3f}B bank parameters per rank"
        )
        model._memory_parallel = self

    # ---- O6: the two process groups ----------------------------------------

    def _build_groups(self):
        """Contiguous mp-groups, strided dp-groups. Both are collective calls:
        every rank walks the whole loop, keeping only the group it is in."""
        if not dist.is_initialized() or self.world_size == 1:
            return None, None

        mp_group = None
        if self.mp_size > 1:
            for start in range(0, self.world_size, self.mp_size):
                ranks = list(range(start, start + self.mp_size))
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    mp_group = group

        dp_group = None
        if 1 < self.dp_size < self.world_size:
            for offset in range(self.mp_size):
                ranks = list(range(offset, self.world_size, self.mp_size))
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    dp_group = group
        # dp_size == world_size (mp == 1) leaves dp_group None, which means the
        # default group -- every rank holds the same slice and all of them must
        # take part. dp_size == 1 means nothing to reduce; `reduce()` checks.
        return mp_group, dp_group

    # ---- O1: what FSDP2 and the backbone optimizer must not touch -----------

    def ignored_parameters(self):
        return {bank.weight for bank in self.banks}

    def build_optimizer(self, lr, weight_decay=0.0, **kwargs):
        """A dedicated AdamW over the fp32 masters.

        `weight_decay` defaults to 0.0 because `AMEAttention` already names both
        banks in `no_weight_decay_parameters` -- the backbone optimizer gives
        them a wd=0 group, and this must reproduce that, not quietly reinstate
        decay on 2.576B parameters.
        """
        for bank in self.banks:
            bank.materialize(bank.weight.device)
        masters = [bank.master for bank in self.banks]
        self.optimizer = torch.optim.AdamW(
            masters, lr=lr, weight_decay=weight_decay, **kwargs
        )
        return self.optimizer

    # ---- O5: the three hook points -----------------------------------------

    def materialize(self):
        for bank in self.banks:
            bank.materialize(bank.weight.device)

    @torch.no_grad()
    def reduce(self):
        """O2 + O8: one all-reduce per optimizer step, divided by WORLD size."""
        if self.dp_size > 1:
            grads = [bank.main_grad for bank in self.banks]
            for grad in grads:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.dp_group)
        if self.world_size > 1:
            # O8. Not dp_size: the forward's id all-gather already summed the
            # mp-group's contributions into this slice.
            for bank in self.banks:
                bank.main_grad.div_(self.world_size)

    @torch.no_grad()
    def local_grad_norm_sq(self, device=None):
        """O4: this rank's contribution to the JOINT (backbone + bank) norm.

        Only the first mp-block of ranks contributes. Every slice is replicated
        across its dp-subgroup after `reduce()`, so letting all of them report
        would count each slice `dp_size` times and inflate the global norm by
        `sqrt(dp_size)` -- which is exactly the kind of silent constant factor
        the grad-norm trace exists to catch.
        """
        reference = self.banks[0].main_grad
        device = device or reference.device
        total = torch.zeros((), dtype=torch.float32, device=device)
        if self.rank < self.mp_size:
            for bank in self.banks:
                # vector_norm, not `.float().pow(2).sum()`: the latter
                # materialises a full-size fp32 copy of every bank gradient --
                # 10.3 GB at mem_parallel_size 1, 2.6 GB at 4 -- to compute one
                # scalar. The reduction reads the buffer and allocates nothing.
                norm = torch.linalg.vector_norm(bank.main_grad, 2.0)
                total = total + norm.float().to(device) ** 2
        return total

    @torch.no_grad()
    def clip_(self, max_norm, total_norm):
        """Apply the same coefficient `torch.nn.utils.clip_grads_with_norm_`
        applies to the backbone, computed from the same joint `total_norm`."""
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for bank in self.banks:
            bank.main_grad.mul_(clip_coef_clamped.to(bank.main_grad.device))

    @torch.no_grad()
    def step(self, lr):
        """O3: step the masters, then refresh the compute parameters."""
        if self.optimizer is None:
            raise RuntimeError("build_optimizer() was never called")
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        for bank in self.banks:
            bank.master.grad = bank.main_grad
        self.optimizer.step()
        for bank in self.banks:
            bank.master.grad = None
            bank.weight.copy_(bank.master.to(bank.weight.dtype))

    @torch.no_grad()
    def zero_grad(self):
        for bank in self.banks:
            bank.main_grad.zero_()

    # ---- bookkeeping --------------------------------------------------------

    def log_metrics(self, metric_logger, lr):
        """O5 hook 3: the bank LR is not the backbone's by construction (it is
        copied each step), so it is logged as its own series rather than assumed."""
        metric_logger.log("train/bank_lr", lr)

    SIDECAR = "memory_parallel_rank{rank}.pt"

    def state_dict(self):
        return {
            "masters": [bank.master for bank in self.banks],
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "mem_parallel_size": self.mp_size,
            "mp_rank": self.mp_rank,
            "world_size": self.world_size,
        }

    def save(self, folder):
        """One file per rank beside the model checkpoint.

        Rank-local rather than gathered, for the reason pkmf reached the same
        way round (O30): gathering 2.576B parameters to rank 0 is minutes of
        wall time and a 10 GB write, and the only thing it buys is a layout that
        this loader would have to re-split anyway.
        """
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, self.SIDECAR.format(rank=self.rank))
        torch.save(self.state_dict(), path)
        logger.info(f"[memory-parallel] wrote bank state to {path}")

    def load(self, folder):
        path = os.path.join(folder, self.SIDECAR.format(rank=self.rank))
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no memory-parallel bank state at {path}. The model has a bank "
                "controller, so the checkpoint must carry one sidecar per rank; "
                "a checkpoint written without mem_parallel_size cannot be "
                "resumed into one."
            )
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        logger.info(f"[memory-parallel] loaded bank state from {path}")

    @torch.no_grad()
    def load_state_dict(self, state):
        if state.get("world_size", self.world_size) != self.world_size:
            raise ValueError(
                "bank checkpoint was written at world_size "
                f"{state['world_size']}, this run uses {self.world_size}; the "
                "sidecars are per rank and do not resplit"
            )
        if state["mem_parallel_size"] != self.mp_size:
            raise ValueError(
                "bank checkpoint was written at mem_parallel_size "
                f"{state['mem_parallel_size']}, this run uses {self.mp_size}; "
                "resharding bank state across mem_parallel_size is not implemented"
            )
        for bank, master in zip(self.banks, state["masters"]):
            bank.materialize(bank.weight.device)
            bank.master.copy_(master.to(bank.master.device))
            bank.weight.copy_(bank.master.to(bank.weight.dtype))
        if state["optimizer"] is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])


def get_controller(model):
    """The controller a model was given, or None.

    Survives `fully_shard`, which returns the module object it was handed, and
    `DistributedDataParallel`, which does not -- DDP returns a wrapper and
    `nn.Module.__getattr__` does not forward unknown attributes to `.module`, so
    a plain getattr on a DDP-wrapped model silently reports "no controller" and
    the banks would never be stepped.
    """
    found = getattr(model, "_memory_parallel", None)
    if found is None:
        inner = getattr(model, "module", None)
        if inner is not None:
            found = getattr(inner, "_memory_parallel", None)
    return found
