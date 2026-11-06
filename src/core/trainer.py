import itertools
import os
import time
from contextlib import contextmanager, nullcontext
from attr import define
import torch
import torch.nn.functional as F
from typing import Optional
from torch.utils.data import IterableDataset
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# FSDP2's module type, for O9's guard. Public under torch.distributed.fsdp from
# 2.7 (entropy: 2.7.1+cu128); older builds expose FSDP2 only under the private
# _composable path (helios ships 2.6.0 on aarch64). None means this build has no
# FSDP2 at all, in which case O9 degrades to a no-op rather than crashing the
# import for every run on that cluster.
from torch.nn.parallel import DistributedDataParallel as _DDP

try:
    from torch.distributed.fsdp import FSDPModule as _FSDPModule
except ImportError:  # pragma: no cover - depends on the cluster's torch
    try:
        from torch.distributed._composable.fsdp import FSDPModule as _FSDPModule
    except ImportError:
        _FSDPModule = None

import torch.distributed.checkpoint as dcp
import logging

from src.core.checkpointing import (
    TrainingState,
    save_training_state,
    step_checkpoint_path,
)
from src.core.metric_loggers import AveDiffMetric, AveMetric, MetricLogger
from src.core.memory_parallel import get_controller
from src.core.mfu import model_flops_per_token, peak_flops_bf16
from src.core.utils import create_batch_fingerprint

logger = logging.getLogger(__name__)


@define(frozen=True)
class LossMetrics:
    total_loss: torch.Tensor
    reported_loss: torch.Tensor


class LoopTimers:
    """Section timers for one training-loop iteration, comparable across
    frameworks (the same class is patched into nanoGPT's train.py for the
    side-by-side profile).

    GPU sections use CUDA event pairs: kernels are launched asynchronously, so
    host timestamps around a section measure launch time, not execution; event
    pairs measure the section's span on the stream. Host sections (the wait for
    the next batch) use perf_counter. read() synchronizes once per step and
    returns {section: milliseconds}. Host and GPU sections overlap in wall
    time, so their sum can exceed the step total. Without CUDA, GPU sections
    fall back to host timing so the CPU gate still runs.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._on_gpu = enabled and torch.cuda.is_available()
        self._host = {}
        self._pairs = []

    @contextmanager
    def gpu(self, name):
        if not self._on_gpu:
            with self.host(name):
                yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            # The same section boundaries double as profiler annotations, so a
            # kernel dump can attribute GPU time to forward/backward instead of
            # guessing from kernel names -- which does not work, because GEMMs
            # and elementwise kernels serve both passes under identical names.
            # Free when no profiler is active.
            with torch.profiler.record_function(f"section:{name}"):
                yield
        finally:
            end.record()
            self._pairs.append((name, start, end))

    @contextmanager
    def host(self, name):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._host[name] = (
                self._host.get(name, 0.0) + (time.perf_counter() - t0) * 1e3
            )

    def read(self):
        """One synchronize, then {section: ms} for this step; resets state."""
        if not self.enabled:
            return {}
        if self._on_gpu:
            torch.cuda.synchronize()
        out = {}
        for name, start, end in self._pairs:
            out[name] = out.get(name, 0.0) + start.elapsed_time(end)
        for name, ms in self._host.items():
            out[name] = out.get(name, 0.0) + ms
        self._pairs.clear()
        self._host.clear()
        return out


@define(slots=False)
class Trainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    gradient_accumulation_steps: int
    training_state: dict
    n_steps: int
    train_dataloader: IterableDataset
    eval_dataloader: IterableDataset
    metric_logger: MetricLogger
    eval_interval: int
    n_eval_steps: int
    gradient_clipping: Optional[float]
    checkpoint: Optional[dict]
    learning_rate: float
    weight_decay: float
    distributed: Optional[dict]
    # The nanoGPT-style single-GPU configuration: a plain (unwrapped) module
    # whose forward+loss run under torch.autocast(bf16) while params, grads
    # and optimizer state stay fp32. Off by default; meant for runs with
    # trainer.distributed=null, where there is no FSDP2 policy to supply bf16.
    autocast_bf16: bool = False
    # Per-step section timers (LoopTimers) printed as a `timers_ms:` log line;
    # measurement-only, off by default. The one synchronize per step sits where
    # the metric logger's .item() already syncs, so step time is unaffected.
    profile_timers: bool = False
    # Kernel-level profile: torch.profiler (CUDA activity) wrapped around
    # steps 12-13, dumped as one `kprof_us:` line per kernel (self time,
    # count, name) for offline aggregation into categories (gemm / attention /
    # fused / eager elementwise / ...). The same dump format is patched into
    # nanoGPT's train.py so the two profiles are directly comparable.
    kernel_profile: bool = False
    # Compile model forward + cross-entropy as ONE inductor graph (what
    # nanoGPT gets by computing the loss inside its compiled model). Use
    # INSTEAD of compile_model, not on top of it: the traced function already
    # contains the model. Speedup-plan item 2.
    compile_loss: bool = False
    # Compile ONLY the cross-entropy, leaving the model's compile scope alone.
    # Unlike compile_loss this never traces self.model, so it composes with
    # FSDP2 and with regional block compile instead of replacing them.
    #
    # MEASURED HARMFUL, 2026-11-03 -- do not enable without re-testing. It buys
    # ~1% step time and 3 GiB, and shifts the loss: over 300 steps at h20 it
    # sits ABOVE the uncompiled-CE twin in 50 of the last 50 steps, last-50 mean
    # |dloss| 0.110 against 0.006 for the same run without it
    # (documents/notes/multigpu_efficiency.md 7.9). A one-directional shift is
    # not the unbiased noise that fusion and reassociation produce.
    #
    # Suspected but NOT confirmed: eager F.cross_entropy accumulates the
    # log-sum-exp in fp32 even for bf16 inputs, and inductor's own kernel need
    # not keep that upcast -- which over a 50304-wide vocabulary matters.
    # Unresolved either way: whether the gradients are worse or only the
    # reported number is biased. Evaluating both final models under one eager
    # fp32 loss would settle it. Rejected regardless: the first is worse
    # training, the second is a metric that lies.
    compile_ce: bool = False

    # O11, fused-linear-CE. Off by default, so every committed number stands.
    #
    # On, the head's projection and the cross-entropy run in chunks of rows and
    # the full [micro_batch, seq, vocab] logits tensor is never materialised.
    # That tensor is the largest activation in the step: 3.07 GiB at h16 /
    # micro-batch 16, 6.14 at micro-batch 32, which is the allocation job 32853
    # OOMed on.
    #
    # Expect MEMORY, not speed. Non-NCCL kernels are 2715.9 ms of the 2893 ms
    # 4-GPU step, so today's kernels cap 4-GPU MFU near 36.8% even with
    # communication perfectly hidden; O11 pays by unblocking a larger
    # micro-batch, and it costs a second head matmul per chunk in backward.
    #
    # `fused_ce_chunk_size` is in ROWS (micro_batch * seq flattened). None means
    # one chunk, i.e. the memory of the unfused path -- set it to something like
    # 2048 to get the win.
    fused_linear_ce: bool = False
    fused_ce_chunk_size: Optional[int] = None
    # O9: under FSDP2, reduce gradients ONCE per optimizer step instead of once
    # per micro-step. Every `loss.backward()` in the accumulation loop otherwise
    # triggers a full reduce-scatter of every gradient, and only the last one is
    # needed -- at gradient_accumulation_steps=32 that is 32 reduce-scatters
    # where 1 suffices. `set_requires_gradient_sync(False)` is FSDP2's
    # equivalent of FSDP1's `no_sync`, and it is mathematically an identity:
    # it changes WHEN gradients are reduced, never what they sum to.
    #
    # NOT free in memory, which is why it is opt-in. With sync suppressed,
    # FSDP2 holds each parameter group's UNSHARDED gradient between micro-steps
    # and, because `reduce_dtype=torch.float32`, upcasts it to fp32 to do so
    # (torch/distributed/fsdp/_fully_shard/_fsdp_param.py,
    # `to_accumulated_grad_if_needed` / `accumulate_unsharded_grad_if_needed`).
    # The resident gradient therefore goes from `P/world_size * 4` bytes to
    # `P * 4` bytes per rank -- at h20's 3.163B parameters on 4 GPUs, 3.16 GB
    # becomes 12.65 GB, i.e. ~9.5 GB/GPU more. Parameters still reshard after
    # backward; it is only the gradient that stays unsharded.
    #
    # So: a FLOP-free throughput win paid for in memory, which may cap the
    # micro-batch. Enable it and measure both numbers, never just the first.
    # Ignored without FSDP2 (plain module, FSDP1, CPU) and when
    # gradient_accumulation_steps == 1, where it would be a no-op anyway.
    skip_grad_sync_during_accumulation: bool = False
    # Tiers 2+3: how many ranks share one copy of the Q/K banks. Declared here
    # so `trainer.mem_parallel_size=N` type-checks at submit time, but the
    # controller itself is built in main.py before fully_shard -- the trainer
    # finds it on the model (`get_controller`) and only cross-checks it against
    # this value, so a config that sets one and a model that was built without
    # the other fails loudly instead of silently training the baseline.
    mem_parallel_size: Optional[int] = None
    # The training length in billions of tokens, when the config gave it that
    # way. Read by nobody here: src.definitions.resolve_n_steps has already
    # turned it into n_steps before this class is built. Declared because
    # every trainer config key reaches the constructor, and kept so the run
    # record shows what was asked for beside the steps it became.
    n_tokens_b: Optional[float] = None

    # Checkpoint save types this trainer can actually write. Subclasses widen this;
    # anything outside it is rejected before training starts.
    SUPPORTED_SAVE_TYPES = ("nano",)

    # Of those, the ones written once when training ends rather than on the periodic
    # interval. Subclasses that add such a type widen this too.
    FINAL_ONLY_SAVE_TYPES = ()

    def __attrs_post_init__(self):
        self._validate_save_type()
        self.processed_tokens = self.training_state["processed_tokens"]
        self.start_step = self.training_state["next_step"]
        self.device = next(self.model.parameters()).device

        # A bare `None` in YAML is the *string* "None" -- YAML's null is `null` or `~`.
        # 37 shipped configs switch eval off with `eval_dataloader: None`, so without
        # this the attribute arrives as a str, which is not None and does have
        # __iter__, and so passes for a dataloader until something calls .to() on one
        # of its characters. Normalise once, here, so every later check is `is None`.
        if isinstance(self.eval_dataloader, str):
            logger.info(
                f"eval_dataloader is the string {self.eval_dataloader!r}, which YAML "
                "produces for a bare `None`; treating it as no eval dataloader."
            )
            self.eval_dataloader = None

        # Not built here. Resuming used to wind this forward by
        # (start_step - 1) // eval_interval * n_eval_steps batches, because one iterator
        # was carried across the whole run and a resumed run had to land where it left
        # off. eval() now restarts the stream on every call -- that restart is what makes
        # the eval set the same samples every time -- so a wound-forward iterator would
        # be discarded unread, after pulling ~10k batches through the tokenizer on a long
        # resume. Do not add the skip back without first removing that restart.
        self.eval_iterator = None
        self.step = self.start_step - 1
        # Disabled placeholder so calculate_loss (also called from eval) can
        # always reference it; train() swaps in the enabled instance.
        self._timers = LoopTimers(False)
        # Resolved lazily on first use (calculate_loss): the save-type test
        # stubs borrow this post-init without the rest of the class.
        self._forward_loss_fn = None
        self._ce_fn = None

        self.loss_averaged_100 = AveMetric(100, "100/train/loss")
        self.time_diff_averaged_100 = AveDiffMetric(100, "100/time", time.time())

        # MFU accounting (src/core/mfu.py). Wall time between successive
        # log_metrics calls covers one full iteration (optimizer step, data
        # fetch, forward+backward), just shifted by half a step -- fine for the
        # steady-state windows the audit averages over. Filled lazily: the
        # sequence length is only known once the first batch arrives.
        self._prev_log_time = None
        self._prev_log_tokens = None
        self._flops_per_token = None
        self._peak_flops = None
        self._seq_len_for_flops = None
        # The last micro-batch's ids, for per-layer attention diagnostics.
        self._last_input_ids = None
        # Modules whose instrumentation raised; warned about once, then skipped
        # (see _log_layer_instrumentation).
        self._instrumentation_failures = set()

        # Tiers 2+3. None unless main.py built a MemoryParallelController, in
        # which case the Q/K banks are outside FSDP2 and outside self.optimizer
        # and this object is the only thing that steps them.
        self._banks = get_controller(self.model)
        requested = self.mem_parallel_size
        if requested is not None and requested > 0 and self._banks is None:
            raise ValueError(
                f"trainer.mem_parallel_size={requested} but the model carries no "
                "memory-parallel controller -- the banks would be trained by the "
                "ordinary FSDP2 path and the run would silently be a baseline"
            )
        if self._banks is not None and requested not in (None, self._banks.mp_size):
            raise ValueError(
                f"trainer.mem_parallel_size={requested} disagrees with the "
                f"controller's {self._banks.mp_size}"
            )

    @property
    def _should_evaluate(self) -> bool:
        return (
            self.eval_interval > 0
            and self.step % self.eval_interval == 0
            and self.step != 0
        )

    @property
    def _should_log_eval_input(self) -> bool:
        return self.step % (self.eval_interval * 100) == 0

    @property
    def _should_save_checkpoint(self) -> bool:
        return (
            self.checkpoint.save.interval > 0
            and (self.step) % self.checkpoint.save.interval == 0
            and self.step != 0
            and self.checkpoint.save.path is not None
        )

    @property
    def _should_save_final_checkpoint(self) -> bool:
        return (
            not self._should_save_checkpoint  # checkpoint was already saved
            and self.step >= self.n_steps - 1
            and self.checkpoint.save.path is not None
        )

    def _batches_from_start_step(self):
        """Training batches, fast-forwarded past the steps a previous run already took.

        The datasets are deterministic iterables that reseed in __iter__ (see
        AbstractDataset), and train() consumes exactly one batch per step. A resumed run
        therefore gets a *fresh* iterator starting at batch 0 and re-trains on data it
        has already seen, while the step counter carries on from the checkpoint.

        That fails quietly in the worst way: the loss after a resume comes out lower,
        not higher, because the model is being shown examples it already learned.

        Skipping start_step batches reproduces the sequence an uninterrupted run would
        have seen. It costs one pass over the skipped batches, which is the price of a
        dataset that cannot seek.
        """
        if not self.start_step:
            return self.train_dataloader
        logger.info(
            f"Resuming at step {self.start_step}: skipping that many batches so the run "
            "does not repeat data it has already trained on."
        )
        return itertools.islice(self.train_dataloader, self.start_step, None)

    def _kernel_profile_tick(self, step):
        """Open torch.profiler at step 12, close and dump it at step 14, so
        exactly two steady-state steps (12-13) are covered, warmup and the
        compile steps excluded. One `kprof_us:` line per kernel."""
        if not self.kernel_profile:
            return
        if step == 12:
            # CPU activity is on so the dump carries the operator tree as well
            # as raw kernels: aten ops and autograd nodes name themselves, and
            # the `section:` ranges LoopTimers opens give exact forward/backward
            # attribution of GPU time. Without it only bare kernel names exist,
            # which cannot be split by pass.
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self._kprof = torch.profiler.profile(activities=activities)
            self._kprof.__enter__()
        elif step == 14 and getattr(self, "_kprof", None) is not None:
            self._kprof.__exit__(None, None, None)
            # The chrome trace, which is the only form that keeps PARENTAGE.
            # key_averages() below aggregates by kernel name, and a GEMM or an
            # elementwise kernel serves forward and backward under one name --
            # so the aggregate cannot be split by pass, however much the
            # `section:` ranges above suggest it could. The trace keeps every
            # kernel's External id, so scripts/kprof_phase.py can walk each one
            # back to the `section:` range that was open when it was launched
            # and give a real forward/backward split per category.
            if int(os.environ.get("RANK", 0)) == 0:
                trace_path = os.path.abspath("kprof_trace_rank0.json")
                try:
                    self._kprof.export_chrome_trace(trace_path)
                    size_mb = os.path.getsize(trace_path) / 2**20
                    logger.info(f"kprof_trace: {trace_path} ({size_mb:.1f} MiB)")
                except Exception as exc:  # noqa: BLE001
                    # Never lose a run over a profiling artefact.
                    logger.warning(f"kprof_trace export failed: {exc}")
            # Every rank profiles (the context manager is cheap and symmetric),
            # but only rank 0 prints. The lines carry no rank of their own and
            # all ranks share one job log, so N ranks printing would interleave
            # into a stream no aggregator could split -- and scripts/kprof_agg.py
            # would read N x every kernel's time.
            if int(os.environ.get("RANK", 0)) == 0:
                for ka in self._kprof.key_averages():
                    # With CPU activity recorded, key_averages() holds both the
                    # operators and the kernels they launch, and an operator can
                    # carry device time of its own. Emitting both into kprof_us
                    # would double-count every GEMM (aten::mm alongside nvjet)
                    # and every collective (nccl:_all_gather_base alongside
                    # ncclDevKernel), so the kernel stream is restricted to
                    # CUDA-side entries and the operators go to kprof_op below.
                    is_kernel = "CUDA" in str(getattr(ka, "device_type", ""))
                    self_us = getattr(ka, "self_device_time_total", 0) or getattr(
                        ka, "self_cuda_time_total", 0
                    )
                    if is_kernel and self_us > 0:
                        logger.info(
                            f"kprof_us: self_cuda={self_us:.0f} count={ka.count} name={ka.key}"
                        )
                    # Operators and `section:` ranges carry GPU time in their
                    # subtree rather than themselves, so they are reported on
                    # the inclusive total: that is what attributes kernels to a
                    # pass or an op. Emitted separately so kprof_agg.py, which
                    # sums self-time, cannot double-count them.
                    total_us = getattr(ka, "device_time_total", 0) or getattr(
                        ka, "cuda_time_total", 0
                    )
                    if not is_kernel and total_us > 0:
                        logger.info(
                            f"kprof_op: cuda_total={total_us:.0f} self={self_us:.0f} "
                            f"count={ka.count} name={ka.key}"
                        )
            self._kprof = None

    def train(self):
        self._timers = LoopTimers(self.profile_timers)
        batches = iter(self._batches_from_start_step())
        for step in range(self.start_step, self.n_steps):
            self._kernel_profile_tick(step)
            step_t0 = time.perf_counter()
            # The host-side wait for the next batch (tokenizer workers,
            # collate, pinning) -- the H2D copy itself is timed as data_h2d
            # inside calculate_loss.
            with self._timers.host("data_fetch"):
                try:
                    batch = next(batches)
                except StopIteration:
                    break
            self.step = step
            self.metric_logger.set_step(step)
            self.metric_logger.set_tokens(self.processed_tokens)
            self.model.train()
            if self._banks is not None:
                # O5 hook 1, step begin: the fp32 master and the gradient
                # accumulator are plain tensors nothing else moves, so this is
                # where they are allocated (first step) and re-homed.
                self._banks.materialize()
            # The model sees batch width - 1 tokens (the shift for next-token
            # targets); that is the sequence length the FLOPs count must use.
            self._seq_len_for_flops = batch.shape[-1] - 1
            loss_metrics = self.calculate_loss(batch)

            if self._banks is not None:
                # O5 hook 2 / O2 / O8: one all-reduce for the whole optimizer
                # step. BEFORE the clip, because the clip's norm has to see the
                # final gradient, not this rank's partial sum.
                with self._timers.gpu("bank_reduce"):
                    self._banks.reduce()

            with self._timers.gpu("clip"):
                grad_norm = self.clip_gradient()

            with self._timers.host("log"):
                self.log_metrics(loss_metrics, grad_norm)

            with self._timers.gpu("optim"):
                self.optimizer.step()
                self.optimizer.zero_grad()
            if self._banks is not None:
                # Read the LR off the backbone optimizer, not off the
                # scheduler's state: scheduler.step() has not run yet, so this
                # is the LR this step was taken at, and copying it keeps the
                # banks on exactly the backbone's schedule.
                #
                # Timed BESIDE `optim`, not inside it: the printed sections are
                # read as a partition of the step, and a nested pair would have
                # the bank step counted twice.
                with self._timers.gpu("bank_step"):
                    self._banks.step(lr=self.optimizer.param_groups[0]["lr"])
                    self._banks.zero_grad()
            self.scheduler.step()

            if self.profile_timers:
                sections = self._timers.read()  # synchronizes
                step_total = (time.perf_counter() - step_t0) * 1e3
                parts = " ".join(
                    f"{name}={sections.get(name, 0.0):.1f}"
                    for name in (
                        "data_fetch",
                        "data_h2d",
                        "forward",
                        "backward",
                        "clip",
                        "optim",
                        "bank_reduce",
                        "bank_step",
                        "log",
                    )
                )
                # Tagged with the rank: all ranks share one job log, and on
                # multi-GPU the spread between ranks is the straggler signal,
                # so these are worth keeping per rank rather than silencing.
                logger.info(
                    f"timers_ms: rank={int(os.environ.get('RANK', 0))} "
                    f"step={step} {parts} step_total={step_total:.1f}"
                )

            if self._should_save_checkpoint:
                self.save_checkpoint()

            if self._should_evaluate:
                self.eval()

            self.metric_logger.flush()

        if torch.cuda.is_available():
            peak_gib = torch.cuda.max_memory_allocated(self.device) / 2**30
            self.metric_logger.log("train/max_memory_gib", peak_gib)
            logger.info(f"Peak CUDA memory: {peak_gib:.2f} GiB")

        self.final_eval()

        if self._should_save_final_checkpoint:
            self.save_final_checkpoint()

    def final_eval(self):
        """Score the held-out set once, after training, and log it as `final_ce`.

        The periodic eval reads `n_eval_steps` batches so it stays cheap enough to run
        during training. This one reads the whole held-out set, which
        `_dataset/fineweb.yaml` pins to 8192 samples of 2048 tokens -- the same samples
        for every run, so the number is comparable across them.

        It runs before the final checkpoint on purpose. A subclass's final save may
        consume the model -- rewriting its parameters, or dropping them off rank 0 --
        so a score taken afterwards would be of whatever the save left behind.
        """
        if self.eval_dataloader is None:
            logger.info("No eval dataloader; skipping the final eval.")
            return

        dataset = getattr(self.eval_dataloader, "dataset", None)
        max_samples = getattr(dataset, "max_samples", None)
        batches = iter(self.eval_dataloader)
        if max_samples is None:
            # An uncapped eval stream never ends, so read the periodic eval's budget
            # rather than hang. Set max_samples on the eval dataset for a fixed set.
            logger.warning(
                "Eval dataset has no max_samples, so the stream is endless; "
                f"final_ce will cover {self.n_eval_steps} batches instead of a fixed set."
            )
            batches = itertools.islice(batches, self.n_eval_steps)

        self.model.eval()
        # Accumulate a weighted sum and a count rather than a list of batch means. The
        # last batch of a fixed-size set is usually short, and with world_size_independent
        # the ranks get within one sample of each other but not exactly the same number,
        # so a mean of means is weighted wrong on both counts.
        loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        n_samples = torch.zeros((), device=self.device, dtype=torch.float64)
        skipped = 0
        with torch.no_grad():
            for batch in batches:
                size = batch.shape[0]
                if size % self.gradient_accumulation_steps:
                    # calculate_loss splits the batch into gradient_accumulation_steps
                    # chunks and divides each chunk's loss by that same number. On a
                    # batch too short to fill every chunk it produces fewer chunks and
                    # still divides by the full count, so the loss comes back
                    # proportionally too low. Skipping the stub is better than folding a
                    # known-wrong number into final_ce; with max_samples a multiple of
                    # the batch size, which the shipped config is, there is no stub.
                    skipped += size
                    continue
                batch = batch.to(self.device)
                loss_metrics = self.calculate_loss(batch)
                loss_sum += loss_metrics.reported_loss.detach().double() * size
                n_samples += size
                self.metric_logger.flush_accumulated_metrics()

        # One reduction, after the loop. Reducing inside it would deadlock the moment
        # two ranks got a different number of batches, which a fixed-size set split
        # across ranks can easily produce.
        if dist.is_initialized():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)

        if n_samples.item() == 0:
            logger.warning("Final eval produced no batches; not logging final_ce.")
            return

        if skipped:
            logger.warning(
                f"Final eval skipped {skipped} samples in batches too short to fill "
                f"gradient_accumulation_steps={self.gradient_accumulation_steps}; set "
                f"max_samples to a multiple of the batch size to score all of them."
            )

        final_ce = (loss_sum / n_samples).item()
        logger.info(f"final_ce over {int(n_samples.item())} samples: {final_ce}")
        self.metric_logger.log("final_ce", final_ce)
        self.metric_logger.flush()

    def save_final_checkpoint(self):
        """Write the checkpoint that is produced once, when training ends.

        Subclasses that widen SUPPORTED_SAVE_TYPES override this to handle the type they
        added, and defer to super() for the rest.
        """
        if self.checkpoint.save.type == "nano":
            self.save_checkpoint()
        else:
            raise ValueError(
                f"No final-checkpoint writer for save.type="
                f"'{self.checkpoint.save.type}' in {type(self).__name__}."
            )

    def _preprocess_input(self, batch):  # TODO test it
        input_ids = batch[:, :-1].contiguous()
        target_ids = batch[:, 1:].contiguous()

        return input_ids, target_ids

    def _forward_ctx(self):
        if self.autocast_bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _loss_ctx(self):
        """Cross-entropy runs with autocast OFF, and that is a memory decision.

        `F.cross_entropy` is on torch's autocast **fp32 list**, so inside an
        autocast region the logits are upcast before the kernel sees them. At
        h20 that is a `[micro_batch, 2047, 50304]` fp32 tensor -- 6.6 GiB at
        micro-batch 16 -- materialised on top of the bf16 logits that already
        exist. It is what OOMed the first DDP run (job 32672, which died
        allocating exactly 6.14 GiB in `cross_entropy`).

        Outside autocast the bf16 logits go straight in and eager
        `F.cross_entropy` accumulates its log-sum-exp in fp32 internally,
        without a full-size fp32 copy. That is also **exactly what the FSDP2
        path does** -- it has no autocast at all, so its bf16 parameters produce
        bf16 logits and the same eager kernel runs. Matching them is what makes
        a DDP run comparable to an FSDP2 one rather than a different
        computation that happens to train.

        A no-op when `autocast_bf16` is off, so no FSDP2 measurement moves.
        """
        if self.autocast_bf16:
            return torch.autocast("cuda", enabled=False)
        return nullcontext()

    def _forward_loss(self, input_ids, target_ids):
        """Model forward + cross-entropy as one function, so compile_loss can
        trace them into a single graph. With the loss outside the compiled
        model (nanoGPT keeps it inside), the logits -- batch x seq x vocab,
        the largest activation there is -- cross the graph boundary every
        micro-step and pay an eager fp32 log-softmax plus bf16<->fp32
        boundary casts; the kernel profile measured that at ~500 ms/step at
        h20 (training_efficiency.md Part 4.2)."""
        if self.fused_linear_ce:
            # O11. The loss comes back from the model; no logits cross this
            # boundary, which is the whole point.
            #
            # `_loss_ctx()` still wraps it, and that is load-bearing rather
            # than copied: `F.cross_entropy` is on autocast's fp32 list, so
            # inside an autocast region it upcasts whatever logits it is handed.
            # Chunking shrinks that tensor but does not remove the upcast --
            # a 2048-row chunk would still become fp32 at 50304 wide. Disabling
            # autocast around the loss keeps every chunk in the model's dtype,
            # exactly as it does on the unfused path.
            target_ids = target_ids.to(next(self.model.parameters()).device)
            with self._loss_ctx():
                return self.model(
                    input_ids,
                    targets=target_ids,
                    ce_chunk_size=self.fused_ce_chunk_size,
                )

        predicted_ids = self.model(input_ids)

        # Tensors should be on the same device for loss calculation #TODO check
        target_ids = target_ids.to(predicted_ids.device)

        if self._ce_fn is None:
            # Lazily, like _forward_loss_fn above: the save-type test stubs
            # borrow __attrs_post_init__ without the rest of the class.
            self._ce_fn = (
                torch.compile(type(self)._cross_entropy)
                if self.compile_ce
                else type(self)._cross_entropy
            )
        with self._loss_ctx():
            return self._ce_fn(predicted_ids, target_ids)

    @staticmethod
    def _cross_entropy(predicted_ids, target_ids):
        """Cross-entropy alone, compilable without tracing the model.

        Split out so `compile_ce` can fuse the log-softmax, the NLL and the
        bf16<->fp32 casts over the logits -- the largest activation in the step
        -- while the model keeps whatever compile scope it was given. Under
        FSDP2 that distinction matters: `compile_loss` traces `self.model`,
        so it is whole-model compile and has to survive the all-gather hooks,
        whereas this does not touch the model at all
        (documents/notes/multigpu_compile_scope_plan.md, R2).
        """
        mask_loss = F.cross_entropy(
            predicted_ids.flatten(0, -2),
            target_ids.reshape(-1).long(),
            reduction="none",
        )
        return mask_loss.mean()

    @contextmanager
    def _accumulation_grad_sync(self):
        """Yields `mark(is_last)`, which says whether this micro-step may reduce.

        O9. Under FSDP2 every `loss.backward()` ends in a reduce-scatter of every
        gradient. Across an accumulation loop only the last one is needed -- the
        others reduce a partial sum that the next micro-step immediately adds to.
        `set_requires_gradient_sync(False)` suppresses them; the gradients
        accumulate locally instead and one reduce at the end produces the same
        sum. FSDP1 spelled this `no_sync()`.

        Identity, not approximation: addition is associative, and the reduce is
        a sum over ranks. What changes is the schedule, not the result -- which
        is why the gate for this change is a *bit-identical* loss trajectory.

        The restore in the `finally` is the part worth not getting wrong. If the
        loop raises part-way through, or if eval runs next, a model left with
        sync off would keep training without ever reducing gradients -- each
        rank silently diverging onto its own model while the loss curve looks
        entirely normal. Always hand it back synchronised.

        Inert, with no behaviour change at all, unless the model is an FSDP2
        module, the opt-in flag is set, there is more than one micro-step, and
        the model is in training mode.
        """
        if self.skip_grad_sync_during_accumulation and _FSDPModule is None:
            logger.warning(
                "skip_grad_sync_during_accumulation is set, but this torch build "
                "(%s) exposes no FSDP2 module type; gradients will be reduced on "
                "every micro-step as before.",
                torch.__version__,
            )
        accumulating = self.gradient_accumulation_steps > 1 and self.model.training

        if accumulating and isinstance(self.model, _DDP):
            # Unconditional for DDP, unlike the FSDP2 branch below, because here
            # it is free. FSDP2 pays for suppressed sync by holding each
            # gradient UNSHARDED between micro-steps (+9.5 GB/GPU at h20), which
            # is why that path is opt-in. DDP's gradient buffers are full-size
            # either way, so `no_sync` skips the all-reduce and costs nothing --
            # and reducing once per optimizer step instead of once per
            # micro-step is never the wrong choice when it is free.
            entered = []

            def mark_ddp(is_last):
                if is_last:
                    while entered:
                        entered.pop().__exit__(None, None, None)
                elif not entered:
                    context = self.model.no_sync()
                    context.__enter__()
                    entered.append(context)

            try:
                yield mark_ddp
            finally:
                while entered:
                    entered.pop().__exit__(None, None, None)
            return

        active = (
            self.skip_grad_sync_during_accumulation
            and accumulating
            and _FSDPModule is not None
            and isinstance(self.model, _FSDPModule)
        )
        if not active:
            yield lambda _is_last: None
            return
        try:
            yield self.model.set_requires_gradient_sync
        finally:
            self.model.set_requires_gradient_sync(True)

    def calculate_loss(self, batch) -> LossMetrics:
        if self._forward_loss_fn is None:
            self._forward_loss_fn = (
                torch.compile(self._forward_loss)
                if self.compile_loss
                else self._forward_loss
            )

        def _hack_for_python_garbage_collection(input_ids, target_ids):
            """we want to have no reference to model output while backpropagating to allow torch to free memory,
            so we wrap loss calculation in a function"""
            with self._forward_ctx():
                reported_loss = self._forward_loss_fn(input_ids, target_ids)
            loss = reported_loss
            reported_loss = reported_loss / self.gradient_accumulation_steps
            loss = loss / self.gradient_accumulation_steps
            return loss, reported_loss

        chunks = batch.chunk(self.gradient_accumulation_steps)

        def _to_device(chunk):
            # chunk() along dim 0 of the loader's pinned, contiguous batch
            # yields contiguous pinned views, so this copy is genuinely
            # asynchronous -- the Python thread moves on to launch kernels
            # instead of stalling until the copy lands. The input/target
            # slicing then happens on the device.
            with self._timers.gpu("data_h2d"):
                return chunk.to(self.device, non_blocking=True)

        total_losses = []
        reported_losses = []
        in_flight = _to_device(chunks[0])
        with self._accumulation_grad_sync() as mark_micro_step:
            for i in range(len(chunks)):
                input_ids, target_ids = self._preprocess_input(in_flight)
                # Kept for the per-layer diagnostics: an id-indexed layer's
                # attention entropy is a function of the ids, and this is the only
                # place that has them. One micro-batch of int64, and it is eager
                # code outside whatever compile scope the model was given.
                self._last_input_ids = input_ids
                if self.model.training:
                    self._update_processed_tokens(input_ids)
                    # O9. Only the final micro-step's backward may reduce: the
                    # earlier ones accumulate locally. A no-op unless FSDP2 and
                    # the opt-in flag are both in play.
                    mark_micro_step(i + 1 == len(chunks))

                with self._timers.gpu("forward"):
                    loss, reported_loss = _hack_for_python_garbage_collection(
                        input_ids, target_ids
                    )
                if self.model.training:
                    with self._timers.gpu("backward"):
                        loss.backward()
                # The next chunk's H2D copy goes out while this chunk's kernels
                # are still executing, not at the top of the next iteration.
                if i + 1 < len(chunks):
                    in_flight = _to_device(chunks[i + 1])
                total_losses.append(loss.detach())
                reported_losses.append(reported_loss.detach())

        # gloo backend supports only sum reduce operation, therfore we first divide by world size and then sum
        avg_total_loss = torch.stack(total_losses).sum()
        avg_reported_loss = torch.stack(reported_losses).sum()
        if dist.is_initialized():
            dist.all_reduce(avg_total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_reported_loss, op=dist.ReduceOp.SUM)

        world_size = float(os.environ["WORLD_SIZE"])
        return LossMetrics(
            total_loss=avg_total_loss / world_size,
            reported_loss=avg_reported_loss / world_size,
        )

    def eval(self):
        self.model.eval()
        saved_step = self.step
        self.metric_logger.set_step(None)  # disables heavy logging
        losses = []
        eval_fingerprint = []
        # Restart the eval stream every time. The iterator used to be created once and
        # carried forward, so each evaluation scored whatever came next rather than the
        # same held-out samples -- eval loss then moved between evaluations partly
        # because the data moved, which is not what an eval number is for. The dataset
        # reseeds in __iter__, so a fresh iterator replays the same samples in the same
        # order.
        self.eval_iterator = iter(self.eval_dataloader)
        with torch.no_grad():
            for _ in range(self.n_eval_steps):
                batch = next(self.eval_iterator)
                batch_fingerprint = create_batch_fingerprint(batch)
                eval_fingerprint.extend(batch_fingerprint)
                batch = batch.to(self.device)
                loss_metrics = self.calculate_loss(batch)
                losses.append(loss_metrics.reported_loss.detach())
                self.metric_logger.flush_accumulated_metrics()
            avg_loss = torch.stack(losses).mean()
            self.metric_logger.log("eval/loss", avg_loss.item())

        if self._should_log_eval_input:
            self.metric_logger.log("eval/batch", str(eval_fingerprint))

        self.step = saved_step  # Restore step
        self.metric_logger.set_step(saved_step)

    def clip_gradient(self):
        if self.gradient_clipping is not None:
            if isinstance(self.model, FSDP):
                return self.model.clip_grad_norm_(self.gradient_clipping)
            else:
                # Plain clip_grad_norm_ on FSDP2 DTensor sharded grads only computes
                # the local shard norm without cross-rank reduction, underestimating
                # the global norm by ~sqrt(world_size). Compute it correctly, and in
                # a handful of multi-tensor kernels rather than one norm launch per
                # parameter (~150 serial launches per step on an h16).
                params = [p for p in self.model.parameters() if p.grad is not None]
                local_grads = [
                    p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
                    for p in params
                ]
                local_norm_sq = (
                    torch.nn.utils.get_total_norm(local_grads, 2.0).float() ** 2
                )
                # Under DDP the backbone gradients are ALREADY all-reduced and
                # identical on every rank, so their norm is the global norm
                # before any reduction here. Summing it across ranks again would
                # inflate the total by sqrt(world_size) -- a factor of 2 at four
                # GPUs -- and clip twice as hard as asked, silently. Only the
                # bank slices, which differ per rank, need summing.
                replicated = isinstance(self.model, _DDP)
                if self._banks is not None:
                    # O4, and deliberately a JOINT clip rather than the separate
                    # one the plan's table lists. The baseline computes one
                    # global norm over every parameter and scales all of them by
                    # one coefficient; clipping the banks on their own is a
                    # different function of the same gradients, and the section
                    # 1.4 gate compares parameter-for-parameter against that
                    # baseline. The banks' own gradients never reach `params`
                    # here -- `_BankGather.backward` returns None, so
                    # `weight.grad` stays None and the filter above drops them --
                    # so their norm has to be added explicitly.
                    bank_norm_sq = self._banks.local_grad_norm_sq(local_norm_sq.device)
                    if replicated and dist.is_initialized():
                        dist.all_reduce(bank_norm_sq, op=dist.ReduceOp.SUM)
                    local_norm_sq = local_norm_sq + bank_norm_sq
                if dist.is_initialized() and not replicated:
                    dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM)
                total_norm = local_norm_sq.sqrt()
                torch.nn.utils.clip_grads_with_norm_(
                    params, self.gradient_clipping, total_norm
                )
                if self._banks is not None:
                    self._banks.clip_(self.gradient_clipping, total_norm)
                return total_norm

    def _update_processed_tokens(self, batch):
        self.processed_tokens += batch.numel() * int(os.environ["WORLD_SIZE"])

    def log_metrics(self, loss_metrics: LossMetrics, grad_norm):
        self.metric_logger.set_tokens(self.processed_tokens)
        self.metric_logger.log("train/loss", loss_metrics.reported_loss.item())
        self.metric_logger.log("train/lr", self.scheduler.get_last_lr()[0])
        if grad_norm is not None:
            self.metric_logger.log("train/grad_norm", grad_norm.item())

        if self._banks is not None:
            # O5 hook 3. The banks are stepped by their own optimizer, so "the
            # LR" is no longer one number by construction -- log the one they
            # were actually stepped at.
            self._banks.log_metrics(
                self.metric_logger, self.optimizer.param_groups[0]["lr"]
            )

        self.loss_averaged_100.log(
            self.metric_logger, loss_metrics.reported_loss.item()
        )
        self.time_diff_averaged_100.log(self.metric_logger, time.time())
        self._log_throughput()
        self._log_layer_instrumentation()

        self.metric_logger.flush_accumulated_metrics()

    def _log_layer_instrumentation(self):
        """Per-layer diagnostics from any module that offers them.

        Duck-typed on an `instrumentation(token_ids=None)` method returning
        `{name: float}`,
        so a layer states its own diagnostics and the trainer stays ignorant of
        attention types. A stack where no module defines it -- every gqa model
        in the repo -- iterates the module list and logs nothing.

        Gated on the same interval as the other heavy metrics: each scalar costs
        a `.item()` sync, and under FSDP2 reading a sharded parameter costs a
        collective, so this is heavy by exactly the definition already in use.
        """
        if not self.metric_logger._should_log_heavy_metrics:
            return
        for module_name, module in self.model.named_modules():
            readout = getattr(module, "instrumentation", None)
            if readout is None:
                continue
            if module_name in self._instrumentation_failures:
                # Already failed once. Do not re-pay the cost every heavy step,
                # and do not re-raise what was already reported.
                continue
            # A diagnostic must never kill a training run. This is observation:
            # if it cannot be taken, the run should lose the observation, not
            # the run.
            #
            # Learned the expensive way on lem, 2026-11-04, jobs 6236888/6236892.
            # An instrumentation readout re-runs part of the model's forward
            # OUTSIDE FSDP2's forward hooks, where the parameters are still
            # sharded DTensors, and `aten.embedding` on a DTensor weight with
            # plain int64 ids raises "got mixed torch.Tensor and DTensor". Both
            # 4-GPU runs died at the first heavy-metric step, after training
            # perfectly well up to it. The readout is a handful of scalars; the
            # run was hours of GPU time.
            try:
                items = readout(token_ids=self._last_input_ids).items()
            except Exception as exc:  # noqa: BLE001 - see above
                if module_name not in self._instrumentation_failures:
                    self._instrumentation_failures.add(module_name)
                    logger.warning(
                        f"instrumentation for {module_name} failed and is "
                        f"disabled for the rest of this run: "
                        f"{type(exc).__name__}: {exc}"
                    )
                continue
            for key, value in items:
                self.metric_logger.log(f"{module_name}/{key}", value)

    def _log_throughput(self):
        """step_time, tokens/s and (on a known GPU) MFU, once per step.

        Pure observation: reads clocks and counters, touches no tensor, so a run
        with and without it is bit-identical. The first step logs nothing -- there
        is no previous timestamp to diff against.
        """
        now = time.time()
        if self._prev_log_time is not None:
            step_time = now - self._prev_log_time
            tokens = self.processed_tokens - self._prev_log_tokens
            if step_time > 0 and tokens > 0:
                tokens_per_s = tokens / step_time
                self.metric_logger.log("train/step_time", step_time)
                self.metric_logger.log("train/tokens_per_s", tokens_per_s)
                if self._flops_per_token is None and torch.cuda.is_available():
                    self._flops_per_token = model_flops_per_token(
                        self.model, self._seq_len_for_flops
                    )
                    device_name = torch.cuda.get_device_name(self.device)
                    # processed_tokens counts GLOBAL tokens (every rank's), so the
                    # denominator must be the GLOBAL peak -- one GPU's peak times
                    # the world size. Dividing global throughput by a single
                    # device's peak reports world_size x the true utilization,
                    # which was invisible while every measured run was world_size 1.
                    world_size = int(os.environ.get("WORLD_SIZE", 1))
                    per_gpu_peak = peak_flops_bf16(device_name)
                    self._peak_flops = per_gpu_peak and per_gpu_peak * world_size
                    logger.info(
                        f"MFU accounting: {device_name} x{world_size}, "
                        f"{self._flops_per_token / 1e9:.3f} GFLOPs/token at "
                        f"seq_len {self._seq_len_for_flops}, peak "
                        f"{self._peak_flops and self._peak_flops / 1e12} TFLOP/s "
                        f"({per_gpu_peak and per_gpu_peak / 1e12} per GPU)"
                    )
                if self._flops_per_token and self._peak_flops:
                    self.metric_logger.log(
                        "train/mfu",
                        tokens_per_s * self._flops_per_token / self._peak_flops,
                    )
        self._prev_log_time = now
        self._prev_log_tokens = self.processed_tokens

    def _validate_save_type(self):
        """Reject a checkpoint save.type this trainer cannot write, before training starts.

        Without this the run trains to completion and only then discovers it has nowhere
        to save to, which on a multi-hour job means losing the whole run.
        """
        save_type = None
        if self.checkpoint is not None and "save" in self.checkpoint:
            save_type = self.checkpoint.save.get("type")

        if save_type is None or save_type in self.SUPPORTED_SAVE_TYPES:
            return

        raise ValueError(
            f"Unknown checkpoint save.type='{save_type}'. {type(self).__name__} supports: "
            f"{', '.join(self.SUPPORTED_SAVE_TYPES)}."
        )

    def save_checkpoint(self):
        save_type = self.checkpoint.save.type  # see SUPPORTED_SAVE_TYPES
        checkpoint_folder = step_checkpoint_path(self.checkpoint.save.path, self.step)

        # Fail before writing anything rather than part-writing a checkpoint.
        self._validate_save_type()

        if save_type in self.FINAL_ONLY_SAVE_TYPES:
            # These are written once, when training ends, by train(). Reaching here means
            # periodic checkpointing was switched on for them, which used to write the
            # training state and no model at all -- a checkpoint interval that silently
            # produced nothing.
            raise NotImplementedError(
                f"Periodic checkpointing is not implemented for save.type='{save_type}', "
                "which is written once when training ends. Set "
                "checkpoint.save.interval: -1, or use save.type='nano' if you need "
                "intermediate checkpoints."
            )

        if save_type == "nano":
            if (
                isinstance(self.model, FSDP)
                or self.model.__module__
                == "torch.distributed.fsdp._fully_shard._fully_shard"
            ):
                # Sharded save
                state_dict = {
                    "app": TrainingState(self.model, self.optimizer, self.scheduler)
                }
                dcp.save(state_dict, checkpoint_id=checkpoint_folder)
                logger.info(f"Saved sharded model checkpoint in {checkpoint_folder}")
            else:
                # Non-sharded save
                if os.environ["RANK"] == "0":
                    os.makedirs(checkpoint_folder, exist_ok=True)
                    checkpoint_path = f"{checkpoint_folder}/{self.checkpoint.save.model_checkpoint_filename}"
                    state_to_save = {
                        "model": self.model.state_dict(),
                        "optim": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                    }
                    torch.save(state_to_save, checkpoint_path)
                    logger.info(
                        f"Saved non-sharded model checkpoint in '{checkpoint_path}'"
                    )

        if self._banks is not None:
            # Every rank, not just rank 0: the banks are sharded by column and
            # each rank's quarter exists nowhere else. Written after the model
            # so a crash leaves no sidecar claiming to match a checkpoint that
            # was never finished.
            self._banks.save(checkpoint_folder)

        if os.environ["RANK"] == "0":
            save_training_state(
                save_config=self.checkpoint.save,
                step=self.step,
                processed_tokens=self.processed_tokens,
                metric_logger=self.metric_logger,
            )
