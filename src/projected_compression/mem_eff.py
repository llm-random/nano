import os
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Optional
import torch.nn.functional as F
from src.projected_compression.initialization import get_topk_indices
from torch.distributed.tensor import distribute_tensor, DTensor


def get_global_grad_norm(params_or_grads, device=None):
    """Compute true global L2 grad norm for FSDP2 Shard(0) DTensor grads.
    torch.nn.utils.get_total_norm only computes local-shard norms without
    cross-rank reduction, underestimating by ~sqrt(world_size).
    For CPU tensors (plain params, no FSDP2), all_reduce is skipped since
    NCCL doesn't support CPU tensors and all ranks already hold identical values."""
    local_norm_sq = torch.tensor(0.0)
    for g in params_or_grads:
        if isinstance(g, nn.Parameter):
            g = g.grad
        if g is None:
            continue
        local_g = g.to_local() if hasattr(g, "to_local") else g
        if device is None:
            device = local_g.device
            local_norm_sq = local_norm_sq.to(device)
        local_norm_sq += local_g.float().to(local_norm_sq.device).norm(2.0) ** 2
    if dist.is_initialized() and local_norm_sq.device.type != "cpu":
        dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM)
    return local_norm_sq.sqrt()


class MemoryEfficientProjectedCompression(nn.Module):
    # fmt: off
    def __init__(
        self,
        source_model: nn.Module,
        target_model: nn.Module,
        path_to_importances: str,
        cast_bfloat16: bool,
        adjust_grad_norm: bool,
        cpu_offload_projections: bool = False,
        block_gpu_compute: bool = True,
    ):
        super().__init__()
        self.source_model = source_model
        self.target_model = target_model
        self.cast_bfloat16 = cast_bfloat16
        self.adjust_grad_norm = adjust_grad_norm
        self.cpu_offload_projections = cpu_offload_projections
        self.block_gpu_compute = block_gpu_compute
        self.projections = Projections(
            q_heads=target_model.encoder.blocks[0].attention_layer.layer.q_heads,
            kv_heads=target_model.encoder.blocks[0].attention_layer.layer.kv_heads,
            base_dmodel=source_model.encoder.blocks[0].attention_layer.layer.dmodel,
            base_dff=source_model.encoder.blocks[0].ff_layer.layer.ff_pre_act.out_features,
            target_dmodel=target_model.encoder.blocks[0].attention_layer.layer.dmodel,
            target_dff=target_model.encoder.blocks[0].ff_layer.layer.ff_pre_act.out_features,
            n_blocks=len(target_model.encoder.blocks),
            vocab_size=target_model.embedding.num_embeddings,
            path_to_importances=path_to_importances,
            cast_bfloat16=cast_bfloat16,
        )
    # fmt: on

    def forward(self, *args, **kwargs):
        x = self.source_model.embedding(*args, **kwargs)
        x = F.linear(x, self.projections.embedding, bias=None)
        x = x + self.projections.auxiliary_embedding_weights(*args)
        x = self.target_model.encoder(x)
        x = self.target_model.head(x)
        return x

    def _get_source_weight(self, w):
        """Return source weight, optionally cast to bfloat16."""
        return w.bfloat16() if self.cast_bfloat16 else w

    def _copy_projected_weight(self, proj_comp, source_weight, target_weight):
        """Compute projected weight and copy into target_weight.

        When source_weight is on CPU (encoder blocks with cpu_offload):
          block_gpu_compute=True  — all ranks transiently move source+proj to GPU,
            compute independently (weights are replicated → same result on every rank),
            then distribute_tensor shards locally.  No NCCL broadcast needed.
          block_gpu_compute=False — rank 0 computes on CPU, broadcasts result via NCCL.

        Otherwise (head/embedding — always GPU, or normal non-offload path):
          Plain-tensor result from non-FSDP2 proj is distribute_tensor'd into the
          DTensor target.  FSDP2-wrapped proj produces a DTensor result directly."""
        if self.cpu_offload_projections and source_weight.device.type == "cpu":
            # CPU path: rank 0 computes, broadcasts to all ranks.
            rank = dist.get_rank() if dist.is_initialized() else 0
            result_gpu = torch.empty(
                target_weight.shape, device="cuda", dtype=torch.float32
            )
            if rank == 0:
                result_gpu.copy_(
                    proj_comp.get_projected_weight(
                        self._get_source_weight(source_weight)
                    )
                )
            if dist.is_initialized():
                dist.broadcast(result_gpu, src=0)
            target_weight.data.copy_(
                distribute_tensor(
                    result_gpu, target_weight.device_mesh, target_weight.placements
                )
            )
        else:
            result = proj_comp.get_projected_weight(
                self._get_source_weight(source_weight)
            )
            if hasattr(target_weight, "device_mesh") and not hasattr(
                result, "device_mesh"
            ):
                # Plain-tensor result (non-FSDP2 proj) into a DTensor target — distribute first.
                target_weight.data.copy_(
                    distribute_tensor(
                        result, target_weight.device_mesh, target_weight.placements
                    )
                )
            else:
                target_weight.copy_(result)

    def _ensure_cpu_threads(self):
        """Set intra-op thread count once for rank 0 CPU matmuls.
        SLURM sets OMP_NUM_THREADS=cpus_per_gpu which caps BLAS thread pools.
        torch.set_num_threads() only affects PyTorch's own small ops — large matmuls
        dispatch to BLAS (MKL/OpenBLAS) with a separate thread pool.
        We call BLAS's own runtime API via ctypes to override the cap."""
        if hasattr(self, "_cpu_threads_configured"):
            return
        self._cpu_threads_configured = True
        if not dist.is_initialized() or dist.get_rank() == 0:
            # Use SLURM's per-task allocation if available — os.cpu_count() returns
            # all node CPUs (80), but SLURM only binds rank 0 to cpus_per_gpu cores.
            # Setting BLAS to 80 threads on 14 physical cores causes context-switch
            # overhead and hurts throughput vs. using the exact allocation.
            n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 80))
            torch.set_num_threads(n_cpus)
            import ctypes

            for lib, fn in [
                ("libmkl_rt.so", "MKL_Set_Num_Threads"),
                ("libopenblas.so", "openblas_set_num_threads"),
                ("libopenblas.so.0", "openblas_set_num_threads"),
            ]:
                try:
                    getattr(ctypes.CDLL(lib), fn)(ctypes.c_int(n_cpus))
                    break
                except (OSError, AttributeError):
                    pass

    def _block_to_gpu(self, block_source, block_proj):
        """Move source block to GPU for compute.
        Proj block stays on GPU permanently — no move needed."""
        block_source.to("cuda")

    def _block_to_cpu(self, block_source, block_proj):
        """Move source block back to CPU after compute.
        Proj block (params + optimizer state) stays on GPU."""
        block_source.to("cpu")

    @staticmethod
    def _move_block_optimizer_state(optimizer, device):
        """Move all optimizer state tensors (exp_avg, exp_avg_sq, …) to device.
        Called just before optimizer.step() to run Adam on GPU, and just after
        to return state to CPU RAM.  On the first step the state is empty and
        AdamW initialises it on whatever device the params are on at that point,
        so moving to GPU before the first step gives GPU-resident initial state
        which is then moved back to CPU immediately after."""
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p in optimizer.state:
                    state = optimizer.state[p]
                    for k in list(state.keys()):
                        if isinstance(state[k], torch.Tensor):
                            state[k] = state[k].to(device)

    def prepare_compressed_weights(self):
        """
        Copies the projected weights from source_model to target_model using the projections.
        cast_bfloat16: whether to cast the source weights to bfloat16 before projection. This argument only exists to have backward compatibility with previous implementation.
                       after testing, we can remove it and never cast to bfloat16.
        cpu_offload_projections: projection/source weights live on CPU; result is scattered
                                 to the GPU-resident Shard(0) DTensor target weights.
        block_gpu_compute: move each block to GPU for compute (one move per block, not per
                           weight), compute all 7 weights, move back.  All ranks work in
                           parallel — source+proj are replicated CPU tensors, same on every
                           rank, so no NCCL broadcast is needed.
        """
        if self.cpu_offload_projections and not self.block_gpu_compute:
            self._ensure_cpu_threads()
        with torch.no_grad():
            for block_target, block_source, block_proj in zip(
                self.target_model.encoder.blocks,
                self.source_model.encoder.blocks,
                self.projections.blocks,
            ):
                if self.cpu_offload_projections and self.block_gpu_compute:
                    self._block_to_gpu(block_source, block_proj)

                self._copy_projected_weight(
                    block_proj.compressible_q,
                    block_source.attention_layer.layer.q_proj.weight,
                    block_target.attention_layer.layer.q_proj.weight,
                )
                self._copy_projected_weight(
                    block_proj.compressible_k,
                    block_source.attention_layer.layer.k_proj.weight,
                    block_target.attention_layer.layer.k_proj.weight,
                )
                self._copy_projected_weight(
                    block_proj.compressible_v,
                    block_source.attention_layer.layer.v_proj.weight,
                    block_target.attention_layer.layer.v_proj.weight,
                )
                self._copy_projected_weight(
                    block_proj.compressible_o,
                    block_source.attention_layer.layer.o_proj.weight,
                    block_target.attention_layer.layer.o_proj.weight,
                )
                self._copy_projected_weight(
                    block_proj.compressible_ff_pre,
                    block_source.ff_layer.layer.ff_pre_act.weight,
                    block_target.ff_layer.layer.ff_pre_act.weight,
                )
                self._copy_projected_weight(
                    block_proj.compressible_ff_gate,
                    block_source.ff_layer.layer.gate.weight,
                    block_target.ff_layer.layer.gate.weight,
                )
                self._copy_projected_weight(
                    block_proj.compressible_ff_post,
                    block_source.ff_layer.layer.ff_post_act.weight,
                    block_target.ff_layer.layer.ff_post_act.weight,
                )

                if self.cpu_offload_projections and self.block_gpu_compute:
                    # No grads yet (no_grad context), so _block_to_cpu's grad handling is a no-op.
                    block_source.to("cpu")
                    block_proj.to("cpu")

            self._copy_projected_weight(
                self.projections.head,
                self.source_model.head.linear.weight,
                self.target_model.head.linear.weight,
            )

    def _gather_block_grads(self, block_target):
        """Gather the 7 Wc gradient shards for one encoder block via a single all_gather.

        Instead of 7 serial redistribute(Replicate) NCCL all_gathers, we flatten all 7
        local Shard(0) shards into one tensor, call dist.all_gather_into_tensor once,
        then reconstruct each full gradient.  Called once per block (16 calls total),
        keeping only 7 full grad tensors in VRAM at a time instead of all 112 at once.

        Returns list[Tensor] of length 7: [q, k, v, o, ff_pre, ff_gate, ff_post] full grads.
        Zeros target_weight.grad for all 7 weights."""
        weights = [
            block_target.attention_layer.layer.q_proj.weight,
            block_target.attention_layer.layer.k_proj.weight,
            block_target.attention_layer.layer.v_proj.weight,
            block_target.attention_layer.layer.o_proj.weight,
            block_target.ff_layer.layer.ff_pre_act.weight,
            block_target.ff_layer.layer.gate.weight,
            block_target.ff_layer.layer.ff_post_act.weight,
        ]
        shards = [
            w.grad.to_local() for w in weights
        ]  # each: [shape[0]//world_size, ...]
        shard_numel = [s.numel() for s in shards]
        local_cat = torch.cat([s.flatten() for s in shards])
        local_numel = local_cat.numel()

        world_size = dist.get_world_size()
        full = torch.empty(
            world_size * local_numel, device=local_cat.device, dtype=local_cat.dtype
        )
        dist.all_gather_into_tensor(full, local_cat)
        # full layout: [rank0_local_cat | rank1_local_cat | ... | rank(N-1)_local_cat]

        full_grads = []
        for i, (w, shard_size) in enumerate(zip(weights, shard_numel)):
            full_shape = w.shape
            shard_rows = full_shape[0] // world_size
            offset_in_rank = sum(shard_numel[:i])
            parts = []
            for rank in range(world_size):
                start = rank * local_numel + offset_in_rank
                parts.append(
                    full[start : start + shard_size].reshape(
                        [shard_rows] + list(full_shape[1:])
                    )
                )
            full_grads.append(torch.cat(parts, dim=0))
            w.grad = None  # consumed

        return full_grads

    def pass_gradient_to_projections(
        self, optimizers: List, schedulers, gradient_clipping, shared_gradient_norms
    ):

        def get_module_grad_norm(module: nn.Module):
            return get_global_grad_norm(
                p.grad for p in module.parameters() if p.grad is not None
            )

        def backward_block(
            block_proj, block_source, block_target, precomputed_grads=None
        ):
            triplets = [
                (
                    block_proj.compressible_q,
                    block_source.attention_layer.layer.q_proj.weight,
                    block_target.attention_layer.layer.q_proj.weight,
                ),
                (
                    block_proj.compressible_k,
                    block_source.attention_layer.layer.k_proj.weight,
                    block_target.attention_layer.layer.k_proj.weight,
                ),
                (
                    block_proj.compressible_v,
                    block_source.attention_layer.layer.v_proj.weight,
                    block_target.attention_layer.layer.v_proj.weight,
                ),
                (
                    block_proj.compressible_o,
                    block_source.attention_layer.layer.o_proj.weight,
                    block_target.attention_layer.layer.o_proj.weight,
                ),
                (
                    block_proj.compressible_ff_pre,
                    block_source.ff_layer.layer.ff_pre_act.weight,
                    block_target.ff_layer.layer.ff_pre_act.weight,
                ),
                (
                    block_proj.compressible_ff_gate,
                    block_source.ff_layer.layer.gate.weight,
                    block_target.ff_layer.layer.gate.weight,
                ),
                (
                    block_proj.compressible_ff_post,
                    block_source.ff_layer.layer.ff_post_act.weight,
                    block_target.ff_layer.layer.ff_post_act.weight,
                ),
            ]
            for j, (bproj, bsrc, btgt) in enumerate(triplets):
                wc_grad = (
                    precomputed_grads[j] if precomputed_grads is not None else None
                )
                self.backward_compressed_weights(bproj, bsrc, btgt, wc_grad)

        self.backward_compressed_weights(
            self.projections.head,
            self.source_model.head.linear.weight,
            self.target_model.head.linear.weight,
        )
        # DTensor handles all-reduce for head projection_in_weight grad automatically
        # (Shard(0)@Shard(0) matmul backward does an all_reduce). No explicit all_reduce needed.

        use_block_gpu = self.cpu_offload_projections and self.block_gpu_compute
        # Whether to use per-block all_gather batching (replaces 7 serial redistributes
        # per block with 1 all_gather; 16 total calls vs 112, minimal extra VRAM).
        use_batched_gather = (
            self.cpu_offload_projections
            and dist.is_initialized()
            and not shared_gradient_norms
        )

        if optimizers is None:
            for block_target, block_source, block_proj in zip(
                self.target_model.encoder.blocks,
                self.source_model.encoder.blocks,
                self.projections.blocks,
            ):
                block_grads = (
                    self._gather_block_grads(block_target)
                    if use_batched_gather
                    else None
                )
                if use_block_gpu:
                    self._block_to_gpu(block_source, block_proj)
                backward_block(block_proj, block_source, block_target, block_grads)
                if use_block_gpu:
                    self._block_to_cpu(block_source, block_proj)

            final_grad_norm = get_global_grad_norm(
                v.grad for v in self.parameters() if v.grad is not None
            )
            return final_grad_norm, [], None, None
        else:
            projection_blocks_grad_norms = []
            start_grad_norm = get_global_grad_norm(
                v.grad for v in self.projections.parameters() if v.grad is not None
            )

            if shared_gradient_norms:
                # Two-pass global clipping: equivalent to old single-optimizer PC.
                # Pass 1: compute all block norms without updating.
                for block_target, block_source, block_proj in zip(
                    self.target_model.encoder.blocks,
                    self.source_model.encoder.blocks,
                    self.projections.blocks,
                ):
                    # Save Wc grads before backward_block consumes them.
                    wc_grads = {id(p): p.grad for p in block_target.parameters()}
                    if use_block_gpu:
                        self._block_to_gpu(block_source, block_proj)
                    backward_block(block_proj, block_source, block_target)
                    if use_block_gpu:
                        self._block_to_cpu(block_source, block_proj)
                    projection_blocks_grad_norms.append(
                        get_module_grad_norm(block_proj)
                    )
                    for p in block_proj.parameters():
                        p.grad = None
                    # Restore Wc grads for pass 2.
                    for p in block_target.parameters():
                        p.grad = wc_grads[id(p)]

                # Include target_model norm layer gradients so the total matches
                # old PC's clip_gradient() which covers all trainable params.
                norm_layer_grads = []
                for block in self.target_model.encoder.blocks:
                    if getattr(block.attention_layer, "norm", None) is not None:
                        norm_layer_grads.extend(
                            [
                                p.grad
                                for p in block.attention_layer.norm.parameters()
                                if p.grad is not None
                            ]
                        )
                    if getattr(block.ff_layer, "norm", None) is not None:
                        norm_layer_grads.extend(
                            [
                                p.grad
                                for p in block.ff_layer.norm.parameters()
                                if p.grad is not None
                            ]
                        )
                if getattr(self.target_model.head, "norm", None) is not None:
                    norm_layer_grads.extend(
                        [
                            p.grad
                            for p in self.target_model.head.norm.parameters()
                            if p.grad is not None
                        ]
                    )
                target_norm_layer_norm = get_global_grad_norm(norm_layer_grads)

                # True global pre-clip norm across all trainable params.
                global_norm = torch.tensor(
                    [start_grad_norm.item(), target_norm_layer_norm.item()]
                    + [n.item() for n in projection_blocks_grad_norms]
                ).norm()

                # Pass 2: re-backward, clip with global norm, update per block.
                for block_target, block_source, block_proj, optimizer, scheduler in zip(
                    self.target_model.encoder.blocks,
                    self.source_model.encoder.blocks,
                    self.projections.blocks,
                    optimizers,
                    schedulers,
                ):
                    if use_block_gpu:
                        self._block_to_gpu(block_source, block_proj)
                    backward_block(block_proj, block_source, block_target)
                    if use_block_gpu:
                        self._block_to_cpu(block_source, block_proj)
                    if gradient_clipping:
                        torch.nn.utils.clip_grads_with_norm_(
                            block_proj.parameters(), gradient_clipping, global_norm
                        )
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                final_grad_norm = global_norm
            else:
                # Per-block independent clipping.
                for block_target, block_source, block_proj, optimizer, scheduler in zip(
                    self.target_model.encoder.blocks,
                    self.source_model.encoder.blocks,
                    self.projections.blocks,
                    optimizers,
                    schedulers,
                ):
                    block_grads = (
                        self._gather_block_grads(block_target)
                        if use_batched_gather
                        else None
                    )
                    if use_block_gpu:
                        self._block_to_gpu(block_source, block_proj)
                    backward_block(block_proj, block_source, block_target, block_grads)
                    block_norm = get_module_grad_norm(block_proj)
                    projection_blocks_grad_norms.append(block_norm)
                    if gradient_clipping:
                        torch.nn.utils.clip_grads_with_norm_(
                            list(block_proj.parameters()), gradient_clipping, block_norm
                        )
                    # Proj params + optimizer state are on GPU — step is a pure GPU op.
                    optimizer.step()
                    optimizer.zero_grad()
                    if use_block_gpu:
                        self._block_to_cpu(block_source, block_proj)
                    scheduler.step()

                # Clip head and embedding projections independently.
                head_params = [
                    p for p in self.projections.head.parameters() if p.grad is not None
                ]
                head_norm = get_global_grad_norm(p.grad for p in head_params)
                if gradient_clipping:
                    torch.nn.utils.clip_grads_with_norm_(
                        head_params, gradient_clipping, head_norm
                    )

                embedding_params = (
                    [self.projections.embedding]
                    if self.projections.embedding.grad is not None
                    else []
                ) + [
                    p
                    for p in self.projections.auxiliary_embedding_weights.parameters()
                    if p.grad is not None
                ]
                embedding_norm = get_global_grad_norm(p.grad for p in embedding_params)
                if gradient_clipping:
                    torch.nn.utils.clip_grads_with_norm_(
                        embedding_params, gradient_clipping, embedding_norm
                    )

                final_grad_norm = torch.tensor(
                    [start_grad_norm.item()]
                    + [n.item() for n in projection_blocks_grad_norms]
                ).norm()

                return (
                    final_grad_norm,
                    projection_blocks_grad_norms,
                    head_norm,
                    embedding_norm,
                )

        return final_grad_norm, projection_blocks_grad_norms, None, None

    def _backward_embedding_cpu(self):
        """Compute CPU grads for projections.embedding and auxiliary_embedding_weights
        from the GPU combined-embedding gradient.

        _combined_embedding is a plain GPU tensor (not a DTensor), so FSDP2 does NOT
        all_reduce its grad.  We must do it manually before computing projection grads,
        otherwise each rank's grad only reflects its own data shard and params diverge.

        After all_reduce, rank 0 computes projection grads and broadcasts the small
        result (proj_emb_grad, ~7.5 MB).  The large aux_emb_grad (0.5 GB) is moved
        CPU-locally on each rank — no broadcast needed since it's the same everywhere
        after all_reduce."""
        if not hasattr(self, "_combined_embedding") or self._combined_embedding is None:
            return
        if self._combined_embedding.grad is None:
            return

        # Correctness: all_reduce so all ranks see the globally accumulated grad.
        if dist.is_initialized():
            dist.all_reduce(self._combined_embedding.grad, op=dist.ReduceOp.SUM)

        rank = dist.get_rank() if dist.is_initialized() else 0

        # aux_emb_grad = emb_grad  (same on all ranks after all_reduce, each rank copies locally)
        aux_emb_grad_cpu = self._combined_embedding.grad.cpu()  # [vocab, target_dmodel]

        # proj_emb_grad: rank 0 computes (big matmul), result broadcast to all (~7.5 MB).
        proj_emb_grad_gpu = torch.empty(
            self.projections.embedding.shape, device="cuda", dtype=torch.float32
        )
        if rank == 0:
            source_emb = (
                self.source_model.embedding.weight.detach()
            )  # [vocab, base_dmodel] CPU
            # d(loss)/d(P_emb) = emb_grad.T @ source_emb    shape [target_dmodel, base_dmodel]
            proj_emb_grad_gpu.copy_(aux_emb_grad_cpu.T @ source_emb)
        if dist.is_initialized():
            dist.broadcast(proj_emb_grad_gpu, src=0)

        self.projections.embedding.grad = proj_emb_grad_gpu.cpu()
        self.projections.auxiliary_embedding_weights.weight.grad = aux_emb_grad_cpu
        self._combined_embedding.grad = None

    def backward_compressed_weights(
        self, proj, source_weight, target_weight, wc_grad=None
    ):
        """Compute gradients for projection parameters.

        wc_grad: optional pre-gathered full plain GPU gradient tensor (from
            _batch_gather_block_grads).  When provided, skips per-weight
            redistribute(Replicate) NCCL all_gather entirely.  When None,
            gathers from target_weight.grad as before."""
        if (
            self.cpu_offload_projections
            and source_weight.device.type == "cpu"
            and hasattr(target_weight.grad, "to_local")
        ):
            # CPU fallback path (block_gpu_compute=False): rank 0 computes, broadcasts grads.
            from torch.distributed.tensor import Replicate

            wc_grad_gpu = target_weight.grad.redistribute(
                placements=[Replicate()]
            ).to_local()
            target_weight.grad = None

            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                src_w = source_weight.detach()
                if self.cast_bfloat16:
                    src_w = src_w.bfloat16()
                weights = proj.get_projected_weight(src_w)
                weights.backward(wc_grad_gpu.cpu())
            for p in proj.parameters():
                grad_buf = torch.empty(p.shape, device="cuda", dtype=p.dtype)
                if rank == 0 and p.grad is not None:
                    grad_buf.copy_(p.grad)
                    p.grad = None
                if dist.is_initialized():
                    dist.broadcast(grad_buf, src=0)
                p.grad = grad_buf.cpu()
        else:
            # GPU path: source_weight is on GPU (native or moved by block_gpu_compute).
            # proj params may be plain tensors (non-FSDP2 cpu_offload) or DTensors.
            source_weight = source_weight.detach()
            if self.cast_bfloat16:
                source_weight = source_weight.bfloat16()
            weights = proj.get_projected_weight(source_weight)

            if wc_grad is None:
                # No pre-gathered grad: gather from target_weight.grad.
                wc_grad = target_weight.grad
                if hasattr(wc_grad, "to_local") and not hasattr(weights, "device_mesh"):
                    # DTensor grad but plain-tensor result (non-FSDP2 proj on GPU, e.g. head).
                    from torch.distributed.tensor import Replicate

                    wc_grad = wc_grad.redistribute(placements=[Replicate()]).to_local()

            target_weight.grad = None
            weights.backward(wc_grad)
            # DTensor autograd outside FSDP2's context leaves gradients as Partial(sum).
            # Redistribute to Shard(0) to match FSDP2's reduce-scatter behavior.
            for p in proj.parameters():
                if p.grad is not None and hasattr(p.grad, "redistribute"):
                    p.grad = p.grad.redistribute(placements=p.placements)


class CompressibleLinear(nn.Module):
    def __init__(
        self,
        base_in_features: int,
        result_in_features: int,
        base_out_features: int,
        result_out_features: int,
        proj_in_topk_indices: Optional[torch.Tensor],
        proj_out_topk_indices: Optional[torch.Tensor],
        cast_bfloat16: bool,
    ):
        super().__init__()
        self.cast_bfloat16 = cast_bfloat16
        self.base_in_features = base_in_features
        self.result_in_features = result_in_features
        self.base_out_features = base_out_features
        self.result_out_features = result_out_features
        self.proj_in_topk_indices = proj_in_topk_indices
        self.proj_out_topk_indices = proj_out_topk_indices

        if self.base_in_features != self.result_in_features:
            assert (
                self.proj_in_topk_indices is not None
            ), "proj_in_topk_indices must be provided if result_in_features is specified."
            weight = torch.zeros(self.base_in_features, self.result_in_features)
            self.projection_in_weight = nn.Parameter(weight, requires_grad=True)

        if self.base_out_features != self.result_out_features:
            assert (
                self.proj_out_topk_indices is not None
            ), "proj_out_topk_indices must be provided if result_out_features is specified."

            weight = torch.zeros(self.result_out_features, self.base_out_features)
            self.projection_out_weight = nn.Parameter(weight, requires_grad=True)

        if self.result_in_features is not None or self.result_out_features is not None:
            final_in_features = (
                self.result_in_features
                if self.result_in_features is not None
                else self.base_in_features
            )
            final_out_features = (
                self.result_out_features
                if self.result_out_features is not None
                else self.base_out_features
            )
            weight = torch.zeros(final_out_features, final_in_features)
            self.auxiliary_weight = nn.Parameter(weight, requires_grad=True)

    def init_projection_weights(self, proj_in_topk_indices, proj_out_topk_indices):
        with torch.no_grad():
            if hasattr(self, "projection_in_weight"):
                weight = torch.zeros(self.projection_in_weight.data.shape)
                weight[proj_in_topk_indices, torch.arange(self.result_in_features)] = 1

                if isinstance(self.projection_in_weight, DTensor):
                    self.projection_in_weight.data.copy_(
                        distribute_tensor(
                            weight,
                            self.projection_in_weight.device_mesh,
                            self.projection_in_weight.placements,
                        )
                    )
                else:
                    self.projection_in_weight.data.copy_(weight)

            if hasattr(self, "projection_out_weight"):
                weight = torch.zeros(self.projection_out_weight.data.shape)
                weight[
                    torch.arange(self.result_out_features), proj_out_topk_indices
                ] = 1

                if isinstance(self.projection_out_weight, DTensor):
                    self.projection_out_weight.data.copy_(
                        distribute_tensor(
                            weight,
                            self.projection_out_weight.device_mesh,
                            self.projection_out_weight.placements,
                        )
                    )
                else:
                    self.projection_out_weight.data.copy_(weight)

            if hasattr(self, "auxiliary_weight"):
                self.auxiliary_weight.data.copy_(
                    torch.zeros_like(self.auxiliary_weight.data)
                )

    def get_projected_weight(self, source_weight):
        if not self.cast_bfloat16:
            weight = source_weight
            if hasattr(self, "projection_in_weight"):
                weight = weight @ self.projection_in_weight
            if hasattr(self, "projection_out_weight"):
                weight = self.projection_out_weight @ weight
            if hasattr(self, "auxiliary_weight"):
                weight = weight + self.auxiliary_weight
            return weight
        else:
            weight = source_weight
            if hasattr(self, "projection_in_weight"):
                weight = weight @ self.projection_in_weight.bfloat16()
            if hasattr(self, "projection_out_weight"):
                weight = self.projection_out_weight.bfloat16() @ weight
            if hasattr(self, "auxiliary_weight"):
                weight = weight + self.auxiliary_weight.bfloat16()
            return weight.float()

    def extra_repr(self) -> str:
        if hasattr(self, "projection_in_weight"):
            in_features, out_features = self.projection_in_weight.shape
            result = f"(projection_in_weight) ({in_features}, {out_features})\n"
        else:
            result = ""
        in_features, out_features = self.W.shape
        result += f"(weight) ({in_features}, {out_features})"
        if hasattr(self, "projection_ouweight_weight"):
            out_features, in_features = self.projection_out_weight.shape
            result += f"\n(projection_out_weight) ({out_features}, {in_features})"
        return result


class CompressibleBlock(nn.Module):
    def __init__(
        self,
        q_heads: int,
        kv_heads: int,
        dhead: int,
        base_dmodel: int,
        target_dmodel: int,
        base_dff: int,
        target_dff: int,
        dmodel_topk_indices: Optional[torch.Tensor],
        dff_topk_indices: Optional[torch.Tensor],
        cast_bfloat16: bool,
    ):
        super().__init__()

        self.compressible_q = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=target_dmodel,
            base_out_features=q_heads * dhead,
            result_out_features=q_heads * dhead,
            proj_in_topk_indices=dmodel_topk_indices,
            proj_out_topk_indices=dmodel_topk_indices,
            cast_bfloat16=cast_bfloat16,
        )
        self.compressible_k = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=target_dmodel,
            base_out_features=kv_heads * dhead,
            result_out_features=kv_heads * dhead,
            proj_in_topk_indices=dmodel_topk_indices,
            proj_out_topk_indices=None,
            cast_bfloat16=cast_bfloat16,
        )
        self.compressible_v = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=target_dmodel,
            base_out_features=kv_heads * dhead,
            result_out_features=kv_heads * dhead,
            proj_in_topk_indices=dmodel_topk_indices,
            proj_out_topk_indices=None,
            cast_bfloat16=cast_bfloat16,
        )
        self.compressible_o = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=base_dmodel,
            base_out_features=base_dmodel,
            result_out_features=target_dmodel,
            proj_in_topk_indices=None,
            proj_out_topk_indices=dmodel_topk_indices,
            cast_bfloat16=cast_bfloat16,
        )
        self.compressible_ff_pre = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=target_dmodel,
            base_out_features=base_dff,
            result_out_features=target_dff,
            proj_in_topk_indices=dmodel_topk_indices,
            proj_out_topk_indices=dff_topk_indices,
            cast_bfloat16=cast_bfloat16,
        )
        self.compressible_ff_gate = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=target_dmodel,
            base_out_features=base_dff,
            result_out_features=target_dff,
            proj_in_topk_indices=dmodel_topk_indices,
            proj_out_topk_indices=dff_topk_indices,
            cast_bfloat16=cast_bfloat16,
        )
        self.compressible_ff_post = CompressibleLinear(
            base_in_features=base_dff,
            result_in_features=target_dff,
            base_out_features=base_dmodel,
            result_out_features=target_dmodel,
            proj_in_topk_indices=dff_topk_indices,
            proj_out_topk_indices=dmodel_topk_indices,
            cast_bfloat16=cast_bfloat16,
        )

    def init_projection_weights(self, dmodel_topk_indices, dff_topk_indices):
        self.compressible_q.init_projection_weights(
            dmodel_topk_indices, dmodel_topk_indices
        )
        self.compressible_k.init_projection_weights(dmodel_topk_indices, None)
        self.compressible_v.init_projection_weights(dmodel_topk_indices, None)
        self.compressible_o.init_projection_weights(None, dmodel_topk_indices)
        self.compressible_ff_pre.init_projection_weights(
            dmodel_topk_indices, dff_topk_indices
        )
        self.compressible_ff_gate.init_projection_weights(
            dmodel_topk_indices, dff_topk_indices
        )
        self.compressible_ff_post.init_projection_weights(
            dff_topk_indices, dmodel_topk_indices
        )


class Projections(nn.Module):
    def __init__(
        self,
        q_heads: int,
        kv_heads: int,
        base_dmodel: int,
        base_dff: int,
        target_dmodel: int,
        target_dff: int,
        n_blocks: int,
        vocab_size: int,
        path_to_importances: str,
        cast_bfloat16: bool,
    ):
        dmodel_topk_indices, dff_topk_indices = get_topk_indices(
            path_to_importances, target_dmodel, target_dff
        )

        super().__init__()
        self.target_dmodel = target_dmodel
        self.target_dff = target_dff

        # --- Embedding start
        weight = torch.zeros(target_dmodel, base_dmodel)
        weight[torch.arange(target_dmodel), dmodel_topk_indices] = 1
        self.embedding = nn.Parameter(weight, requires_grad=True)

        zeros = torch.zeros(vocab_size, target_dmodel)
        self.auxiliary_embedding_weights = nn.Embedding(
            vocab_size, target_dmodel, _weight=zeros
        )
        # --- Embedding end

        self.blocks = nn.ModuleList(
            [
                CompressibleBlock(
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    dhead=base_dmodel // q_heads,
                    base_dmodel=base_dmodel,
                    target_dmodel=target_dmodel,
                    base_dff=base_dff,
                    target_dff=target_dff,
                    dmodel_topk_indices=dmodel_topk_indices,
                    dff_topk_indices=dff_topk_indices[i],
                    cast_bfloat16=cast_bfloat16,
                )
                for i in range(n_blocks)
            ]
        )
        self.head = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=target_dmodel,
            base_out_features=vocab_size,
            result_out_features=vocab_size,
            proj_in_topk_indices=dmodel_topk_indices,
            proj_out_topk_indices=None,
            cast_bfloat16=cast_bfloat16,
        )

    def init_projection_weights(self, path_to_importances):
        dmodel_topk_indices, dff_topk_indices = get_topk_indices(
            path_to_importances, self.target_dmodel, self.target_dff
        )

        with torch.no_grad():
            weight = torch.zeros(self.embedding.data.shape)
            weight[torch.arange(self.target_dmodel), dmodel_topk_indices] = 1

            if isinstance(self.embedding, DTensor):
                self.embedding.data.copy_(
                    distribute_tensor(
                        weight,
                        self.embedding.device_mesh,
                        self.embedding.placements,
                    )
                )
            else:
                self.embedding.data.copy_(weight)

            self.auxiliary_embedding_weights.weight.data.copy_(
                torch.zeros_like(self.auxiliary_embedding_weights.weight.data)
            )

            for block, block_dff_topk_indices in zip(self.blocks, dff_topk_indices):
                block.init_projection_weights(
                    dmodel_topk_indices, block_dff_topk_indices
                )

            self.head.init_projection_weights(self.head.proj_in_topk_indices, None)
