"""SM120 multi-rank AutoCP ablation: baseline / aggregate / comm / both.

Single GPU (ordered rank emulation, NOT a communication benchmark):
  .venv/bin/python -m benchmark.demo_dual_cp --local-check --world-size 4 --heads 4
Real GPUs (world size comes from torchrun; checks all modes, then times selected modes):
  .venv/bin/torchrun --standalone --nproc_per_node=2 -m benchmark.demo_dual_cp --mode all

Packed sequences: --seq-lengths 521 7162 509 (total 8192 tokens).
BF16 inputs/M, FP32 gradients/seeds, K=V=128, KxV layout.
Contexts use build_cp_context with AutoCP enabled, without forced partitions.
Hooks exist only inside each demo process's scoped patch context; no library
default dispatch changes. comm caches forward remote M and all-gathers only dh
in backward. Local/card M computation is retained in all modes.
"""

import argparse
from contextlib import contextmanager
from bisect import bisect_right
from datetime import timedelta
from functools import partial
from itertools import accumulate
import os
from statistics import median
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F

# Select the device before importing the architecture-dependent FlashQLA modules.
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
from flash_qla.ops.gated_delta_rule import chunk
from flash_qla.ops.gated_delta_rule.chunk.cp import preprocess
from flash_qla.ops.gated_delta_rule.chunk.cp import build_cp_context
from flash_qla.ops.gated_delta_rule.chunk.blackwell_sm120 import cp_comm
from flash_qla.ops.gated_delta_rule.chunk.blackwell_sm120.cp_fwd import tilelang_aggregate_dh_ws


# (use WS aggregation, reuse forward remote M in backward)
MODES = {
    "baseline": (False, False),
    "aggregate": (True, False),
    "comm": (False, True),
    "both": (True, True),
}
OUTPUT_NAMES = ("o", "ht", "dq", "dk", "dv", "db", "dg", "dh0")
ERROR_TOLERANCE = 0.02
ORIGINAL_AGGREGATE = chunk.aggregate_card_state
ORIGINAL_PREPROCESS_FWD = chunk.cp_preprocess_fwd


def aggregate_ws(ht_buffer, mt_buffer, fallback_mask, seq_map_r2c, raw_h0=None,
                 state_v_first=False, reverse=False, transpose_m=False, compute_m=True):
    if not reverse:
        return ORIGINAL_AGGREGATE(ht_buffer, mt_buffer, fallback_mask, seq_map_r2c,
                                  raw_h0=raw_h0, state_v_first=state_v_first,
                                  reverse=reverse, transpose_m=transpose_m, compute_m=compute_m)
    assert transpose_m and not compute_m and not state_v_first
    assert ht_buffer.dtype == torch.float32 and mt_buffer.dtype == torch.bfloat16
    shape = (len(seq_map_r2c) - 1, ht_buffer.shape[1], 128, 128)
    seed = raw_h0 if raw_h0 is not None else ht_buffer.new_zeros(shape)
    out = torch.empty_like(seed)
    tilelang_aggregate_dh_ws(ht_buffer.shape[1])(
        seed, ht_buffer, mt_buffer, fallback_mask, seq_map_r2c, out)
    return out, None


@contextmanager
def mode_hooks(mode):
    """Temporary process-wide overrides for this standalone, single-threaded demo."""
    use_aggregate, use_comm = MODES[mode]
    with patch.object(chunk, "aggregate_card_state", aggregate_ws if use_aggregate else ORIGINAL_AGGREGATE), \
         patch.object(chunk, "cp_preprocess_fwd", partial(
             ORIGINAL_PREPROCESS_FWD, cache_remote_m=use_comm)):
        yield


def make_context(args, group):
    cu = torch.tensor([0, *accumulate(args.seq_lengths)], device="cuda", dtype=torch.int32)
    return build_cp_context(
        cu, enable_inter=True, enable_intra=True, group=group,
        num_v_heads=args.heads, chunk_size=chunk.CHUNK_SIZE, is_train=True)


def make_inputs(args, ctx, rank):
    torch.manual_seed(123 + rank)
    shape = (1, args.tokens // args.world_size, args.heads, 128)
    q, k = [F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()
            for _ in range(2)]
    v, do = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    g = -torch.rand(shape[:-1], device="cuda") * args.gate_scale
    beta = torch.rand(shape[:-1], device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(321)  # Identical global boundary seeds on both ranks.
    first = bisect_right([0, *accumulate(args.seq_lengths)], rank * shape[1]) - 1
    h0, dht = [
        (torch.randn((len(args.seq_lengths), args.heads, 128, 128), device="cuda") * 0.1)
        [first:first + ctx.num_seqs].clone() for _ in range(2)
    ]
    if not ctx.is_first_rank:
        h0[0] = 0
    if not ctx.is_last_rank:
        dht[-1] = 0
    return dict(q=q, k=k, v=v, g=g, beta=beta, do=do, h0=h0, dht=dht)


def forward(data, ctx, seeded):
    return chunk.chunk_gated_delta_rule_fwd(
        data["q"], data["k"], data["v"], data["g"], data["beta"],
        initial_state=data["h0"] if seeded else None, output_final_state=True,
        enable_fwd_cp_cache=True, cp_context=ctx)


def backward(data, ctx, fw, seeded):
    return chunk.chunk_gated_delta_rule_bwd(
        data["q"], data["k"], data["v"], fw[0], data["beta"], fw[1], data["do"],
        initial_state=data["h0"] if seeded else None,
        dht=data["dht"] if seeded else None, cp_cache=fw[5], cp_context=ctx)


def collect_outputs(fw, bw):
    return dict(zip(OUTPUT_NAMES, (fw[2], fw[4], *bw)))


def step(data, ctx, seeded):
    fw = forward(data, ctx, seeded)
    bw = backward(data, ctx, fw, seeded)
    return collect_outputs(fw, bw)


def check_outputs(actual, expected, label, group=None):
    """Compare outputs outside timing; treat a NaN error as a failed comparison."""
    errors = {}
    for name, value in actual.items():
        ref = expected[name]
        assert (value is None) == (ref is None), f"{label}: {name} presence differs from baseline"
        if value is not None:
            error = (value.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-6)
            errors[name] = error
    local_errors = torch.stack(list(errors.values()))
    # Tensor max propagates NaN; convert it before Python/collective MAX can hide it.
    error = local_errors.max().nan_to_num(nan=float("inf"), posinf=float("inf"))
    if group is not None:
        # All ranks participate before asserting, so a peer also sees the failure.
        dist.all_reduce(error, op=dist.ReduceOp.MAX, group=group)
    relative = error.item()
    assert relative < ERROR_TOLERANCE, (
        f"{label}: max rel error={relative:.3e}; local output errors="
        f"{dict(zip(errors, local_errors.tolist()))}")
    return relative


def local_check(args):
    """Real AutoCP/kernels; emulate all-gather with a forward summary collection pass."""
    wire = {}
    rank = 0

    def gather(inp, out=None, group=None, async_op=False):
        wire[rank] = inp.clone()
        received = inp.new_zeros((args.world_size, *inp.shape))
        for source, value in wire.items():
            received[source].copy_(value)
        return received, None

    with patch.object(dist, "get_rank", lambda group=None: rank), \
         patch.object(dist, "get_world_size", lambda group=None: args.world_size), \
         patch.object(preprocess, "all_gather_into_tensor", gather), \
         patch.object(cp_comm, "all_gather_into_tensor", gather):
        contexts, inputs = [], []
        for rank in range(args.world_size):
            # Distinct group identities keep rank-local context cache entries apart.
            ctx = make_context(args, object())
            contexts.append(ctx)
            inputs.append(make_inputs(args, ctx, rank))
            parts = len(ctx.intra_cp_cu_seqlens) - 1 if ctx.is_intra else ctx.num_seqs
            print(f"rank={rank}: AutoCP intra={ctx.is_intra}, sequences={ctx.num_seqs}, parts={parts}")
        for seeded in (False, True):
            # Forward correction needs preceding ranks, so ascending execution is
            # sufficient to collect every independent outgoing (h, M) summary.
            # Replay with all summaries present before copying future M into cache.
            wire.clear()
            with mode_hooks("baseline"):
                for rank in range(args.world_size):
                    forward(inputs[rank], contexts[rank], seeded)
            forward_wire = dict(wire)
            reference = None
            for mode in MODES:
                fw, results = {}, {}
                with mode_hooks(mode):
                    wire = dict(forward_wire)
                    for rank in range(args.world_size):
                        fw[rank] = forward(inputs[rank], contexts[rank], seeded)
                    wire.clear()
                    for rank in reversed(range(args.world_size)):
                        bw = backward(inputs[rank], contexts[rank], fw[rank], seeded)
                        results[rank] = collect_outputs(fw[rank], bw)
                if reference is None:
                    reference = results
                error = max(check_outputs(
                    results[r], reference[r], f"{mode}, seeded={seeded}, rank={r}")
                    for r in range(args.world_size))
                payload = wire[0].numel() * wire[0].element_size()
                print(f"local-check {mode:9} seeded={seeded}: max rel error={error:.3e}, "
                      f"backward payload/rank={payload} bytes", flush=True)
    print("PASS: one-GPU AutoCP equivalence only; real NCCL and multi-GPU timing NOT tested.")


def distributed_check_and_bench(args):
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    rank, group = dist.get_rank(), dist.group.WORLD
    ctx = make_context(args, group)
    data = make_inputs(args, ctx, rank)
    parts = len(ctx.intra_cp_cu_seqlens) - 1 if ctx.is_intra else ctx.num_seqs
    print(f"rank={rank}: AutoCP intra={ctx.is_intra}, sequences={ctx.num_seqs}, parts={parts}", flush=True)
    dist.barrier()
    for seeded in (False, True):
        reference = None
        for mode in MODES:
            with mode_hooks(mode):
                result = step(data, ctx, seeded)
            if reference is None:
                reference = result
            error = check_outputs(result, reference, f"{mode}, seeded={seeded}, rank={rank}", group)
            if rank == 0:
                print(f"{args.world_size}-GPU check {mode:9} seeded={seeded}: max rel error={error:.3e}", flush=True)

    del reference, result
    modes = list(MODES) if args.mode == "all" else [args.mode]
    for mode in modes:
        with mode_hooks(mode):
            for _ in range(args.warmup):
                step(data, ctx, True)
    torch.cuda.synchronize()
    samples = {mode: [] for mode in modes}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for i in range(args.repeats):
        order = modes[i % len(modes):] + modes[:i % len(modes)]
        for mode in order:
            dist.barrier()
            torch.cuda.synchronize()
            with mode_hooks(mode):
                start.record()
                result = step(data, ctx, True)
                end.record()
            end.synchronize()
            elapsed = torch.tensor(start.elapsed_time(end), device="cuda")
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            samples[mode].append(elapsed.item())
    if rank == 0:
        print("Training step (fwd+bwd), nonzero h0/dht; median of worst-rank CUDA-event times:")
        baseline = median(samples["baseline"]) if "baseline" in samples else None
        for mode in modes:
            elapsed = median(samples[mode])
            speedup = f"  {baseline / elapsed:.3f}x baseline" if baseline is not None else ""
            print(f"{mode:9} {elapsed:.3f} ms{speedup}  "
                  f"[{min(samples[mode]):.3f}, {max(samples[mode]):.3f}]", flush=True)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[*MODES, "all"], default="all")
    parser.add_argument("--local-check", action="store_true")
    parser.add_argument("--world-size", type=int, default=4, help="emulated ranks for --local-check")
    parser.add_argument("--tokens", type=int, default=8192, help="global sequence length")
    parser.add_argument("--seq-lengths", type=int, nargs="+", help="packed global sequence lengths; overrides --tokens")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--gate-scale", type=float, default=0.05,
                        help="negative gate magnitude; 0 stresses long-lived boundary states")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if not args.local_check:
        args.world_size = int(os.environ.get("WORLD_SIZE", "0"))
    args.seq_lengths = args.seq_lengths or [args.tokens]
    args.tokens = sum(args.seq_lengths)
    assert torch.cuda.get_device_capability() == (12, 0), "SM120 only"
    assert args.world_size >= 2, "Use torchrun with >=2 GPUs or --local-check"
    assert min(args.seq_lengths) > 0 and args.tokens % args.world_size == 0 and args.heads > 0
    assert args.warmup > 0 and args.repeats > 0
    assert args.gate_scale >= 0
    print(f"GPU={torch.cuda.get_device_name()}, tokens={args.tokens}, heads={args.heads}, "
          f"ranks={args.world_size}, sequences={args.seq_lengths}, gate_scale={args.gate_scale}", flush=True)
    if args.local_check:
        local_check(args)
    else:
        distributed_check_and_bench(args)


if __name__ == "__main__":
    main()
