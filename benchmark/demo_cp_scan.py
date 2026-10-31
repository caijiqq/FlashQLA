"""Single-GPU SM120 reverse aggregation demo; no distributed process group needed.

Run from the repository root:
    .venv/bin/python -m benchmark.demo_cp_scan

Compare tile=32/64 with a Hopper-style TMA/warp-specialized reverse scan.
Outputs and kernels are allocated before timing. CUDA graphs measure GPU work
without Python launch gaps. This is not an end-to-end Dual-CP test.
"""

import argparse
from functools import partial
from statistics import median

import torch

from flash_qla.ops.gated_delta_rule.chunk.blackwell_sm120.cp_fwd import (
    tilelang_aggregate_dh_ws,
    tilelang_correct_h0,
)


def reference(dh, m, fallback, seed):
    """One sequence, KxV layout; reproduce the BF16 operand rounding."""
    state = seed[0].float()
    for i in range(len(dh) - 1, -1, -1):
        updated = dh[i] + m[i].float().transpose(-1, -2) @ state.to(m.dtype).float()
        state = torch.where(fallback[i, :, None, None], updated, dh[i])
    return state.unsqueeze(0)


def check(actual, expected, label):
    error = (actual.float() - expected.float()).abs().max().item()
    relative = error / max(expected.float().abs().max().item(), 1e-6)
    assert relative < 0.02, f"{label}: relative max error {relative:.3e} >= 0.02"
    return relative


def bench_us(jobs, repeats):
    """Rotate/reverse measurement order to reduce clock and temperature bias."""
    graphs = []
    for _, fn, _ in jobs:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                fn()
        graphs.append(graph)
    for _ in range(5):
        for graph in graphs:
            graph.replay()
    torch.cuda.synchronize()
    samples = [[] for _ in jobs]
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for round_idx in range(9):
        order = [(i + round_idx) % len(jobs) for i in range(len(jobs))]
        if round_idx % 2:
            order.reverse()
        for i in order:
            start.record()
            graphs[i].replay()
            end.record()
            end.synchronize()
            samples[i].append(start.elapsed_time(end) * 1000 / repeats)
    return samples


def check_ws(kernel, heads):
    """Exercise independent sequences, ring-buffer wraparound, and history reset."""
    offsets = [0, 1, 3, 8, 25, 90]
    shape = (offsets[-1], heads, 128, 128)
    dh = torch.randn(shape, device="cuda") * 0.1
    m = (0.95 * torch.eye(128, device="cuda")
         + torch.randn(shape, device="cuda") * 0.005).bfloat16()
    seed = torch.randn((len(offsets) - 1, heads, 128, 128), device="cuda") * 0.1
    seq_map = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    fallback = torch.empty(shape[:2], device="cuda", dtype=torch.bool)
    out = torch.empty_like(seed)
    worst = 0
    for matrix in ("near_identity", "dense"):
        if matrix == "dense":
            m = (torch.randn_like(dh) * 0.04).bfloat16()
        seed.normal_(std=0.1)
        for zero_seed in (False, True):
            if zero_seed:
                seed.zero_()
            for pattern in ("all", "none", "mixed", "random"):
                fallback.fill_(pattern != "none")
                if pattern == "mixed":
                    fallback[::2, ::2] = False
                if pattern == "random":
                    fallback.copy_(torch.rand(fallback.shape, device="cuda") < 0.7)
                expected = torch.cat([
                    reference(dh[lo:hi], m[lo:hi], fallback[lo:hi], seed[b:b + 1])
                    for b, (lo, hi) in enumerate(zip(offsets, offsets[1:]))
                ])
                label = f"ws {matrix}, {pattern}, zero_seed={zero_seed}"
                kernel(seed, dh, m, fallback, seq_map, out)
                first = out.clone()
                if pattern == "none":
                    assert torch.equal(out, dh[offsets[:-1]]), "History reset must copy dh exactly"
                for _ in range(20):
                    kernel(seed, dh, m, fallback, seq_map, out)
                    worst = max(worst, check(out, expected, label))
                    assert torch.equal(out, first), f"Non-deterministic result: {label}"
    print(f"WS stress passed: lengths 1/2/5/17/65, two matrix families, zero/nonzero seeds, "
          f"four fallback patterns, 320 repeats; max rel err={worst:.3e}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    assert min(args.heads, args.segments, args.repeats) > 0
    assert torch.cuda.get_device_capability() == (12, 0), "This demo targets SM120"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    heads, segments = args.heads, args.segments
    shape = (segments, heads, 128, 128)
    dh = torch.randn(shape, device="cuda") * 0.1
    # Near-identity transitions retain history over many segments without exploding.
    m = (0.95 * torch.eye(128, device="cuda")
         + torch.randn(shape, device="cuda") * 0.005).bfloat16()
    seed = torch.randn((1, heads, 128, 128), device="cuda") * 0.1
    seq_map = torch.tensor([0, segments], device="cuda", dtype=torch.int32)
    fallback = torch.ones((segments, heads), device="cuda", dtype=torch.bool)
    print(f"GPU: {torch.cuda.get_device_name()}; heads={heads}, segments={segments}")
    print("K=V=128, KxV layout, BF16 M, FP32 dh; all-fallback timing")
    print("Correctness: all/mixed fallback + nonzero seed; baseline tile=32", flush=True)

    jobs = []
    for tile in (32, 64):
        out = torch.empty_like(seed)
        kernel = tilelang_correct_h0(
            H=heads, DK=128, DV=128, res_dtype=out.dtype,
            accum_dtype="float32", h_buffer_dtype=dh.dtype,
            m_buffer_dtype=m.dtype, seqlen_dtype=seq_map.dtype,
            mask_dtype=fallback.dtype, use_raw_h0=True, state_v_first=False,
            reverse=True, transpose_m=True, block_DV=tile, store_inter_h=True,
        )
        jobs.append((f"tile{tile}", partial(kernel, seed, dh, m, fallback, seq_map, out), out))

    ws_out = torch.empty_like(seed)
    ws_kernel = tilelang_aggregate_dh_ws(heads)
    check_ws(ws_kernel, heads)
    jobs.append(("ws_tma", partial(
        ws_kernel, seed, dh, m, fallback, seq_map, ws_out
    ), ws_out))

    # Validate every candidate before timing; mixed fallback also checks history reset.
    errors = {}
    for mixed in (False, True):
        fallback.fill_(True)
        if mixed:
            fallback[segments // 2, ::2] = False
        expected = reference(dh, m, fallback, seed)
        for name, fn, out in jobs:
            fn()
            error = check(out, expected, f"{name}, mixed={mixed}")
            errors[name] = max(errors.get(name, 0), error)

    fallback.fill_(True)
    print("Correctness passed. Timing preallocated kernels (9 interleaved graph samples).")
    samples = bench_us(jobs, args.repeats)
    expected = reference(dh, m, fallback, seed)
    print(f"{'aggregate_dh':<16} {'us':>10} {'vs tile32':>10} {'min..max us':>20} {'max rel err':>12}")
    baseline = median(samples[0])
    for (name, fn, out), times in zip(jobs, samples):
        errors[name] = max(errors[name], check(out, expected, name + " after graph replay"))
        elapsed = median(times)
        print(f"{name:<16} {elapsed:10.3f} "
              f"{baseline / elapsed:9.2f}x {min(times):8.3f}..{max(times):8.3f} "
              f"{errors[name]:12.3e}", flush=True)


if __name__ == "__main__":
    main()
