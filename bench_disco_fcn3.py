#!/usr/bin/env python3
import argparse
import gc
import math
import statistics
from dataclasses import dataclass

import torch

from torch_harmonics import DiscreteContinuousConvS2
from torch_harmonics.disco.optimized.disco_optimized import _disco_s2_contraction_optimized


@dataclass(frozen=True)
class BufferSpec:
    name: str
    in_shape: tuple[int, int]
    out_shape: tuple[int, int]
    grid_in: str
    grid_out: str
    theta_factor: float


@dataclass(frozen=True)
class BenchCase:
    name: str
    buffer: str
    shape: tuple[int, int, int, int]
    trace_count: int


BUFFERS = {
    "enc_721_to_360": BufferSpec("enc_721_to_360", (721, 1440), (360, 720), "equiangular", "legendre-gauss", 1.0),
    "local_360": BufferSpec("local_360", (360, 720), (360, 720), "legendre-gauss", "legendre-gauss", 2.0),
    "dec_721": BufferSpec("dec_721", (721, 1440), (721, 1440), "equiangular", "equiangular", 1.0),
}

CASES = [
    BenchCase("enc_b1_c7", "enc_721_to_360", (1, 7, 721, 1440), 12),
    BenchCase("enc_b1_c12", "enc_721_to_360", (1, 12, 721, 1440), 12),
    BenchCase("enc_b13_c5", "enc_721_to_360", (13, 5, 721, 1440), 12),
    BenchCase("local_b1_c677", "local_360", (1, 677, 360, 720), 96),
    BenchCase("dec_b1_c56", "dec_721", (1, 56, 721, 1440), 12),
    BenchCase("dec_b13_c45", "dec_721", (13, 45, 721, 1440), 12),
]


def cutoff_radius(nlat: int, kernel_shape: tuple[int, int] = (3, 3), basis_type: str = "morlet") -> float:
    factors = {"piecewise linear": 0.5, "morlet": 0.5, "zernike": math.sqrt(2.0)}
    return (kernel_shape[0] + 1) * factors[basis_type] * math.pi / float(nlat - 1)


def build_buffer(spec: BufferSpec, device: torch.device):
    theta_cutoff = spec.theta_factor * cutoff_radius(spec.in_shape[0])
    conv = DiscreteContinuousConvS2(
        1,
        1,
        in_shape=spec.in_shape,
        out_shape=spec.out_shape,
        kernel_shape=(3, 3),
        basis_type="morlet",
        basis_norm_mode="mean",
        grid_in=spec.grid_in,
        grid_out=spec.grid_out,
        bias=False,
        theta_cutoff=theta_cutoff,
        optimized_kernel=True,
    ).to(device)
    return {
        "roff": conv.psi_roff_idx,
        "ker": conv.psi_ker_idx,
        "row": conv.psi_row_idx,
        "col": conv.psi_col_idx,
        "vals": conv.psi_vals,
        "kernel_size": conv.kernel_size,
        "nlat_out": conv.nlat_out,
        "nlon_out": conv.nlon_out,
    }


def summarize_buffer(name: str, buf: dict):
    print(
        "BUFFER",
        name,
        f"roff={buf['roff'].numel()}",
        f"nnz={buf['vals'].numel()}",
        f"idx_dtype={buf['roff'].dtype}",
        f"K={buf['kernel_size']}",
        f"out={buf['nlat_out']}x{buf['nlon_out']}",
        flush=True,
    )


def time_case(case: BenchCase, buf: dict, warmup: int, repeats: int, dtype: torch.dtype):
    device = torch.device("cuda")
    torch.manual_seed(123)
    x = torch.randn(case.shape, dtype=dtype, device=device)
    for _ in range(warmup):
        y = _disco_s2_contraction_optimized(
            x,
            buf["roff"],
            buf["ker"],
            buf["row"],
            buf["col"],
            buf["vals"],
            buf["kernel_size"],
            buf["nlat_out"],
            buf["nlon_out"],
        )
        y.sum().item()
        del y
    torch.cuda.synchronize()

    timings = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        y = _disco_s2_contraction_optimized(
            x,
            buf["roff"],
            buf["ker"],
            buf["row"],
            buf["col"],
            buf["vals"],
            buf["kernel_size"],
            buf["nlat_out"],
            buf["nlon_out"],
        )
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
        del y

    del x
    gc.collect()
    torch.cuda.empty_cache()
    return timings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--case", action="append", choices=[case.name for case in CASES])
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    device = torch.device("cuda")
    print("DEVICE", torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device), flush=True)
    selected = [case for case in CASES if args.case is None or case.name in set(args.case)]
    needed_buffers = sorted({case.buffer for case in selected})
    buffers = {}
    for name in needed_buffers:
        print("BUILD_BUFFER", name, flush=True)
        buffers[name] = build_buffer(BUFFERS[name], device)
        summarize_buffer(name, buffers[name])

    aggregate_ms = 0.0
    for case in selected:
        print("RUN_CASE", case.name, f"shape={case.shape}", f"trace_count={case.trace_count}", flush=True)
        timings = time_case(case, buffers[case.buffer], args.warmup, args.repeats, torch.float32)
        mean_ms = statistics.mean(timings)
        median_ms = statistics.median(timings)
        stdev_ms = statistics.stdev(timings) if len(timings) > 1 else 0.0
        trace_ms = mean_ms * case.trace_count
        aggregate_ms += trace_ms
        samples = ",".join(f"{t:.3f}" for t in timings)
        print(
            "RESULT",
            case.name,
            f"mean_ms={mean_ms:.3f}",
            f"median_ms={median_ms:.3f}",
            f"stdev_ms={stdev_ms:.3f}",
            f"trace_weighted_ms={trace_ms:.3f}",
            f"samples=[{samples}]",
            flush=True,
        )
    print(f"AGGREGATE trace_weighted_ms={aggregate_ms:.3f}", flush=True)


if __name__ == "__main__":
    main()
