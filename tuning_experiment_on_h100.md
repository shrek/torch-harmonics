# DISCO Kernel Tuning Session

This note summarizes the DISCO forward-kernel tuning done for the FCN3 inference
workflow:

```text
earth2studio-project/serve/server/example_workflows/foundry_fcn3.py
```

The profiling trace used for this session was:

```text
/outputs/foundry_fcn3_workflow/exec_1778186051_0d7d4ad8/fcn3-0507.trace.json.gz
```

The goal was to benchmark candidate DISCO kernel changes against trace-shaped
inputs, keep changes only when they improved the measured FCN3-shaped benchmark,
and build only for the GPU in this environment.

## Build Target

The machine GPU was:

```text
NVIDIA H100 80GB HBM3
CUDA capability: 9.0
```

The extension was rebuilt with:

```bash
TORCH_CUDA_ARCH_LIST=9.0 FORCE_CUDA_EXTENSION=1 python setup.py build_ext --inplace
```

After clearing the stale build directory and rebuilding, `cuobjdump --list-elf`
showed only sm90 cubins:

```text
ELF file 1: _C.cpython-312-x86_64-linux-gnu.1.sm_90.cubin
ELF file 2: _C.cpython-312-x86_64-linux-gnu.2.sm_90.cubin
```

## Trace Inputs

The trace showed DISCO forward kernels dominating the GPU kernel time:

| Kernel | Calls | Total time |
| --- | ---: | ---: |
| `disco_fwd_blk_k<64, 12, float, float>` | 132 | ~5925 ms |
| `disco_fwd_blk_k<64, 23, float, float>` | 24 | ~917 ms |

Combined DISCO forward time was approximately 6.84 s out of 11.48 s total GPU
kernel time in the trace.

The FCN3 trace produced the following DISCO-shaped input groups:

| Case | Input shape | K | Output | roff | nnz | Trace calls |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Encoder small channels | `(1, 7, 721, 1440)` | 9 | `360 x 720` | 3241 | 194760 | 12 |
| Encoder medium channels | `(1, 12, 721, 1440)` | 9 | `360 x 720` | 3241 | 194760 | 12 |
| Encoder batched | `(13, 5, 721, 1440)` | 9 | `360 x 720` | 3241 | 194760 | 12 |
| Local high channels | `(1, 677, 360, 720)` | 9 | `360 x 720` | 3241 | 585036 | 96 |
| Decoder medium channels | `(1, 56, 721, 1440)` | 9 | `721 x 1440` | 6490 | 444195 | 12 |
| Decoder batched | `(13, 45, 721, 1440)` | 9 | `721 x 1440` | 6490 | 444195 | 12 |

The matched FCN3 DISCO buffers were:

| Buffer | Input grid | Output grid | Input | Output | Kernel | Cutoff factor |
| --- | --- | --- | --- | --- | --- | ---: |
| Encoder | equiangular | legendre-gauss | `721 x 1440` | `360 x 720` | morlet `(3, 3)` | 1 |
| Local | legendre-gauss | legendre-gauss | `360 x 720` | `360 x 720` | morlet `(3, 3)` | 2 |
| Decoder | equiangular | equiangular | `721 x 1440` | `721 x 1440` | morlet `(3, 3)` | 1 |

## Benchmark Method

The benchmark script added for this session is:

```text
bench_disco_fcn3.py
```

It constructs the FCN3-shaped `DiscreteContinuousConvS2` buffers, runs
`_disco_s2_contraction_optimized` directly, and reports CUDA-event timings for
each trace-shaped case.

Command used for the final timing pass:

```bash
python bench_disco_fcn3.py --warmup 3 --repeats 8
```

The aggregate metric is trace-weighted mean latency:

```text
sum(mean_ms_for_case * trace_call_count)
```

## Baseline

Baseline timing on the sm90-only H100 build before the kept kernel changes:

| Case | Mean ms | Trace-weighted ms |
| --- | ---: | ---: |
| `enc_b1_c7` | 0.807 | 9.684 |
| `enc_b1_c12` | 1.065 | 12.780 |
| `enc_b13_c5` | 3.707 | 44.484 |
| `local_b1_c677` | 64.179 | 6161.184 |
| `dec_b1_c56` | 8.098 | 97.176 |
| `dec_b13_c45` | 77.420 | 929.040 |
| **Aggregate** | | **7254.308** |

## Net Performance Improvements

The table below attributes the measured trace-weighted aggregate improvements to
the techniques that were kept. The attribution is incremental in the order the
techniques were benchmarked, so small differences include benchmark noise between
runs.

| Technique | Aggregate before | Aggregate after | Net improvement | Decision |
| --- | ---: | ---: | ---: | --- |
| Conditional `torch::empty` allocation for dense rows | 7254.308 ms | 6973.143 ms | **281.165 ms, 3.9% faster** | Kept |
| Forward `PSCALE` specialization | 6973.143 ms | 6169.654 ms | **803.489 ms, 11.5% faster incrementally** | Kept |
| Dense-row `ker,row` derivation from `blockIdx.x` | 6169.654 ms | 6159.774 ms | **9.880 ms, 0.2% faster incrementally** | Kept |
| Final combined kept changes | 7254.308 ms | 6159.774 ms | **1094.534 ms, 15.1% faster overall** | Kept |

## Strategies Tried

### 1. Confirm and Add Forward pscale Specialization

Finding:

The pscale specialization was already present in the backward CUDA kernel, but
the trace was using the forward path. Forward did not have the same compile-time
`PSCALE` specialization, so the traced forward kernels were not using it.

Change kept:

Forward launch now switches on runtime `pscale` and instantiates specialized
forward kernels for `PSCALE = 1`, `2`, `3`, and a generic fallback.

Reason kept:

This improved the dominant trace-shaped benchmark when combined with the other
kept forward changes.

### 2. Replace Full Output Zero-Fill With Conditional Empty Allocation

Initial idea:

The forward kernel writes every output element for dense row layouts, so
allocating with `torch::zeros` adds unnecessary work.

Change kept:

Use `torch::empty` only when `nrows == K * Ho`. Preserve `torch::zeros` for
general sparse layouts where some output rows may be absent and must remain zero.

Standalone benchmark signal:

| Variant | Aggregate trace-weighted ms |
| --- | ---: |
| Baseline | 7254.308 |
| Empty allocation experiment | 6973.143 |

Reason kept:

It improved the benchmark and was made conditional to preserve correctness for
non-dense sparse cases.

### 3. Dense-Row Fast Path for ker,row Lookup

Observation:

The FCN3 trace-shaped buffers have dense output rows:

```text
nrows == K * Ho
```

For these buffers, `ker` and `row` can be derived from `blockIdx.x`:

```text
ker = blockIdx.x / Ho
row = blockIdx.x - ker * Ho
```

Change kept:

Added a compile-time `DENSE_ROWS` specialization. Dense buffers avoid loading
`ker_idx[soff]` and `row_idx[soff]`; non-dense buffers still use the original
index tensors.

Final combined benchmark:

| Case | Baseline mean ms | Final mean ms | Trace calls | Final trace-weighted ms |
| --- | ---: | ---: | ---: | ---: |
| `enc_b1_c7` | 0.807 | 0.726 | 12 | 8.710 |
| `enc_b1_c12` | 1.065 | 0.994 | 12 | 11.930 |
| `enc_b13_c5` | 3.707 | 3.495 | 12 | 41.944 |
| `local_b1_c677` | 64.179 | 54.131 | 96 | 5196.576 |
| `dec_b1_c56` | 8.098 | 7.089 | 12 | 85.064 |
| `dec_b13_c45` | 77.420 | 67.963 | 12 | 815.550 |
| **Aggregate** | | | | **6159.774** |

Overall trace-weighted improvement:

```text
7254.308 ms -> 6159.774 ms
~15.1% faster
```

The dominant local case improved:

```text
64.179 ms -> 54.131 ms
~15.7% faster
```

### 4. Launch Configuration Tuning

Experiment:

Change the traced `Wo <= 64 * ELXTH_MAX` launch path from 64 threads to 128
threads.

Result:

| Variant | Aggregate trace-weighted ms |
| --- | ---: |
| Baseline | 7254.308 |
| 128-thread launch experiment | 9150.973 |

Decision:

Dropped. It was substantially slower, especially for the dominant local case.

### 5. Narrow int32 Index Buffers

Experiment:

Create persistent int32 versions of the DISCO index buffers and dispatch CUDA
kernels over narrower index types.

Result:

| Variant | Aggregate trace-weighted ms |
| --- | ---: |
| Best kept path before int32 experiment | ~6169.654 |
| int32 index experiment | 6212.467 |

Decision:

Dropped. The local case was roughly neutral, but decoder performance regressed,
and the extra buffer/storage complexity was not justified.

### 6. Full Row-Segment Metadata

Investigation:

Rows contain multiple contiguous input-row segments. Segment statistics for the
FCN3-shaped buffers were:

| Buffer | Rows | nnz | Segments per row mean | nnz per segment mean |
| --- | ---: | ---: | ---: | ---: |
| Encoder `721 -> 360` | 3240 | 194760 | ~4.01 | ~15.0 |
| Local `360 -> 360` | 3240 | 585036 | ~8.94 | ~20.19 |
| Decoder `721 -> 721` | 6489 | 444195 | ~4.99 | ~13.71 |

Decision:

The full segment-offset approach was not implemented in this pass because it
would require new metadata and op-schema changes. The lower-risk dense-row fast
path was tried first and delivered a clear improvement without changing the
Python-visible kernel contract.

## Correctness Checks

Focused pytest command:

```bash
python -m pytest tests/test_convolution.py -k test_optimized_against_torch -q
```

Result:

```text
68 passed, 192 deselected, 15 warnings
```

Additional targeted checks compared optimized CUDA against the torch fallback
for dense and general cases, including a sparse missing-row zero-fill case.

## Changes Kept

Main kept file change:

```text
torch_harmonics/disco/optimized/kernels_cuda/disco_cuda_fwd.cu
```

Kept forward-kernel changes:

- Compile-time `PSCALE` specialization in forward.
- Compile-time `DENSE_ROWS` specialization in forward.
- Conditional `torch::empty` output allocation for dense rows.
- Existing zero-fill behavior preserved for non-dense sparse rows.

Benchmark helper added:

```text
bench_disco_fcn3.py
```

## Changes Dropped

Dropped changes:

- 128-thread launch replacement for the 64-thread traced path.
- Persistent int32 index buffers and index-type dispatch.
- Full row-segment metadata rewrite.

## Ideas Not Tried Yet

These are possible follow-up ideas that were not implemented in this session:

1. Add exact-shape specializations for the common FCN3 cases, especially
   `K = 9`, `Wo = 720`, `Wo = 1440`, `pscale = 1`, and `pscale = 2`.
2. Revisit full row-segment metadata to reduce the per-nonzero input-row change
   branch and make shared-memory row loads more explicit.
3. Benchmark vectorized shared-memory row loads, such as loading multiple
   contiguous values per thread when alignment allows it.
4. Try a warp-level implementation for smaller channel/batch products where the
   current block-per-row strategy may underutilize the GPU.
5. Investigate whether the local `360 x 720` high-channel case can reuse loaded
   input rows across neighboring kernel rows or output rows.
6. Explore a fused FCN3-specific path if adjacent operations around DISCO make
   material intermediate tensors or dtype conversions expensive.
7. Profile with Nsight Compute to check whether the final kernel is limited by
   shared-memory bandwidth, global-memory bandwidth, occupancy, or instruction
   throughput on H100.
8. Tune `ELXTH_MAX` and unroll depth per `Wo` instead of using one shared
   recursive launch policy across all output widths.
9. Consider precomputing row-local column offsets if it reduces integer
   arithmetic enough to offset the extra metadata reads.
10. Benchmark fp16/bfloat16 inference paths separately if FCN3 inference can run
    at reduced precision without accuracy regressions.
