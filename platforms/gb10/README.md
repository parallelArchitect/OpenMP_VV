# GB10 / DGX Spark — OpenMP VV Platform Notes

Platform: NVIDIA GB10 (SM 12.1), AArch64

This document describes candidate OpenMP VV tests for
`#pragma omp requires unified_shared_memory`.
The platform on which these tests are intended to run reports:

`cudaDevAttrPageableMemoryAccessUsesHostPageTables = 1`

Data:
https://forums.developer.nvidia.com/t/gb10-hardware-baseline-first-direct-measurements-and-findings/367851

---

## Platform Characteristics Relevant to OpenMP Conformance

Hardware measurements collected across three independent GB10 systems
produced the following baseline observations:

| Metric                        | Value  |
| ----------------------------- | ------ |
| Fault COLD/WARM p50 ratio     | 1.00x  |
| Atomic SYS/GPU scope ratio    | 1.00x  |
| NVLink-C2C coherence overhead | 0.0 ns |

Source:
https://github.com/parallelArchitect/nvidia-uma-fault-probe

These measurements are consistent with a hardware-coherent memory model
and motivate evaluation of OpenMP unified shared memory behavior on
this platform.

OpenMP map-clause behavior under
`#pragma omp requires unified_shared_memory`
has not yet been characterized on GB10.
The test candidates in this directory focus on host-pointer
accessibility and unified shared-memory semantics without explicit
mapping directives.

---

## Test Candidates

Both tests include a runtime platform gate.
`cudaDevAttrPageableMemoryAccessUsesHostPageTables`
is queried during initialization.

Systems reporting:
- `0` -> SKIP
- `1` -> RUN

### `tests/gb10/test_gb10_requires_unified_shared_memory_nomap.c`

Evaluates heap allocation access and interleaved CPU/GPU updates under
`#pragma omp requires unified_shared_memory`
without explicit map clauses.

### `tests/gb10/test_gb10_uma_pointer_validity.c`

Evaluates host stack and heap pointer accessibility inside an OpenMP
target region when unified shared memory is active and neither
`map` nor `is_device_ptr` clauses are provided.

Both tests follow established OMPVV conventions and macro usage.

---

## Validation Status

Neither test has been executed on GB10 hardware at the time of writing.
Execution on GB10 would provide conformance data for OpenMP unified
shared memory constructs on a platform reporting:

`cudaDevAttrPageableMemoryAccessUsesHostPageTables = 1`

---

## References

Hardware baseline and measurement data:
https://forums.developer.nvidia.com/t/gb10-hardware-baseline-first-direct-measurements-and-findings/367851

UMA characterization tools:
https://github.com/parallelArchitect/nvidia-uma-fault-probe

Cross-platform UMA comparison (GB10, GH200, H100-SXM5):
https://github.com/parallelArchitect/nvidia-uma-fault-probe/issues/2

OpenMP unified shared memory specification:
https://www.openmp.org/spec-html/5.0/openmpsu13.html
