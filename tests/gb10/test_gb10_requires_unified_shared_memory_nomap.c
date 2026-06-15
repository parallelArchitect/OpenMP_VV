//===---test_gb10_requires_unified_shared_memory_nomap.c -------------------===//
//
// OpenMP API Version 5.0 Nov 2018
// GB10 / DGX Spark Platform Test
//
// Validates #pragma omp requires unified_shared_memory on hardware-coherent
// UMA platforms (GB10, GH200) without explicit map clauses.
//
// On discrete GPU (A100, H100): NVHPC fails this — map clause required.
// On GB10 (NVLink-C2C hardware-coherent UMA): map clause is unnecessary.
// The pointer is valid on both CPU and GPU by hardware design.
//
// Confirmed GB10 baseline:
//   Fault COLD/WARM ratio: 1.00x — zero migration cost
//   NVLink-C2C coherence overhead: 0.0 ns at atomic level
//   Source: https://github.com/parallelArchitect/nvidia-uma-fault-probe
//
////===----------------------------------------------------------------------===//
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

#pragma omp requires unified_shared_memory

int gb10_nomap_heap() {
  OMPVV_INFOMSG("GB10 UMA: heap pointer, no map clause");
  int errors = 0;

  int *arr = (int*)malloc(sizeof(int) * N);
  if (!arr) { OMPVV_INFOMSG("malloc failed"); return 1; }

  for (int i = 0; i < N; i++) arr[i] = i;

#pragma omp target
  { for (int i = 0; i < N; i++) arr[i] += 10; }

  for (int i = 0; i < N; i++) arr[i] += 10;

  int mismatch = 0;
#pragma omp target map(tofrom:mismatch)
  { for (int i = 0; i < N; i++) if (arr[i] != i + 20) mismatch++; }

  OMPVV_TEST_AND_SET_VERBOSE(errors, mismatch != 0);
  free(arr);
  return errors;
}

int gb10_nomap_interleaved() {
  OMPVV_INFOMSG("GB10 UMA: interleaved CPU/GPU writes, no map clause");
  int errors = 0;

  int *arr = (int*)malloc(sizeof(int) * N);
  if (!arr) return 1;
  for (int i = 0; i < N; i++) arr[i] = 0;

#pragma omp target
  { arr[0] = 1; }
  arr[1] = 2;
#pragma omp target
  { arr[2] = arr[0] + arr[1]; }

  OMPVV_TEST_AND_SET_VERBOSE(errors, arr[0] != 1);
  OMPVV_TEST_AND_SET_VERBOSE(errors, arr[1] != 2);
  OMPVV_TEST_AND_SET_VERBOSE(errors, arr[2] != 3);

  free(arr);
  return errors;
}

int main() {
  int isOffloading;
  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
  OMPVV_WARNING_IF(!isOffloading, "Without offloading, unified shared memory is trivially satisfied");
  OMPVV_WARNING_IF(isOffloading, "GB10 test: map clause intentionally omitted — hardware coherence required");

  int errors = 0;
  /* Platform gate — hardware-coherent UMA required.
   * cudaDevAttrPageableMemoryAccessUsesHostPageTables returns 1 on GB10/GH200.
   * On discrete GPU this returns 0 — test is not meaningful, skip cleanly. */
  int hpt = 0;
  cudaDeviceGetAttribute(&hpt,
      cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0);
  if (!hpt) {
    OMPVV_WARNING_IF(1,
      "GB10 test requires hardware-coherent UMA (cudaDevAttrPageableMemoryAccessUsesHostPageTables=1) — skipping");
    OMPVV_REPORT_AND_RETURN(0);
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, gb10_nomap_heap());
  OMPVV_TEST_AND_SET_VERBOSE(errors, gb10_nomap_interleaved());
  OMPVV_REPORT_AND_RETURN(errors);
}
