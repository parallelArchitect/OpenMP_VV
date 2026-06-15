//===---test_gb10_uma_pointer_validity.c -----------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
// GB10 / DGX Spark Platform Test
//
// Validates that host pointers are natively valid on device on
// hardware-coherent UMA platforms without is_device_ptr or map clauses.
//
// On discrete GPU: host pointer on device = undefined behavior.
// On GB10 (NVLink-C2C): host and device share one physical address space.
// Same pointer, same physical memory, no translation needed.
//
// This test is not meaningful on discrete GPU platforms.
// It is specifically designed for hardware-coherent UMA:
//   GB10 (NVIDIA Grace Blackwell, NVLink-C2C)
//   GH200 (NVIDIA Grace Hopper, NVLink-C2C)
//   MI300A (AMD, unified HBM)
//
// Covers: stack and heap host pointers without map clauses or is_device_ptr.
// Pending validation on GB10-class hardware.
//
// Confirmed GB10 baseline:
//   Atomic SYS/GPU scope ratio: 1.00x
//   Coherence cost: 0.0 ns overhead at atomic level
//   Source: https://github.com/parallelArchitect/nvidia-uma-fault-probe
//
////===----------------------------------------------------------------------===//
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

#pragma omp requires unified_shared_memory

int gb10_pointer_validity_stack() {
  OMPVV_INFOMSG("GB10 UMA: stack pointer validity on device");
  int errors = 0;
  int arr[N];

  for (int i = 0; i < N; i++) arr[i] = i;

#pragma omp target
  { for (int i = 0; i < N; i++) arr[i] *= 2; }

  for (int i = 0; i < N; i++)
    OMPVV_TEST_AND_SET_VERBOSE(errors, arr[i] != i * 2);

  return errors;
}

int gb10_pointer_validity_heap() {
  OMPVV_INFOMSG("GB10 UMA: heap pointer validity on device");
  int errors = 0;

  int *arr = (int*)malloc(sizeof(int) * N);
  if (!arr) return 1;

  for (int i = 0; i < N; i++) arr[i] = i;

  /* Pass host pointer to device via firstprivate — valid on HW coherent UMA.
   * Under requires unified_shared_memory, this pointer refers to shared memory.
   * On discrete GPU this would be undefined behavior. */
  int *ptr = arr;
#pragma omp target firstprivate(ptr)
  { for (int i = 0; i < N; i++) ptr[i] *= 2; }

  for (int i = 0; i < N; i++)
    OMPVV_TEST_AND_SET_VERBOSE(errors, arr[i] != i * 2);

  free(arr);
  return errors;
}

int main() {
  int isOffloading;
  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
  OMPVV_WARNING_IF(!isOffloading,
    "Without offloading, pointer validity is trivially satisfied on host");
  OMPVV_WARNING_IF(isOffloading,
    "GB10 test: host pointers used on device — requires hardware-coherent UMA");

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

  OMPVV_TEST_AND_SET_VERBOSE(errors, gb10_pointer_validity_stack());
  OMPVV_TEST_AND_SET_VERBOSE(errors, gb10_pointer_validity_heap());
  OMPVV_REPORT_AND_RETURN(errors);
}
