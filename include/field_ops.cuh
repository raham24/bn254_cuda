#ifndef FIELD_OPS_CUH
#define FIELD_OPS_CUH

#include "bn254_types.cuh"
#include <cuda_runtime.h>

/**
 * Field Operations for BN254 Curve (16-bit simplified version)
 *
 * This header declares all field arithmetic operations:
 * - Addition and subtraction modulo p
 * - Multiplication modulo p (schoolbook method)
 * - Montgomery multiplication (optimized)
 * - Batch operations for GPU acceleration
 *
 * All operations work on arrays of FieldElements for vectorized processing.
 */

// ====================================================================================
// DEVICE FUNCTIONS (called from kernels)
// ====================================================================================

/**
 * Add two field elements: result = (a + b) mod p
 * Returns true if carry occurred (for debugging)
 */
__device__ bool field_add_single(FieldElement* result, const FieldElement* a, const FieldElement* b);

/**
 * Subtract two field elements: result = (a - b) mod p
 * Returns true if borrow occurred (for debugging)
 */
__device__ bool field_sub_single(FieldElement* result, const FieldElement* a, const FieldElement* b);

/**
 * Multiply two field elements: result = (a * b) mod p
 * Uses schoolbook multiplication followed by reduction
 */
__device__ void field_mul_single(FieldElement* result, const FieldElement* a, const FieldElement* b);

/**
 * Montgomery multiplication: result = (a * b * R^-1) mod p
 * More efficient for repeated multiplications
 */
__device__ void field_mul_montgomery_single(FieldElement* result, const FieldElement* a, const FieldElement* b);

/**
 * Convert to Montgomery form: result = (a * R) mod p
 */
__device__ void to_montgomery_single(FieldElement* result, const FieldElement* a);

/**
 * Convert from Montgomery form: result = (a * R^-1) mod p
 */
__device__ void from_montgomery_single(FieldElement* result, const FieldElement* a);

/**
 * Reduce a value modulo p
 * Used after additions or multiplications
 */
__device__ void field_reduce_single(FieldElement* result, const FieldElement* a);

/**
 * Compare two field elements
 * Returns: -1 if a < b, 0 if a == b, 1 if a > b
 */
__device__ int field_compare(const FieldElement* a, const FieldElement* b);

// ====================================================================================
// CUDA KERNELS (launched from host)
// ====================================================================================

/**
 * Batch addition kernel: results[i] = (a[i] + b[i]) mod p
 * Uses one thread per operation with grid-stride loop
 */
__global__ void field_add_kernel(FieldElement* results, const FieldElement* a,
                                 const FieldElement* b, size_t n);

/**
 * Batch subtraction kernel: results[i] = (a[i] - b[i]) mod p
 */
__global__ void field_sub_kernel(FieldElement* results, const FieldElement* a,
                                 const FieldElement* b, size_t n);

/**
 * Batch multiplication kernel: results[i] = (a[i] * b[i]) mod p
 */
__global__ void field_mul_kernel(FieldElement* results, const FieldElement* a,
                                 const FieldElement* b, size_t n);

/**
 * Batch Montgomery multiplication kernel: results[i] = (a[i] * b[i] * R^-1) mod p
 */
__global__ void field_mul_montgomery_kernel(FieldElement* results, const FieldElement* a,
                                           const FieldElement* b, size_t n);

/**
 * Batch conversion to Montgomery form
 */
__global__ void to_montgomery_kernel(FieldElement* results, const FieldElement* a, size_t n);

/**
 * Batch conversion from Montgomery form
 */
__global__ void from_montgomery_kernel(FieldElement* results, const FieldElement* a, size_t n);

// ====================================================================================
// HOST API FUNCTIONS (C++ interface)
// ====================================================================================

/**
 * Batch field addition on GPU
 *
 * @param results Output array (device memory)
 * @param a First input array (device memory)
 * @param b Second input array (device memory)
 * @param n Number of elements to process
 * @param stream CUDA stream (nullptr for default stream)
 * @return Error code (BN254_SUCCESS on success)
 */
BN254Error field_add_batch(FieldElement* results, const FieldElement* a,
                          const FieldElement* b, size_t n, cudaStream_t stream = nullptr);

/**
 * Batch field subtraction on GPU
 */
BN254Error field_sub_batch(FieldElement* results, const FieldElement* a,
                          const FieldElement* b, size_t n, cudaStream_t stream = nullptr);

/**
 * Batch field multiplication on GPU (schoolbook method)
 */
BN254Error field_mul_batch(FieldElement* results, const FieldElement* a,
                          const FieldElement* b, size_t n, cudaStream_t stream = nullptr);

/**
 * Batch Montgomery multiplication on GPU
 */
BN254Error field_mul_montgomery_batch(FieldElement* results, const FieldElement* a,
                                     const FieldElement* b, size_t n, cudaStream_t stream = nullptr);

/**
 * Batch conversion to Montgomery form
 */
BN254Error to_montgomery_batch(FieldElement* results, const FieldElement* a,
                              size_t n, cudaStream_t stream = nullptr);

/**
 * Batch conversion from Montgomery form
 */
BN254Error from_montgomery_batch(FieldElement* results, const FieldElement* a,
                                size_t n, cudaStream_t stream = nullptr);

// ====================================================================================
// CPU REFERENCE IMPLEMENTATIONS (for testing/verification)
// ====================================================================================

/**
 * CPU reference: field addition
 * Used to verify GPU results
 */
void field_add_cpu(FieldElement* result, const FieldElement* a, const FieldElement* b);

/**
 * CPU reference: field subtraction
 */
void field_sub_cpu(FieldElement* result, const FieldElement* a, const FieldElement* b);

/**
 * CPU reference: field multiplication
 */
void field_mul_cpu(FieldElement* result, const FieldElement* a, const FieldElement* b);

/**
 * CPU reference: Montgomery multiplication
 */
void field_mul_montgomery_cpu(FieldElement* result, const FieldElement* a, const FieldElement* b);

/**
 * CPU reference: reduce modulo p
 */
void field_reduce_cpu(FieldElement* result, const FieldElement* a);

// ====================================================================================
// UTILITY FUNCTIONS
// ====================================================================================

/**
 * Allocate device memory for field elements
 */
BN254Error allocate_device_memory(FieldElement** ptr, size_t n);

/**
 * Free device memory
 */
void free_device_memory(FieldElement* ptr);

/**
 * Copy field elements from host to device
 */
BN254Error copy_to_device(FieldElement* dst, const FieldElement* src, size_t n);

/**
 * Copy field elements from device to host
 */
BN254Error copy_from_device(FieldElement* dst, const FieldElement* src, size_t n);

/**
 * Get GPU device properties
 */
void print_device_info();

/**
 * Calculate optimal grid/block dimensions for batch size
 */
void calculate_launch_params(size_t n, dim3* grid, dim3* block);

#endif // FIELD_OPS_CUH
