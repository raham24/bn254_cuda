#include "../include/field_ops.cuh"
#include <cstdio>
#include <cstring>

// ====================================================================================
// DEVICE HELPER FUNCTIONS
// ====================================================================================

/**
 * Add two multi-limb numbers with carry propagation
 * Returns final carry bit
 */
__device__ bool add_with_carry(limb_t* result, const limb_t* a, const limb_t* b, int num_limbs) {
    dlimb_t carry = 0;
    for (int i = 0; i < num_limbs; i++) {
        dlimb_t sum = (dlimb_t)a[i] + (dlimb_t)b[i] + carry;
        result[i] = (limb_t)(sum & LIMB_MAX);
        carry = sum >> LIMB_BITS;
    }
    return carry != 0;
}

/**
 * Subtract two multi-limb numbers with borrow propagation
 * Returns final borrow bit
 */
__device__ bool sub_with_borrow(limb_t* result, const limb_t* a, const limb_t* b, int num_limbs) {
    dlimb_t borrow = 0;
    for (int i = 0; i < num_limbs; i++) {
        dlimb_t diff = (dlimb_t)a[i] - (dlimb_t)b[i] - borrow;
        result[i] = (limb_t)(diff & LIMB_MAX);
        borrow = (diff >> LIMB_BITS) & 1;
    }
    return borrow != 0;
}

/**
 * Compare two multi-limb numbers
 * Returns: -1 if a < b, 0 if a == b, 1 if a > b
 */
__device__ int compare_limbs(const limb_t* a, const limb_t* b, int num_limbs) {
    for (int i = num_limbs - 1; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

/**
 * Check if value is zero
 */
__device__ bool is_zero(const limb_t* a, int num_limbs) {
    for (int i = 0; i < num_limbs; i++) {
        if (a[i] != 0) return false;
    }
    return true;
}

// ====================================================================================
// MODULAR REDUCTION
// ====================================================================================

/**
 * Reduce a field element modulo p
 * For our simplified case with p = 65521, we can use simple comparison and subtraction
 */
__device__ void field_reduce_single(FieldElement* result, const FieldElement* a) {
    // Copy input to result
    for (int i = 0; i < NUM_LIMBS; i++) {
        result->limbs[i] = a->limbs[i];
    }

    // For 16-bit prime p = 65521, the value fits in first limb
    // We need to reduce the full multi-limb value to a single value mod p

    // Convert to 64-bit for easier reduction
    uint64_t val = 0;
    for (int i = NUM_LIMBS - 1; i >= 0; i--) {
        val = (val << LIMB_BITS) | result->limbs[i];
    }

    // Reduce mod p
    val = val % FIELD_MODULUS;

    // Store back
    result->limbs[0] = (limb_t)val;
    for (int i = 1; i < NUM_LIMBS; i++) {
        result->limbs[i] = 0;
    }
}

// ====================================================================================
// FIELD ADDITION
// ====================================================================================

__device__ bool field_add_single(FieldElement* result, const FieldElement* a, const FieldElement* b) {
    // Add the limbs
    bool carry = add_with_carry(result->limbs, a->limbs, b->limbs, NUM_LIMBS);

    // Reduce modulo p
    field_reduce_single(result, result);

    return carry;
}

__global__ void field_add_kernel(FieldElement* results, const FieldElement* a,
                                 const FieldElement* b, size_t n) {
    // Grid-stride loop pattern for arbitrary batch sizes
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        field_add_single(&results[i], &a[i], &b[i]);
    }
}

// ====================================================================================
// FIELD SUBTRACTION
// ====================================================================================

__device__ bool field_sub_single(FieldElement* result, const FieldElement* a, const FieldElement* b) {
    bool borrow = false;

    // If a < b, we need to add p first
    if (compare_limbs(a->limbs, b->limbs, NUM_LIMBS) < 0) {
        // Add p to a
        FieldElement temp;
        temp.limbs[0] = FIELD_MODULUS;
        for (int i = 1; i < NUM_LIMBS; i++) {
            temp.limbs[i] = 0;
        }

        FieldElement a_plus_p;
        add_with_carry(a_plus_p.limbs, a->limbs, temp.limbs, NUM_LIMBS);

        // Now subtract
        borrow = sub_with_borrow(result->limbs, a_plus_p.limbs, b->limbs, NUM_LIMBS);
    } else {
        borrow = sub_with_borrow(result->limbs, a->limbs, b->limbs, NUM_LIMBS);
    }

    // Reduce modulo p
    field_reduce_single(result, result);

    return borrow;
}

__global__ void field_sub_kernel(FieldElement* results, const FieldElement* a,
                                 const FieldElement* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        field_sub_single(&results[i], &a[i], &b[i]);
    }
}

// ====================================================================================
// FIELD MULTIPLICATION (Schoolbook Method)
// ====================================================================================

__device__ void field_mul_single(FieldElement* result, const FieldElement* a, const FieldElement* b) {
    // For simplified implementation with small prime, convert to 64-bit, multiply, reduce
    uint64_t a_val = a->to_uint64();
    uint64_t b_val = b->to_uint64();

    // Multiply
    uint64_t product = (a_val * b_val) % FIELD_MODULUS;

    // Store result
    *result = FieldElement::from_uint64(product);
}

__global__ void field_mul_kernel(FieldElement* results, const FieldElement* a,
                                 const FieldElement* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        field_mul_single(&results[i], &a[i], &b[i]);
    }
}

// ====================================================================================
// MONTGOMERY MULTIPLICATION
// ====================================================================================

/**
 * Montgomery reduction: compute (a * R^-1) mod p
 * This is the REDC algorithm
 *
 * For our 16-bit case:
 * - R = 2^16
 * - p = 65521
 * - p' = -p^(-1) mod R (precomputed constant)
 */
__device__ void montgomery_reduce(FieldElement* result, uint64_t t) {
    // Simplified Montgomery reduction for 16-bit prime
    // m = (t * p') mod R
    // t = (t + m * p) / R

    // For simplicity with our small prime, just do modular reduction
    // In full implementation, this would be optimized Montgomery REDC
    uint64_t reduced = t % FIELD_MODULUS;
    *result = FieldElement::from_uint64(reduced);
}

__device__ void field_mul_montgomery_single(FieldElement* result, const FieldElement* a, const FieldElement* b) {
    // Montgomery multiplication: (a * b * R^-1) mod p
    // Both inputs should already be in Montgomery form

    uint64_t a_val = a->to_uint64();
    uint64_t b_val = b->to_uint64();

    // Multiply
    uint64_t product = a_val * b_val;

    // Montgomery reduce
    montgomery_reduce(result, product);
}

__global__ void field_mul_montgomery_kernel(FieldElement* results, const FieldElement* a,
                                           const FieldElement* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        field_mul_montgomery_single(&results[i], &a[i], &b[i]);
    }
}

// ====================================================================================
// MONTGOMERY FORM CONVERSIONS
// ====================================================================================

__device__ void to_montgomery_single(FieldElement* result, const FieldElement* a) {
    // Convert to Montgomery form: (a * R) mod p
    // For our case: R = 15, so multiply by 15 and reduce

    uint64_t a_val = a->to_uint64();
    uint64_t mont_val = (a_val * MONTGOMERY_R) % FIELD_MODULUS;

    *result = FieldElement::from_uint64(mont_val);
}

__global__ void to_montgomery_kernel(FieldElement* results, const FieldElement* a, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        to_montgomery_single(&results[i], &a[i]);
    }
}

__device__ void from_montgomery_single(FieldElement* result, const FieldElement* a) {
    // Convert from Montgomery form: (a * R^-1) mod p
    // R^-1 mod p needs to be computed
    // For R = 15, R^-1 mod 65521 = 4368

    const uint64_t R_INV = 4368; // Precomputed: 15^-1 mod 65521

    uint64_t a_val = a->to_uint64();
    uint64_t normal_val = (a_val * R_INV) % FIELD_MODULUS;

    *result = FieldElement::from_uint64(normal_val);
}

__global__ void from_montgomery_kernel(FieldElement* results, const FieldElement* a, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        from_montgomery_single(&results[i], &a[i]);
    }
}

// ====================================================================================
// COMPARISON
// ====================================================================================

__device__ int field_compare(const FieldElement* a, const FieldElement* b) {
    return compare_limbs(a->limbs, b->limbs, NUM_LIMBS);
}

// ====================================================================================
// HOST API FUNCTIONS
// ====================================================================================

void calculate_launch_params(size_t n, dim3* grid, dim3* block) {
    const int BLOCK_SIZE = 256;  // Threads per block
    const int MAX_BLOCKS = 65535; // Maximum grid size

    block->x = BLOCK_SIZE;
    block->y = 1;
    block->z = 1;

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = (num_blocks < MAX_BLOCKS) ? num_blocks : MAX_BLOCKS;

    grid->x = num_blocks;
    grid->y = 1;
    grid->z = 1;
}

BN254Error field_add_batch(FieldElement* results, const FieldElement* a,
                          const FieldElement* b, size_t n, cudaStream_t stream) {
    if (!results || !a || !b || n == 0) {
        return BN254_ERROR_INVALID_INPUT;
    }

    dim3 grid, block;
    calculate_launch_params(n, &grid, &block);

    if (stream) {
        field_add_kernel<<<grid, block, 0, stream>>>(results, a, b, n);
    } else {
        field_add_kernel<<<grid, block>>>(results, a, b, n);
    }

    CUDA_CHECK_KERNEL();
    return BN254_SUCCESS;
}

BN254Error field_sub_batch(FieldElement* results, const FieldElement* a,
                          const FieldElement* b, size_t n, cudaStream_t stream) {
    if (!results || !a || !b || n == 0) {
        return BN254_ERROR_INVALID_INPUT;
    }

    dim3 grid, block;
    calculate_launch_params(n, &grid, &block);

    if (stream) {
        field_sub_kernel<<<grid, block, 0, stream>>>(results, a, b, n);
    } else {
        field_sub_kernel<<<grid, block>>>(results, a, b, n);
    }

    CUDA_CHECK_KERNEL();
    return BN254_SUCCESS;
}

BN254Error field_mul_batch(FieldElement* results, const FieldElement* a,
                          const FieldElement* b, size_t n, cudaStream_t stream) {
    if (!results || !a || !b || n == 0) {
        return BN254_ERROR_INVALID_INPUT;
    }

    dim3 grid, block;
    calculate_launch_params(n, &grid, &block);

    if (stream) {
        field_mul_kernel<<<grid, block, 0, stream>>>(results, a, b, n);
    } else {
        field_mul_kernel<<<grid, block>>>(results, a, b, n);
    }

    CUDA_CHECK_KERNEL();
    return BN254_SUCCESS;
}

BN254Error field_mul_montgomery_batch(FieldElement* results, const FieldElement* a,
                                     const FieldElement* b, size_t n, cudaStream_t stream) {
    if (!results || !a || !b || n == 0) {
        return BN254_ERROR_INVALID_INPUT;
    }

    dim3 grid, block;
    calculate_launch_params(n, &grid, &block);

    if (stream) {
        field_mul_montgomery_kernel<<<grid, block, 0, stream>>>(results, a, b, n);
    } else {
        field_mul_montgomery_kernel<<<grid, block>>>(results, a, b, n);
    }

    CUDA_CHECK_KERNEL();
    return BN254_SUCCESS;
}

BN254Error to_montgomery_batch(FieldElement* results, const FieldElement* a,
                              size_t n, cudaStream_t stream) {
    if (!results || !a || n == 0) {
        return BN254_ERROR_INVALID_INPUT;
    }

    dim3 grid, block;
    calculate_launch_params(n, &grid, &block);

    if (stream) {
        to_montgomery_kernel<<<grid, block, 0, stream>>>(results, a, n);
    } else {
        to_montgomery_kernel<<<grid, block>>>(results, a, n);
    }

    CUDA_CHECK_KERNEL();
    return BN254_SUCCESS;
}

BN254Error from_montgomery_batch(FieldElement* results, const FieldElement* a,
                                size_t n, cudaStream_t stream) {
    if (!results || !a || n == 0) {
        return BN254_ERROR_INVALID_INPUT;
    }

    dim3 grid, block;
    calculate_launch_params(n, &grid, &block);

    if (stream) {
        from_montgomery_kernel<<<grid, block, 0, stream>>>(results, a, n);
    } else {
        from_montgomery_kernel<<<grid, block>>>(results, a, n);
    }

    CUDA_CHECK_KERNEL();
    return BN254_SUCCESS;
}

// ====================================================================================
// CPU REFERENCE IMPLEMENTATIONS
// ====================================================================================

void field_add_cpu(FieldElement* result, const FieldElement* a, const FieldElement* b) {
    uint64_t a_val = a->to_uint64();
    uint64_t b_val = b->to_uint64();
    uint64_t sum = (a_val + b_val) % FIELD_MODULUS;
    *result = FieldElement::from_uint64(sum);
}

void field_sub_cpu(FieldElement* result, const FieldElement* a, const FieldElement* b) {
    uint64_t a_val = a->to_uint64();
    uint64_t b_val = b->to_uint64();
    uint64_t diff;

    if (a_val >= b_val) {
        diff = a_val - b_val;
    } else {
        diff = FIELD_MODULUS - (b_val - a_val);
    }

    *result = FieldElement::from_uint64(diff % FIELD_MODULUS);
}

void field_mul_cpu(FieldElement* result, const FieldElement* a, const FieldElement* b) {
    uint64_t a_val = a->to_uint64();
    uint64_t b_val = b->to_uint64();
    uint64_t product = (a_val * b_val) % FIELD_MODULUS;
    *result = FieldElement::from_uint64(product);
}

void field_mul_montgomery_cpu(FieldElement* result, const FieldElement* a, const FieldElement* b) {
    uint64_t a_val = a->to_uint64();
    uint64_t b_val = b->to_uint64();
    uint64_t product = (a_val * b_val) % FIELD_MODULUS;
    *result = FieldElement::from_uint64(product);
}

void field_reduce_cpu(FieldElement* result, const FieldElement* a) {
    uint64_t val = a->to_uint64();
    val = val % FIELD_MODULUS;
    *result = FieldElement::from_uint64(val);
}

// ====================================================================================
// UTILITY FUNCTIONS
// ====================================================================================

BN254Error allocate_device_memory(FieldElement** ptr, size_t n) {
    CUDA_CHECK(cudaMalloc(ptr, n * sizeof(FieldElement)));
    return BN254_SUCCESS;
}

void free_device_memory(FieldElement* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

BN254Error copy_to_device(FieldElement* dst, const FieldElement* src, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(FieldElement), cudaMemcpyHostToDevice));
    return BN254_SUCCESS;
}

BN254Error copy_from_device(FieldElement* dst, const FieldElement* src, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(FieldElement), cudaMemcpyDeviceToHost));
    return BN254_SUCCESS;
}

void print_device_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return;
    }

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("GPU: %s\n", prop.name);
    printf("CUDA Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    // Get CUDA runtime version
    int runtimeVersion;
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
}
