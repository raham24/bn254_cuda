#ifndef BN254_TYPES_CUH
#define BN254_TYPES_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

/**
 * BN254 Field Types and Constants
 *
 * This is a simplified 16-bit implementation for learning purposes.
 * We use 16-bit limbs to make debugging easier before scaling to 256-bit operations.
 *
 * For educational use, we use p = 65521 (largest 16-bit prime)
 * instead of the full BN254 prime.
 */

// Configuration: Using 16-bit limbs for simplicity
#define LIMB_BITS 16
#define NUM_LIMBS 4  // 4 limbs = 64 bits total

// Use 16-bit unsigned integers as limbs
typedef uint16_t limb_t;

// Use 32-bit for intermediate calculations to avoid overflow
typedef uint32_t dlimb_t;  // Double-width limb for multiplication

// Field modulus: Using largest 16-bit prime for testing
// In production BN254, this would be the full 254-bit prime
#define FIELD_MODULUS 65521u  // 0xFFF1

// Montgomery constant: R = 2^16 mod p for our 16-bit implementation
// R = 65536 mod 65521 = 15
#define MONTGOMERY_R 15u

// Montgomery R^2 mod p (for converting to Montgomery form)
// R^2 = 15^2 = 225
#define MONTGOMERY_R2 225u

// Precomputed: -p^(-1) mod 2^16 for Montgomery reduction
// This is used in Montgomery multiplication
#define MONTGOMERY_INV 65521u  // Will compute proper value

// Maximum value for a limb
#define LIMB_MAX ((1u << LIMB_BITS) - 1)

/**
 * Field Element Structure
 * Represents a field element as an array of limbs (little-endian)
 * For 16-bit limbs: element = limbs[0] + limbs[1]*2^16 + limbs[2]*2^32 + limbs[3]*2^48
 */
struct FieldElement {
    limb_t limbs[NUM_LIMBS];

    // Host/Device constructors
    __host__ __device__ FieldElement() {
        for (int i = 0; i < NUM_LIMBS; i++) {
            limbs[i] = 0;
        }
    }

    __host__ __device__ FieldElement(limb_t value) {
        limbs[0] = value;
        for (int i = 1; i < NUM_LIMBS; i++) {
            limbs[i] = 0;
        }
    }

    // Print helper (host only)
    __host__ void print() const {
        printf("[");
        for (int i = NUM_LIMBS - 1; i >= 0; i--) {
            printf("%04x", limbs[i]);
            if (i > 0) printf(" ");
        }
        printf("]");
    }

    // Convert to 64-bit value (for testing with small values)
    __host__ __device__ uint64_t to_uint64() const {
        uint64_t result = 0;
        for (int i = NUM_LIMBS - 1; i >= 0; i--) {
            result = (result << LIMB_BITS) | limbs[i];
        }
        return result;
    }

    // Create from 64-bit value
    __host__ __device__ static FieldElement from_uint64(uint64_t value) {
        FieldElement elem;
        for (int i = 0; i < NUM_LIMBS; i++) {
            elem.limbs[i] = (limb_t)(value & LIMB_MAX);
            value >>= LIMB_BITS;
        }
        return elem;
    }

    // Compare two field elements
    __host__ __device__ bool operator==(const FieldElement& other) const {
        for (int i = 0; i < NUM_LIMBS; i++) {
            if (limbs[i] != other.limbs[i]) return false;
        }
        return true;
    }

    __host__ __device__ bool operator!=(const FieldElement& other) const {
        return !(*this == other);
    }
};

// Special field elements
namespace FieldConstants {
    // Zero element
    __host__ __device__ inline FieldElement zero() {
        return FieldElement(0);
    }

    // One element
    __host__ __device__ inline FieldElement one() {
        return FieldElement(1);
    }

    // Modulus as field element
    __host__ __device__ inline FieldElement modulus() {
        return FieldElement(FIELD_MODULUS);
    }

    // Montgomery R
    __host__ __device__ inline FieldElement montgomery_r() {
        return FieldElement(MONTGOMERY_R);
    }

    // Montgomery R^2
    __host__ __device__ inline FieldElement montgomery_r2() {
        return FieldElement(MONTGOMERY_R2);
    }
}

// Error codes
enum BN254Error {
    BN254_SUCCESS = 0,
    BN254_ERROR_CUDA = 1,
    BN254_ERROR_INVALID_INPUT = 2,
    BN254_ERROR_OUT_OF_MEMORY = 3
};

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            return BN254_ERROR_CUDA; \
        } \
    } while(0)

// CUDA kernel launch checking
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel launch error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            return BN254_ERROR_CUDA; \
        } \
    } while(0)

#endif // BN254_TYPES_CUH
