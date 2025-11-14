#include "../include/field_ops.cuh"
#include <iostream>
#include <vector>
#include <cstring>

/**
 * BN254 High-Level API
 *
 * This file provides a C++-friendly wrapper around the CUDA kernels,
 * handling memory management and providing convenient batch operations.
 */

namespace BN254 {

/**
 * RAII wrapper for device memory
 */
template<typename T>
class DeviceBuffer {
private:
    T* ptr_;
    size_t size_;

public:
    DeviceBuffer(size_t n) : ptr_(nullptr), size_(n) {
        if (n > 0) {
            cudaError_t err = cudaMalloc(&ptr_, n * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to allocate device memory");
            }
        }
    }

    ~DeviceBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    // Prevent copying
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Allow moving
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }

    void copy_from_host(const T* src, size_t n) {
        if (n > size_) n = size_;
        cudaError_t err = cudaMemcpy(ptr_, src, n * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy to device");
        }
    }

    void copy_to_host(T* dst, size_t n) const {
        if (n > size_) n = size_;
        cudaError_t err = cudaMemcpy(dst, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy from device");
        }
    }
};

/**
 * High-level field operations API
 */
class FieldAPI {
public:
    /**
     * Batch addition: results = a + b (element-wise, mod p)
     */
    static void add(std::vector<FieldElement>& results,
                   const std::vector<FieldElement>& a,
                   const std::vector<FieldElement>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Input vectors must have same size");
        }

        size_t n = a.size();
        results.resize(n);

        // Allocate device memory
        DeviceBuffer<FieldElement> d_a(n);
        DeviceBuffer<FieldElement> d_b(n);
        DeviceBuffer<FieldElement> d_results(n);

        // Copy inputs to device
        d_a.copy_from_host(a.data(), n);
        d_b.copy_from_host(b.data(), n);

        // Perform addition on GPU
        BN254Error err = field_add_batch(d_results.get(), d_a.get(), d_b.get(), n);
        if (err != BN254_SUCCESS) {
            throw std::runtime_error("GPU addition failed");
        }

        // Copy results back
        d_results.copy_to_host(results.data(), n);
    }

    /**
     * Batch subtraction: results = a - b (element-wise, mod p)
     */
    static void subtract(std::vector<FieldElement>& results,
                        const std::vector<FieldElement>& a,
                        const std::vector<FieldElement>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Input vectors must have same size");
        }

        size_t n = a.size();
        results.resize(n);

        DeviceBuffer<FieldElement> d_a(n);
        DeviceBuffer<FieldElement> d_b(n);
        DeviceBuffer<FieldElement> d_results(n);

        d_a.copy_from_host(a.data(), n);
        d_b.copy_from_host(b.data(), n);

        BN254Error err = field_sub_batch(d_results.get(), d_a.get(), d_b.get(), n);
        if (err != BN254_SUCCESS) {
            throw std::runtime_error("GPU subtraction failed");
        }

        d_results.copy_to_host(results.data(), n);
    }

    /**
     * Batch multiplication: results = a * b (element-wise, mod p)
     */
    static void multiply(std::vector<FieldElement>& results,
                        const std::vector<FieldElement>& a,
                        const std::vector<FieldElement>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Input vectors must have same size");
        }

        size_t n = a.size();
        results.resize(n);

        DeviceBuffer<FieldElement> d_a(n);
        DeviceBuffer<FieldElement> d_b(n);
        DeviceBuffer<FieldElement> d_results(n);

        d_a.copy_from_host(a.data(), n);
        d_b.copy_from_host(b.data(), n);

        BN254Error err = field_mul_batch(d_results.get(), d_a.get(), d_b.get(), n);
        if (err != BN254_SUCCESS) {
            throw std::runtime_error("GPU multiplication failed");
        }

        d_results.copy_to_host(results.data(), n);
    }

    /**
     * Batch Montgomery multiplication
     */
    static void multiply_montgomery(std::vector<FieldElement>& results,
                                   const std::vector<FieldElement>& a,
                                   const std::vector<FieldElement>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Input vectors must have same size");
        }

        size_t n = a.size();
        results.resize(n);

        DeviceBuffer<FieldElement> d_a(n);
        DeviceBuffer<FieldElement> d_b(n);
        DeviceBuffer<FieldElement> d_results(n);

        d_a.copy_from_host(a.data(), n);
        d_b.copy_from_host(b.data(), n);

        BN254Error err = field_mul_montgomery_batch(d_results.get(), d_a.get(), d_b.get(), n);
        if (err != BN254_SUCCESS) {
            throw std::runtime_error("GPU Montgomery multiplication failed");
        }

        d_results.copy_to_host(results.data(), n);
    }

    /**
     * Convert to Montgomery form
     */
    static void to_montgomery(std::vector<FieldElement>& results,
                             const std::vector<FieldElement>& a) {
        size_t n = a.size();
        results.resize(n);

        DeviceBuffer<FieldElement> d_a(n);
        DeviceBuffer<FieldElement> d_results(n);

        d_a.copy_from_host(a.data(), n);

        BN254Error err = to_montgomery_batch(d_results.get(), d_a.get(), n);
        if (err != BN254_SUCCESS) {
            throw std::runtime_error("GPU to_montgomery failed");
        }

        d_results.copy_to_host(results.data(), n);
    }

    /**
     * Convert from Montgomery form
     */
    static void from_montgomery(std::vector<FieldElement>& results,
                               const std::vector<FieldElement>& a) {
        size_t n = a.size();
        results.resize(n);

        DeviceBuffer<FieldElement> d_a(n);
        DeviceBuffer<FieldElement> d_results(n);

        d_a.copy_from_host(a.data(), n);

        BN254Error err = from_montgomery_batch(d_results.get(), d_a.get(), n);
        if (err != BN254_SUCCESS) {
            throw std::runtime_error("GPU from_montgomery failed");
        }

        d_results.copy_to_host(results.data(), n);
    }

    /**
     * Print device information
     */
    static void print_device_info() {
        ::print_device_info();
    }
};

} // namespace BN254
