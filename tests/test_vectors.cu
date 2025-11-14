#include "../include/field_ops.cuh"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

/**
 * Vector/Batch Operation Tests for BN254 Field Operations
 *
 * Tests batch operations with various sizes (1K, 10K, 100K elements)
 * and verifies correctness against CPU reference implementation
 */

// Test result tracking
int tests_passed = 0;
int tests_failed = 0;

#define TEST_START(name) \
    printf("[Test] %s... ", name); \
    fflush(stdout)

#define TEST_PASS(time_ms) \
    do { \
        printf("PASS (%.2fms)\n", time_ms); \
        tests_passed++; \
    } while(0)

#define TEST_FAIL(msg) \
    do { \
        printf("FAIL: %s\n", msg); \
        tests_failed++; \
    } while(0)

// ====================================================================================
// HELPER FUNCTIONS
// ====================================================================================

/**
 * Generate random field elements
 */
void generate_random_elements(std::vector<FieldElement>& elements, size_t n) {
    elements.resize(n);
    for (size_t i = 0; i < n; i++) {
        // Generate random value less than field modulus
        uint64_t val = rand() % FIELD_MODULUS;
        elements[i] = FieldElement::from_uint64(val);
    }
}

/**
 * Compare two vectors of field elements
 */
bool compare_vectors(const std::vector<FieldElement>& a,
                    const std::vector<FieldElement>& b) {
    if (a.size() != b.size()) return false;

    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].to_uint64() != b[i].to_uint64()) {
            printf("\nMismatch at index %zu: %llu != %llu",
                   i,
                   (unsigned long long)a[i].to_uint64(),
                   (unsigned long long)b[i].to_uint64());
            return false;
        }
    }
    return true;
}

/**
 * Get elapsed time in milliseconds
 */
double get_elapsed_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1000000.0;
}

// ====================================================================================
// TEST FUNCTIONS
// ====================================================================================

bool test_batch_addition(size_t n) {
    char test_name[256];
    snprintf(test_name, sizeof(test_name), "Batch Addition (%zuK elements)", n / 1000);
    TEST_START(test_name);

    // Generate random inputs
    std::vector<FieldElement> a, b;
    generate_random_elements(a, n);
    generate_random_elements(b, n);

    // Allocate device memory
    FieldElement *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, n * sizeof(FieldElement));
    cudaMalloc(&d_b, n * sizeof(FieldElement));
    cudaMalloc(&d_result, n * sizeof(FieldElement));

    // Copy inputs to device
    cudaMemcpy(d_a, a.data(), n * sizeof(FieldElement), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(FieldElement), cudaMemcpyHostToDevice);

    // Run GPU computation
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    field_add_batch(d_result, d_a, d_b, n);
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_ms = get_elapsed_ms(start, end);

    // Copy results back
    std::vector<FieldElement> gpu_result(n);
    cudaMemcpy(gpu_result.data(), d_result, n * sizeof(FieldElement), cudaMemcpyDeviceToHost);

    // Compute CPU reference
    std::vector<FieldElement> cpu_result(n);
    for (size_t i = 0; i < n; i++) {
        field_add_cpu(&cpu_result[i], &a[i], &b[i]);
    }

    // Compare results
    bool success = compare_vectors(gpu_result, cpu_result);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    if (success) {
        TEST_PASS(time_ms);
        return true;
    } else {
        TEST_FAIL("GPU results don't match CPU");
        return false;
    }
}

bool test_batch_subtraction(size_t n) {
    char test_name[256];
    snprintf(test_name, sizeof(test_name), "Batch Subtraction (%zuK elements)", n / 1000);
    TEST_START(test_name);

    std::vector<FieldElement> a, b;
    generate_random_elements(a, n);
    generate_random_elements(b, n);

    FieldElement *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, n * sizeof(FieldElement));
    cudaMalloc(&d_b, n * sizeof(FieldElement));
    cudaMalloc(&d_result, n * sizeof(FieldElement));

    cudaMemcpy(d_a, a.data(), n * sizeof(FieldElement), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(FieldElement), cudaMemcpyHostToDevice);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    field_sub_batch(d_result, d_a, d_b, n);
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_ms = get_elapsed_ms(start, end);

    std::vector<FieldElement> gpu_result(n);
    cudaMemcpy(gpu_result.data(), d_result, n * sizeof(FieldElement), cudaMemcpyDeviceToHost);

    std::vector<FieldElement> cpu_result(n);
    for (size_t i = 0; i < n; i++) {
        field_sub_cpu(&cpu_result[i], &a[i], &b[i]);
    }

    bool success = compare_vectors(gpu_result, cpu_result);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    if (success) {
        TEST_PASS(time_ms);
        return true;
    } else {
        TEST_FAIL("GPU results don't match CPU");
        return false;
    }
}

bool test_batch_multiplication(size_t n) {
    char test_name[256];
    snprintf(test_name, sizeof(test_name), "Batch Multiplication (%zuK elements)", n / 1000);
    TEST_START(test_name);

    std::vector<FieldElement> a, b;
    generate_random_elements(a, n);
    generate_random_elements(b, n);

    FieldElement *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, n * sizeof(FieldElement));
    cudaMalloc(&d_b, n * sizeof(FieldElement));
    cudaMalloc(&d_result, n * sizeof(FieldElement));

    cudaMemcpy(d_a, a.data(), n * sizeof(FieldElement), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(FieldElement), cudaMemcpyHostToDevice);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    field_mul_batch(d_result, d_a, d_b, n);
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_ms = get_elapsed_ms(start, end);

    std::vector<FieldElement> gpu_result(n);
    cudaMemcpy(gpu_result.data(), d_result, n * sizeof(FieldElement), cudaMemcpyDeviceToHost);

    std::vector<FieldElement> cpu_result(n);
    for (size_t i = 0; i < n; i++) {
        field_mul_cpu(&cpu_result[i], &a[i], &b[i]);
    }

    bool success = compare_vectors(gpu_result, cpu_result);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    if (success) {
        TEST_PASS(time_ms);
        return true;
    } else {
        TEST_FAIL("GPU results don't match CPU");
        return false;
    }
}

bool test_large_batch_operations() {
    TEST_START("Large Batch Operations (100K elements)");

    const size_t n = 100000;

    std::vector<FieldElement> a, b;
    generate_random_elements(a, n);
    generate_random_elements(b, n);

    FieldElement *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, n * sizeof(FieldElement));
    cudaMalloc(&d_b, n * sizeof(FieldElement));
    cudaMalloc(&d_result, n * sizeof(FieldElement));

    cudaMemcpy(d_a, a.data(), n * sizeof(FieldElement), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(FieldElement), cudaMemcpyHostToDevice);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Perform multiple operations
    field_add_batch(d_result, d_a, d_b, n);
    field_mul_batch(d_result, d_result, d_a, n);
    field_sub_batch(d_result, d_result, d_b, n);

    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_ms = get_elapsed_ms(start, end);

    // Just verify no errors occurred
    cudaError_t err = cudaGetLastError();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    if (err == cudaSuccess) {
        TEST_PASS(time_ms);
        return true;
    } else {
        TEST_FAIL("CUDA error occurred");
        return false;
    }
}

bool test_edge_cases_batch() {
    TEST_START("Edge Cases Batch (zeros, ones, max values)");

    const size_t n = 1000;

    // Create vectors with edge case values
    std::vector<FieldElement> a(n), b(n);

    // Fill with alternating patterns
    for (size_t i = 0; i < n; i++) {
        if (i % 3 == 0) {
            a[i] = FieldConstants::zero();
            b[i] = FieldElement::from_uint64(rand() % FIELD_MODULUS);
        } else if (i % 3 == 1) {
            a[i] = FieldConstants::one();
            b[i] = FieldConstants::one();
        } else {
            a[i] = FieldElement::from_uint64(FIELD_MODULUS - 1);
            b[i] = FieldElement::from_uint64(FIELD_MODULUS - 1);
        }
    }

    FieldElement *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, n * sizeof(FieldElement));
    cudaMalloc(&d_b, n * sizeof(FieldElement));
    cudaMalloc(&d_result, n * sizeof(FieldElement));

    cudaMemcpy(d_a, a.data(), n * sizeof(FieldElement), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(FieldElement), cudaMemcpyHostToDevice);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    field_add_batch(d_result, d_a, d_b, n);
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_ms = get_elapsed_ms(start, end);

    std::vector<FieldElement> gpu_result(n);
    cudaMemcpy(gpu_result.data(), d_result, n * sizeof(FieldElement), cudaMemcpyDeviceToHost);

    std::vector<FieldElement> cpu_result(n);
    for (size_t i = 0; i < n; i++) {
        field_add_cpu(&cpu_result[i], &a[i], &b[i]);
    }

    bool success = compare_vectors(gpu_result, cpu_result);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    if (success) {
        TEST_PASS(time_ms);
        return true;
    } else {
        TEST_FAIL("Edge case handling failed");
        return false;
    }
}

bool test_montgomery_batch() {
    TEST_START("Montgomery Batch Operations (10K elements)");

    const size_t n = 10000;

    std::vector<FieldElement> a;
    generate_random_elements(a, n);

    FieldElement *d_a, *d_mont, *d_back;
    cudaMalloc(&d_a, n * sizeof(FieldElement));
    cudaMalloc(&d_mont, n * sizeof(FieldElement));
    cudaMalloc(&d_back, n * sizeof(FieldElement));

    cudaMemcpy(d_a, a.data(), n * sizeof(FieldElement), cudaMemcpyHostToDevice);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Convert to Montgomery and back
    to_montgomery_batch(d_mont, d_a, n);
    from_montgomery_batch(d_back, d_mont, n);

    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_ms = get_elapsed_ms(start, end);

    std::vector<FieldElement> result(n);
    cudaMemcpy(result.data(), d_back, n * sizeof(FieldElement), cudaMemcpyDeviceToHost);

    bool success = compare_vectors(a, result);

    cudaFree(d_a);
    cudaFree(d_mont);
    cudaFree(d_back);

    if (success) {
        TEST_PASS(time_ms);
        return true;
    } else {
        TEST_FAIL("Montgomery round-trip failed");
        return false;
    }
}

// ====================================================================================
// MAIN TEST RUNNER
// ====================================================================================

int main() {
    // Seed random number generator
    srand(time(NULL));

    printf("\n");
    printf("========================================\n");
    printf("BN254 CUDA Vector Operation Tests\n");
    printf("========================================\n");
    printf("\n");

    print_device_info();
    printf("\n");

    // Run tests with various batch sizes
    test_batch_addition(1000);
    test_batch_addition(10000);
    test_batch_addition(100000);

    test_batch_subtraction(1000);
    test_batch_subtraction(10000);

    test_batch_multiplication(1000);
    test_batch_multiplication(10000);

    test_large_batch_operations();
    test_edge_cases_batch();
    test_montgomery_batch();

    // Print summary
    printf("\n");
    printf("========================================\n");
    printf("Test Summary\n");
    printf("========================================\n");
    printf("Tests Passed: %d\n", tests_passed);
    printf("Tests Failed: %d\n", tests_failed);
    printf("Total Tests:  %d\n", tests_passed + tests_failed);
    printf("\n");

    if (tests_failed == 0) {
        printf("All tests PASSED!\n");
        return 0;
    } else {
        printf("Some tests FAILED.\n");
        return 1;
    }
}
