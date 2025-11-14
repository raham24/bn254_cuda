#include "../include/field_ops.cuh"
#include <cstdio>
#include <cstdlib>
#include <ctime>

/**
 * Basic Functionality Tests for BN254 Field Operations
 *
 * Tests individual operations and edge cases
 */

// Test result tracking
int tests_passed = 0;
int tests_failed = 0;

#define TEST_START(name) \
    printf("[Test] %s... ", name); \
    fflush(stdout)

#define TEST_PASS() \
    do { \
        printf("PASS\n"); \
        tests_passed++; \
    } while(0)

#define TEST_FAIL(msg) \
    do { \
        printf("FAIL: %s\n", msg); \
        tests_failed++; \
    } while(0)

#define ASSERT_EQUAL(a, b, msg) \
    if ((a) != (b)) { \
        TEST_FAIL(msg); \
        return false; \
    }

// ====================================================================================
// TEST FUNCTIONS
// ====================================================================================

bool test_field_element_creation() {
    TEST_START("Field Element Creation");

    FieldElement zero = FieldConstants::zero();
    if (zero.to_uint64() != 0) {
        TEST_FAIL("Zero element not zero");
        return false;
    }

    FieldElement one = FieldConstants::one();
    if (one.to_uint64() != 1) {
        TEST_FAIL("One element not one");
        return false;
    }

    FieldElement custom = FieldElement::from_uint64(12345);
    if (custom.to_uint64() != 12345) {
        TEST_FAIL("Custom element incorrect");
        return false;
    }

    TEST_PASS();
    return true;
}

bool test_basic_addition() {
    TEST_START("Basic Addition (2 + 3 = 5)");

    FieldElement a = FieldElement::from_uint64(2);
    FieldElement b = FieldElement::from_uint64(3);
    FieldElement result;

    field_add_cpu(&result, &a, &b);

    if (result.to_uint64() != 5) {
        TEST_FAIL("2 + 3 != 5");
        return false;
    }

    TEST_PASS();
    return true;
}

bool test_addition_with_modulo() {
    TEST_START("Addition with Modulo");

    // Test: (p-1) + 2 should give 1 (mod p)
    FieldElement a = FieldElement::from_uint64(FIELD_MODULUS - 1);
    FieldElement b = FieldElement::from_uint64(2);
    FieldElement result;

    field_add_cpu(&result, &a, &b);

    if (result.to_uint64() != 1) {
        printf("\nExpected: 1, Got: %llu", (unsigned long long)result.to_uint64());
        TEST_FAIL("Modular reduction failed");
        return false;
    }

    TEST_PASS();
    return true;
}

bool test_basic_subtraction() {
    TEST_START("Basic Subtraction (5 - 3 = 2)");

    FieldElement a = FieldElement::from_uint64(5);
    FieldElement b = FieldElement::from_uint64(3);
    FieldElement result;

    field_sub_cpu(&result, &a, &b);

    if (result.to_uint64() != 2) {
        TEST_FAIL("5 - 3 != 2");
        return false;
    }

    TEST_PASS();
    return true;
}

bool test_subtraction_with_underflow() {
    TEST_START("Subtraction with Underflow (3 - 5)");

    FieldElement a = FieldElement::from_uint64(3);
    FieldElement b = FieldElement::from_uint64(5);
    FieldElement result;

    field_sub_cpu(&result, &a, &b);

    // Should give p - 2
    uint64_t expected = FIELD_MODULUS - 2;
    if (result.to_uint64() != expected) {
        printf("\nExpected: %llu, Got: %llu",
               (unsigned long long)expected,
               (unsigned long long)result.to_uint64());
        TEST_FAIL("Underflow handling failed");
        return false;
    }

    TEST_PASS();
    return true;
}

bool test_basic_multiplication() {
    TEST_START("Basic Multiplication (7 * 11 = 77)");

    FieldElement a = FieldElement::from_uint64(7);
    FieldElement b = FieldElement::from_uint64(11);
    FieldElement result;

    field_mul_cpu(&result, &a, &b);

    if (result.to_uint64() != 77) {
        TEST_FAIL("7 * 11 != 77");
        return false;
    }

    TEST_PASS();
    return true;
}

bool test_multiplication_with_modulo() {
    TEST_START("Multiplication with Modulo");

    // Test multiplication that requires reduction
    FieldElement a = FieldElement::from_uint64(1000);
    FieldElement b = FieldElement::from_uint64(100);
    FieldElement result;

    field_mul_cpu(&result, &a, &b);

    uint64_t expected = (1000ULL * 100ULL) % FIELD_MODULUS;
    if (result.to_uint64() != expected) {
        printf("\nExpected: %llu, Got: %llu",
               (unsigned long long)expected,
               (unsigned long long)result.to_uint64());
        TEST_FAIL("Modular multiplication failed");
        return false;
    }

    TEST_PASS();
    return true;
}

bool test_zero_and_one() {
    TEST_START("Zero and One Edge Cases");

    FieldElement zero = FieldConstants::zero();
    FieldElement one = FieldConstants::one();
    FieldElement a = FieldElement::from_uint64(42);
    FieldElement result;

    // Test: a + 0 = a
    field_add_cpu(&result, &a, &zero);
    if (result.to_uint64() != 42) {
        TEST_FAIL("a + 0 != a");
        return false;
    }

    // Test: a * 1 = a
    field_mul_cpu(&result, &a, &one);
    if (result.to_uint64() != 42) {
        TEST_FAIL("a * 1 != a");
        return false;
    }

    // Test: a * 0 = 0
    field_mul_cpu(&result, &a, &zero);
    if (result.to_uint64() != 0) {
        TEST_FAIL("a * 0 != 0");
        return false;
    }

    TEST_PASS();
    return true;
}

bool test_commutativity() {
    TEST_START("Commutativity (a+b = b+a, a*b = b*a)");

    FieldElement a = FieldElement::from_uint64(123);
    FieldElement b = FieldElement::from_uint64(456);
    FieldElement result1, result2;

    // Test addition commutativity
    field_add_cpu(&result1, &a, &b);
    field_add_cpu(&result2, &b, &a);
    if (result1.to_uint64() != result2.to_uint64()) {
        TEST_FAIL("Addition not commutative");
        return false;
    }

    // Test multiplication commutativity
    field_mul_cpu(&result1, &a, &b);
    field_mul_cpu(&result2, &b, &a);
    if (result1.to_uint64() != result2.to_uint64()) {
        TEST_FAIL("Multiplication not commutative");
        return false;
    }

    TEST_PASS();
    return true;
}

bool test_associativity() {
    TEST_START("Associativity ((a+b)+c = a+(b+c))");

    FieldElement a = FieldElement::from_uint64(100);
    FieldElement b = FieldElement::from_uint64(200);
    FieldElement c = FieldElement::from_uint64(300);
    FieldElement temp, result1, result2;

    // Test: (a + b) + c
    field_add_cpu(&temp, &a, &b);
    field_add_cpu(&result1, &temp, &c);

    // Test: a + (b + c)
    field_add_cpu(&temp, &b, &c);
    field_add_cpu(&result2, &a, &temp);

    if (result1.to_uint64() != result2.to_uint64()) {
        printf("\n(a+b)+c = %llu, a+(b+c) = %llu",
               (unsigned long long)result1.to_uint64(),
               (unsigned long long)result2.to_uint64());
        TEST_FAIL("Addition not associative");
        return false;
    }

    TEST_PASS();
    return true;
}

bool test_gpu_vs_cpu_single() {
    TEST_START("GPU vs CPU Single Operation");

    FieldElement a = FieldElement::from_uint64(1234);
    FieldElement b = FieldElement::from_uint64(5678);
    FieldElement cpu_result, gpu_result;

    // CPU computation
    field_add_cpu(&cpu_result, &a, &b);

    // GPU computation
    FieldElement *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, sizeof(FieldElement));
    cudaMalloc(&d_b, sizeof(FieldElement));
    cudaMalloc(&d_result, sizeof(FieldElement));

    cudaMemcpy(d_a, &a, sizeof(FieldElement), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(FieldElement), cudaMemcpyHostToDevice);

    field_add_batch(d_result, d_a, d_b, 1);

    cudaMemcpy(&gpu_result, d_result, sizeof(FieldElement), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    if (cpu_result.to_uint64() != gpu_result.to_uint64()) {
        printf("\nCPU: %llu, GPU: %llu",
               (unsigned long long)cpu_result.to_uint64(),
               (unsigned long long)gpu_result.to_uint64());
        TEST_FAIL("GPU result doesn't match CPU");
        return false;
    }

    TEST_PASS();
    return true;
}

bool test_montgomery_conversion() {
    TEST_START("Montgomery Form Conversion");

    FieldElement a = FieldElement::from_uint64(100);
    FieldElement mont, back;

    // Convert to Montgomery form
    FieldElement *d_a, *d_mont, *d_back;
    cudaMalloc(&d_a, sizeof(FieldElement));
    cudaMalloc(&d_mont, sizeof(FieldElement));
    cudaMalloc(&d_back, sizeof(FieldElement));

    cudaMemcpy(d_a, &a, sizeof(FieldElement), cudaMemcpyHostToDevice);

    // To Montgomery
    to_montgomery_batch(d_mont, d_a, 1);

    // Back from Montgomery
    from_montgomery_batch(d_back, d_mont, 1);

    cudaMemcpy(&back, d_back, sizeof(FieldElement), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_mont);
    cudaFree(d_back);

    if (back.to_uint64() != a.to_uint64()) {
        printf("\nOriginal: %llu, After conversion: %llu",
               (unsigned long long)a.to_uint64(),
               (unsigned long long)back.to_uint64());
        TEST_FAIL("Montgomery round-trip failed");
        return false;
    }

    TEST_PASS();
    return true;
}

// ====================================================================================
// MAIN TEST RUNNER
// ====================================================================================

int main() {
    printf("\n");
    printf("========================================\n");
    printf("BN254 CUDA Basic Functionality Tests\n");
    printf("========================================\n");
    printf("\n");

    // Print GPU info
    print_device_info();
    printf("\n");

    printf("Field Modulus: p = %u\n", FIELD_MODULUS);
    printf("Montgomery R: %u\n", MONTGOMERY_R);
    printf("Limb size: %d bits\n", LIMB_BITS);
    printf("Number of limbs: %d\n", NUM_LIMBS);
    printf("\n");

    // Run all tests
    test_field_element_creation();
    test_basic_addition();
    test_addition_with_modulo();
    test_basic_subtraction();
    test_subtraction_with_underflow();
    test_basic_multiplication();
    test_multiplication_with_modulo();
    test_zero_and_one();
    test_commutativity();
    test_associativity();
    test_gpu_vs_cpu_single();
    test_montgomery_conversion();

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
