#include "../include/field_ops.cuh"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>

/**
 * Performance Benchmarking Tests for BN254 Field Operations
 *
 * Measures throughput, latency, and compares GPU vs CPU performance
 */

// ====================================================================================
// HELPER FUNCTIONS
// ====================================================================================

void generate_random_elements(std::vector<FieldElement>& elements, size_t n) {
    elements.resize(n);
    for (size_t i = 0; i < n; i++) {
        uint64_t val = rand() % FIELD_MODULUS;
        elements[i] = FieldElement::from_uint64(val);
    }
}

double get_elapsed_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1000000.0;
}

void format_throughput(double ops_per_sec, char* buffer, size_t size) {
    if (ops_per_sec >= 1e9) {
        snprintf(buffer, size, "%.2f GOps/s", ops_per_sec / 1e9);
    } else if (ops_per_sec >= 1e6) {
        snprintf(buffer, size, "%.2f MOps/s", ops_per_sec / 1e6);
    } else if (ops_per_sec >= 1e3) {
        snprintf(buffer, size, "%.2f KOps/s", ops_per_sec / 1e3);
    } else {
        snprintf(buffer, size, "%.2f Ops/s", ops_per_sec);
    }
}

// ====================================================================================
// BENCHMARK FUNCTIONS
// ====================================================================================

void benchmark_gpu_addition(size_t n, int warmup_runs, int benchmark_runs) {
    printf("GPU Addition (%zu elements): ", n);
    fflush(stdout);

    std::vector<FieldElement> a, b;
    generate_random_elements(a, n);
    generate_random_elements(b, n);

    FieldElement *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, n * sizeof(FieldElement));
    cudaMalloc(&d_b, n * sizeof(FieldElement));
    cudaMalloc(&d_result, n * sizeof(FieldElement));

    cudaMemcpy(d_a, a.data(), n * sizeof(FieldElement), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(FieldElement), cudaMemcpyHostToDevice);

    // Warmup runs
    for (int i = 0; i < warmup_runs; i++) {
        field_add_batch(d_result, d_a, d_b, n);
    }
    cudaDeviceSynchronize();

    // Benchmark runs
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < benchmark_runs; i++) {
        field_add_batch(d_result, d_a, d_b, n);
    }
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &end);

    double total_ms = get_elapsed_ms(start, end);
    double avg_ms = total_ms / benchmark_runs;
    double ops_per_sec = (n / avg_ms) * 1000.0;

    char throughput_str[64];
    format_throughput(ops_per_sec, throughput_str, sizeof(throughput_str));

    printf("%.3f ms/batch, %s\n", avg_ms, throughput_str);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

void benchmark_gpu_multiplication(size_t n, int warmup_runs, int benchmark_runs) {
    printf("GPU Multiplication (%zu elements): ", n);
    fflush(stdout);

    std::vector<FieldElement> a, b;
    generate_random_elements(a, n);
    generate_random_elements(b, n);

    FieldElement *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, n * sizeof(FieldElement));
    cudaMalloc(&d_b, n * sizeof(FieldElement));
    cudaMalloc(&d_result, n * sizeof(FieldElement));

    cudaMemcpy(d_a, a.data(), n * sizeof(FieldElement), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(FieldElement), cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        field_mul_batch(d_result, d_a, d_b, n);
    }
    cudaDeviceSynchronize();

    // Benchmark
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < benchmark_runs; i++) {
        field_mul_batch(d_result, d_a, d_b, n);
    }
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &end);

    double total_ms = get_elapsed_ms(start, end);
    double avg_ms = total_ms / benchmark_runs;
    double ops_per_sec = (n / avg_ms) * 1000.0;

    char throughput_str[64];
    format_throughput(ops_per_sec, throughput_str, sizeof(throughput_str));

    printf("%.3f ms/batch, %s\n", avg_ms, throughput_str);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

void benchmark_cpu_addition(size_t n, int benchmark_runs) {
    printf("CPU Addition (%zu elements): ", n);
    fflush(stdout);

    std::vector<FieldElement> a, b, result(n);
    generate_random_elements(a, n);
    generate_random_elements(b, n);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int run = 0; run < benchmark_runs; run++) {
        for (size_t i = 0; i < n; i++) {
            field_add_cpu(&result[i], &a[i], &b[i]);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double total_ms = get_elapsed_ms(start, end);
    double avg_ms = total_ms / benchmark_runs;
    double ops_per_sec = (n / avg_ms) * 1000.0;

    char throughput_str[64];
    format_throughput(ops_per_sec, throughput_str, sizeof(throughput_str));

    printf("%.3f ms/batch, %s\n", avg_ms, throughput_str);
}

void benchmark_cpu_multiplication(size_t n, int benchmark_runs) {
    printf("CPU Multiplication (%zu elements): ", n);
    fflush(stdout);

    std::vector<FieldElement> a, b, result(n);
    generate_random_elements(a, n);
    generate_random_elements(b, n);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int run = 0; run < benchmark_runs; run++) {
        for (size_t i = 0; i < n; i++) {
            field_mul_cpu(&result[i], &a[i], &b[i]);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double total_ms = get_elapsed_ms(start, end);
    double avg_ms = total_ms / benchmark_runs;
    double ops_per_sec = (n / avg_ms) * 1000.0;

    char throughput_str[64];
    format_throughput(ops_per_sec, throughput_str, sizeof(throughput_str));

    printf("%.3f ms/batch, %s\n", avg_ms, throughput_str);
}

void benchmark_memory_bandwidth() {
    printf("\nMemory Bandwidth Test:\n");

    const size_t sizes[] = {1000, 10000, 100000, 1000000};
    const int runs = 100;

    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        size_t n = sizes[i];
        size_t bytes = n * sizeof(FieldElement);

        std::vector<FieldElement> data(n);
        generate_random_elements(data, n);

        FieldElement *d_data;
        cudaMalloc(&d_data, bytes);

        // Benchmark host-to-device transfer
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        for (int r = 0; r < runs; r++) {
            cudaMemcpy(d_data, data.data(), bytes, cudaMemcpyHostToDevice);
        }
        cudaDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &end);

        double total_ms = get_elapsed_ms(start, end);
        double avg_ms = total_ms / runs;
        double bandwidth_gbps = (bytes / (avg_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

        printf("  %zu elements: %.3f GB/s (H2D)\n", n, bandwidth_gbps);

        cudaFree(d_data);
    }
}

void benchmark_scalability() {
    printf("\nScalability Test (varying batch sizes):\n");
    printf("%-15s %-15s %-15s\n", "Batch Size", "Time (ms)", "Throughput");
    printf("--------------------------------------------------\n");

    const size_t sizes[] = {100, 1000, 10000, 100000, 1000000};
    const int runs = 10;

    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        size_t n = sizes[i];

        std::vector<FieldElement> a, b;
        generate_random_elements(a, n);
        generate_random_elements(b, n);

        FieldElement *d_a, *d_b, *d_result;
        cudaMalloc(&d_a, n * sizeof(FieldElement));
        cudaMalloc(&d_b, n * sizeof(FieldElement));
        cudaMalloc(&d_result, n * sizeof(FieldElement));

        cudaMemcpy(d_a, a.data(), n * sizeof(FieldElement), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), n * sizeof(FieldElement), cudaMemcpyHostToDevice);

        // Warmup
        field_mul_batch(d_result, d_a, d_b, n);
        cudaDeviceSynchronize();

        // Benchmark
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        for (int r = 0; r < runs; r++) {
            field_mul_batch(d_result, d_a, d_b, n);
        }
        cudaDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &end);

        double total_ms = get_elapsed_ms(start, end);
        double avg_ms = total_ms / runs;
        double ops_per_sec = (n / avg_ms) * 1000.0;

        char throughput_str[64];
        format_throughput(ops_per_sec, throughput_str, sizeof(throughput_str));

        printf("%-15zu %-15.3f %-15s\n", n, avg_ms, throughput_str);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
    }
}

void gpu_vs_cpu_comparison() {
    printf("\nGPU vs CPU Performance Comparison:\n");
    printf("%-15s %-20s %-20s %-15s\n", "Batch Size", "GPU (ms)", "CPU (ms)", "Speedup");
    printf("------------------------------------------------------------------------\n");

    const size_t sizes[] = {1000, 10000, 100000};

    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        size_t n = sizes[i];

        std::vector<FieldElement> a, b;
        generate_random_elements(a, n);
        generate_random_elements(b, n);

        // GPU timing
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

        double gpu_ms = get_elapsed_ms(start, end);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);

        // CPU timing
        std::vector<FieldElement> result(n);
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (size_t j = 0; j < n; j++) {
            field_mul_cpu(&result[j], &a[j], &b[j]);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);

        double cpu_ms = get_elapsed_ms(start, end);
        double speedup = cpu_ms / gpu_ms;

        printf("%-15zu %-20.3f %-20.3f %-15.2fx\n", n, gpu_ms, cpu_ms, speedup);
    }
}

// ====================================================================================
// MAIN BENCHMARK RUNNER
// ====================================================================================

int main() {
    // Seed random number generator
    srand(time(NULL));

    printf("\n");
    printf("========================================\n");
    printf("BN254 CUDA Performance Benchmarks\n");
    printf("========================================\n");
    printf("\n");

    print_device_info();
    printf("\n");

    printf("Configuration:\n");
    printf("  Field Modulus: p = %u\n", FIELD_MODULUS);
    printf("  Limb size: %d bits\n", LIMB_BITS);
    printf("  Number of limbs: %d\n", NUM_LIMBS);
    printf("  Element size: %zu bytes\n", sizeof(FieldElement));
    printf("\n");

    // GPU Benchmarks
    printf("========================================\n");
    printf("GPU Operation Benchmarks\n");
    printf("========================================\n");

    benchmark_gpu_addition(10000, 5, 100);
    benchmark_gpu_addition(100000, 5, 50);
    benchmark_gpu_addition(1000000, 2, 20);

    printf("\n");

    benchmark_gpu_multiplication(10000, 5, 100);
    benchmark_gpu_multiplication(100000, 5, 50);
    benchmark_gpu_multiplication(1000000, 2, 20);

    // CPU Benchmarks
    printf("\n");
    printf("========================================\n");
    printf("CPU Operation Benchmarks\n");
    printf("========================================\n");

    benchmark_cpu_addition(10000, 10);
    benchmark_cpu_multiplication(10000, 10);

    // Comparison
    printf("\n");
    printf("========================================\n");
    gpu_vs_cpu_comparison();

    // Memory bandwidth
    printf("\n");
    printf("========================================\n");
    benchmark_memory_bandwidth();

    // Scalability
    printf("\n");
    printf("========================================\n");
    benchmark_scalability();

    printf("\n");
    printf("========================================\n");
    printf("Benchmarks Complete\n");
    printf("========================================\n");
    printf("\n");

    return 0;
}
