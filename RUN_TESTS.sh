#!/bin/bash
# Quick test runner for BN254 CUDA project

cd /home/raham/cuda/bn254_cuda

echo "========================================"
echo "BN254 CUDA - Test Runner"
echo "========================================"
echo ""
echo "Which test would you like to run?"
echo ""
echo "1) Basic Tests (12 tests, ~1 second)"
echo "2) Vector Tests (9 tests, ~3 seconds)"
echo "3) Performance Tests (6 benchmarks, ~10 seconds)"
echo "4) All Tests (runs all of the above)"
echo "5) Exit"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "Running Basic Tests..."
        echo "========================================"
        ./bin/test_basic
        ;;
    2)
        echo ""
        echo "Running Vector Tests..."
        echo "========================================"
        ./bin/test_vectors
        ;;
    3)
        echo ""
        echo "Running Performance Tests..."
        echo "========================================"
        ./bin/test_performance
        ;;
    4)
        echo ""
        echo "Running ALL Tests..."
        echo "========================================"
        ./bin/test_basic
        echo ""
        echo "========================================"
        ./bin/test_vectors
        echo ""
        echo "========================================"
        ./bin/test_performance
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
