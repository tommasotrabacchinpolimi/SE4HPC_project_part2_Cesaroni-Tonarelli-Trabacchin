#include "matrix_multiplication.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <gtest/gtest.h>
#include <mpi.h>

/*
 * Method to automatically execute tests
 */
std::vector<std::vector<int>> executeTest(std::vector<std::vector<int>>& A, std::vector<std::vector<int>>& B){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rowsA, colsA, rowsB, colsB;
    rowsA = A.size();
    colsA = A[0].size();
    rowsB = B.size();
    colsB = B[0].size();

    MPI_Bcast(&rowsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rowsB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colsB, 1, MPI_INT, 0, MPI_COMM_WORLD);


    if (rank != 0) {
        A.resize(rowsA, std::vector<int>(colsA));
    }
    for (int i = 0; i < rowsA; ++i) {
        MPI_Bcast(A[i].data(), colsA, MPI_INT, 0, MPI_COMM_WORLD);
    }


    if (rank != 0) {
        B.resize(rowsB, std::vector<int>(colsB));
    }
    for (int i = 0; i < rowsB; ++i) {
        MPI_Bcast(B[i].data(), colsB, MPI_INT, 0, MPI_COMM_WORLD);
    }

    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));
    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    return C;
}


// ######################### Source code of multiplyMatrices in src/matrix_mult

/*
 * Tests copied form SE4HPC_project_part1_Cesaroni-Tonarelli-Trabacchin
 */
TEST(MatrixMultiplicationTest, TestAssociativePropertyMatrices) {
    int n = 10;
    std::srand((unsigned)std::time(NULL));
    std::vector<std::vector<int>> A(n, std::vector<int>(n, 0));
    std::vector<std::vector<int>> B(n, std::vector<int>(n, 0));
    std::vector<std::vector<int>> C(n, std::vector<int>(n, 0));
    std::vector<std::vector<int>> temp(n, std::vector<int>(n, 0));
    std::vector<std::vector<int>> result1(n, std::vector<int>(n, 0));
    std::vector<std::vector<int>> result2(n, std::vector<int>(n, 0));

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            A[i][j] = std::rand() % 10;
            B[i][j] = std::rand() % 10;
            C[i][j] = std::rand() % 10;
        }
    }

    // (AB)C
    temp = executeTest(A, B);
    result1 = executeTest(temp, C);
    // A(BC)
    temp = executeTest(B, C);
    result2 = executeTest(A, temp);

    ASSERT_EQ(result1, result2) << "Associativity Matrix multiplication test failed! :(((()";
}

TEST(MatrixMultiplicationTest, TestIdentityElementMatrices) {
    int n = 9;
    std::srand((unsigned)std::time(NULL));
    std::vector<std::vector<int>> A(n, std::vector<int>(n, 0));
    std::vector<std::vector<int>> I(n, std::vector<int>(n, 0));
    std::vector<std::vector<int>> result1(n, std::vector<int>(n, 0));
    std::vector<std::vector<int>> result2(n, std::vector<int>(n, 0));
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            A[i][j] = std::rand() % 10;
            if(i==j){
                I[i][j] = 1;
            }
        }
    }
    // AI
    result1 = executeTest(A, I);
    // IA
    result2 = executeTest(I, A);

    ASSERT_EQ(result1, result2) << "Identity Element Matrix multiplication test failed! :(((()";
}

TEST(MatrixMultiplicationTest, TestNullElementMatrices) {
    int n = 8;
    std::srand((unsigned)std::time(NULL));
    std::vector<std::vector<int>> A(n, std::vector<int>(n, 0));
    std::vector<std::vector<int>> zero(n, std::vector<int>(n, 0));
    std::vector<std::vector<int>> result1(n, std::vector<int>(n, 0));
    std::vector<std::vector<int>> result2(n, std::vector<int>(n, 0));

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            A[i][j] = std::rand() % 10;
        }
    }

    result1 = executeTest(A, zero);

    ASSERT_EQ(result1, zero) << "Null Element Matrix multiplication test failed! :(((()";
}

TEST(MatrixMultiplicationTest, TestRandom1Matrices) {
    int n = 5;
    std::vector<std::vector<int>> A(n, std::vector<int>(n, 0));
    std::vector<std::vector<int>> B(n, std::vector<int>(n, 0));
    std::vector<std::vector<int>> C(n, std::vector<int>(n, 0));

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            A[i][j]=i+1;
            if(i==j){
                B[i][j]=1;
            }
        }
    }

    C = executeTest(A, B);

    ASSERT_EQ(C, A) << "Random1 Element Matrix multiplication test failed! :(((()";
}

TEST(MatrixMultiplicationTest, TestRandom2Matrices) {
    int n = 8;
    std::vector<std::vector<int>> A(n, std::vector<int>(n, 0));
    std::vector<std::vector<int>> B(n, std::vector<int>(n, 0));
    std::vector<std::vector<int>> C(n, std::vector<int>(n, 0));

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            A[i][j]=j+1;
            if(i==j){
                B[i][j]=1;
            }
        }
    }

    C = executeTest(A, B);

    ASSERT_EQ(C, A) << "Random2 Element Matrix multiplication test failed! :(((()";
}

TEST(MatrixMultiplicationTest, TestRandomRectangularMatrices) {
    size_t rowA = 8;
    size_t colA = 5;
    size_t colB = 8;
    std::srand((unsigned)std::time(NULL));
    std::vector<std::vector<int>> A(rowA, std::vector<int>(colA, 0));
    std::vector<std::vector<int>> B(colA, std::vector<int>(colB, 0));
    std::vector<std::vector<int>> C(rowA, std::vector<int>(colB, 0));
    std::vector<std::vector<int>> expected(rowA, std::vector<int>(colB, 0));

    for(int i = 0; i < rowA; i++){
        for(int j = 0; j < colA; j++){
            A[i][j] = std::rand();
        }
    }
    for(int i = 0; i < colA; i++){
        for(int j = 0; j < colB; j++){
            if(i == j){
                B[i][j] = 1;
            }
        }
    }
    for(int i = 0; i < rowA; i++){
        for(int j = 0; j < colB; j++){
            if(j < colA){
                expected[i][j] = A[i][j];
            }
        }
    }

    C = executeTest(A, B);

    ASSERT_EQ(C, expected) << "Rectangular Matrix multiplication test failed! :(((()";
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    int results = RUN_ALL_TESTS();
    MPI_Finalize();
    return results;
}
