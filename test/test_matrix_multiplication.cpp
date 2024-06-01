#include "matrix_multiplication.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <gtest/gtest.h>
#include <mpi.h>

void executeTest(std::vector<std::vector<int>>& A, std::vector<std::vector<int>>& B, std::vector<std::vector<int>>& expected){
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

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((((";
}


// ######################### Source code of multiplyMatrices in src/matrix_mult

TEST(MatrixMultiplicationTest, TestMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    std::vector<std::vector<int>> B = {
        {7, 8},
        {9, 10},
        {11, 12}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    std::vector<std::vector<int>> expected = {
            {58, 64},
            {139, 154}
    };
    executeTest(A, B, expected);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    int results = RUN_ALL_TESTS();
    MPI_Finalize();
    return results;
}
