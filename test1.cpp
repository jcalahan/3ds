#include <cstdio>
#include <string>
using namespace std;
#include <gtest/gtest.h>

void print_vector(string what, int* x, int n) {
    printf("vector %s (%d) { ", what.c_str(), n);
    for (int i=0; i<n; ++i) {
        printf("%d%s", x[i], i<(n-1) ? ", " : " ");
    }
    printf("}\n");
}

void print_matrix(string what, int* A, int m, int n) {
    printf("matrix %s (%dx%d) {\n", what.c_str(), m, n);
    for (int i=0; i<m; ++i) {
        printf("  { ");
        for (int j=0; j<n; ++j) {
            // A[i][j] == A[n * i +j];
            printf("%d%s", A[n * i + j], j<(n-1) ? ", " : " ");
        }
        printf("}\n");
    }
    printf("}\n");
}

void matrix_vector_product(int* A, int* x, int* b, int m, int n) {
   for (int i=0; i<m; ++i) { // for each row in A
       for (int j=0; j<n; ++j) { // for each column in B / 'row' in x
           //b[i] += A[i][j] * x[j];
           b[i] += A[n*i+j] * x[j];
       }
   }
}

void matrix_matrix_product(int* A, int* B, int* C, int m, int n, int p) {
   for (int i=0; i<m; ++i) { // for each row in A
       for (int j=0; j<p; ++j) { // for each column in B
           for (int k=0; k<n; ++k) { // for each column in A / row in B
               //C[i][j] += A[i][k] * B[k][j];
               C[p*i+j] += A[n*i+k] * B[p*k+j];
           }
       }
   }
}

void matrix_transpose(int* A, int* At, int m, int n) {
    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            At[m * j + i] = A[n * i + j];
        }
    }
}

void csr_format(int* A, int m, int n, int* values, int* IA, int* IJ); // TODO

void csr_matrix_vector_product(int* Val, int* RowPtr, int* Col, int n, int* d, int* result) {
    for (int k=0; k<n; ++k)
        result[k] = 0;

    for (int i=0; i<n; ++i) {  
        for (int k=RowPtr[i]; k<RowPtr[i+1]; ++k) {  
            result[i] += Val[k]*d[Col[k]];
        }  
    } 
}

void csr_transpose(int* A, int* IA, int* JA, int m, int n, int* At) {
    for (int i=0; i<m*n; ++i) At[i] = 0;

    for (int i=0; i<m; ++i) {
        for (int j=IA[i]; j<IA[i+1]; ++j) {
            int row = i;
            int col = JA[j];
            At[n*col+row] = A[j];
        }
    }
}

TEST(DenseMatrixTest, MatrixVectorProduct) { 
    // http://mathinsight.org/matrix_vector_multiplication

    const int m = 2, n = 3;
    int A[] = { // input matrix
        1, -1, 2, 
        0, -3, 1
    };

    // assert A[column_count * i + j] index scheme
    int i, j;
    i=0, j=0; ASSERT_EQ(1, A[n*i+j]);
    i=0, j=1; ASSERT_EQ(-1, A[n*i+j]);
    i=0, j=2; ASSERT_EQ(2, A[n*i+j]);
    i=1, j=0; ASSERT_EQ(0, A[n*i+j]);
    i=1, j=1; ASSERT_EQ(-3, A[n*i+j]);
    i=1, j=2; ASSERT_EQ(1, A[n*i+j]);

    int x[n] = {2, 1, 0}; // input vector
    int b[m] = {0}; // output vector

    ASSERT_EQ(0, b[0]);
    ASSERT_EQ(0, b[1]);

    matrix_vector_product(A, x, b, m, n); // A * x = b

#ifdef VERBOSE
    print_matrix("A", A, m, n);
    print_vector("x", x, n);
    print_vector("b", b, m);
#endif

    ASSERT_EQ(1, b[0]);
    ASSERT_EQ(-3, b[1]);
}

TEST(DenseMatrixTest, MatrixMatrixProduct) { 
    const int m = 2, n = 3, p = 2;

    int A[] = {
        0, 4, -2,
        -4, -3, 0
    };

    int B[] = {
        0, 1, 
        1, -1,
        2, 3
    };
    
    int C[m*p] = {0};

    matrix_matrix_product(A, B, C, m, n, p); // A * B = C

#ifdef VERBOSE
    print_matrix("A", A, m, n);
    print_matrix("B", B, n, p);
    print_matrix("C", C, m, p);
#endif

    int i, j;
    i=0, j=0; ASSERT_EQ(0, C[p*i+j]);
    i=0, j=1; ASSERT_EQ(-10, C[p*i+j]);
    i=1, j=0; ASSERT_EQ(-3, C[p*i+j]);
    i=1, j=1; ASSERT_EQ(-1, C[p*i+j]);
}

TEST(DenseMatrixTest, MatrixTranspose) { 
    const int m = 2, n = 3;

    int A[] = {
        1, 2, 3,
        4, 5, 6
    };

    int At[m*n];

    matrix_transpose(A, At, m, n);

#ifdef VERBOSE
    print_vector("A", A, m*n);
    print_matrix("A", A, m, n);
    print_matrix("At", At, n, m);
#endif

    int i, j;
    i=0, j=0; ASSERT_EQ(1, At[m*i+j]);
    i=0, j=1; ASSERT_EQ(4, At[m*i+j]);
    i=1, j=0; ASSERT_EQ(2, At[m*i+j]);
    i=1, j=1; ASSERT_EQ(5, At[m*i+j]);
    i=2, j=0; ASSERT_EQ(3, At[m*i+j]);
    i=2, j=1; ASSERT_EQ(6, At[m*i+j]);
}

TEST(SparseMatrixTest, CSR_Construct) { 
}

TEST(SparseMatrixTest, MatrixVectorProduct) { 
    // https://en.wikipedia.org/wiki/Sparse_matrix (section on CSR format)
    // http://www.mathcs.emory.edu/~cheung/Courses/561/Syllabus/3-C/sparse.html

    const int m = 4, n = 4;

    int M[] = {
        0, 0, 0, 0,
        5, 8, 0, 0,
        0, 0, 3, 0,
        0, 6, 0, 0
    };

    // computed w/ dense method for santity check
    int x[n] = {4, 3, 2, 1};
    int b[m] = {0};

    matrix_vector_product(M, x, b, m, n);

#ifdef VERBOSE
    print_matrix("M", M, m, n);
    print_vector("x", x, n);
    print_vector("b", b, m);
#endif

    ASSERT_EQ(0, b[0]);
    ASSERT_EQ(44, b[1]);
    ASSERT_EQ(6, b[2]);
    ASSERT_EQ(18, b[3]);

    // stored and computed w/ Compressed Row Format (CSR)
    int A[] = { 5, 8, 3, 6 }; // non-zero values of M; left-to-right
    int IA[] = { 0, 0, 2, 3, 4 }; // 0th = 0, ..., N-1th = NNZ
    int JA[] = { 0, 1, 2, 1 }; // column index of each element in A

    int bb[m] = {0};
    csr_matrix_vector_product(A, IA, JA, m, x, bb);

#ifdef VERBOSE
    print_vector("result", bb, m);
#endif

    ASSERT_EQ(0, bb[0]);
    ASSERT_EQ(44, bb[1]);
    ASSERT_EQ(6, bb[2]);
    ASSERT_EQ(18, bb[3]);
}

TEST(SparseMatrixTest, MatrixTranspose) { 
    const int m = 4, n = 4;

    int M[] = {
        0, 0, 0, 0,
        5, 8, 0, 0,
        0, 0, 3, 0,
        0, 6, 0, 0
    };

    int A[] = { 5, 8, 3, 6 };
    int IA[] = { 0, 0, 2, 3, 4 };
    int JA[] = { 0, 1, 2, 1 };

    int Mt[n*m];
    matrix_transpose(M, Mt, m, n); // compute w/ dense method for comparison

    int Mtt[n*m];
    csr_transpose(A, IA, JA, m, n, Mtt);

    for (int i=0; i<m*n; ++i) {
        ASSERT_EQ(Mt[i], Mtt[i]);
    }

#ifdef VERBOSE
    print_matrix("M", M, m, n);
    print_matrix("Mt", Mt, n, m);
    print_matrix("Mtt", Mtt, n, m);
#endif

    // TODO: modify csr_transpose to output in csr storage
    int At_expected[] = { 5, 8, 6, 3 };
    int IAt_expected[] = { 0, 1, 3, 4, 4 };
    int JAt_expected[] = { 1, 1, 3, 2 };
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
