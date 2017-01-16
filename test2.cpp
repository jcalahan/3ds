#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>

class Matrix {
    int rows, cols;

protected: // disable copies
    Matrix() = default;
    ~Matrix() = default;

    Matrix(Matrix const &) = delete;
    void operator=(Matrix const &x) = delete;

public:
    Matrix(int m, int n) : rows(m), cols(n) {}

    virtual std::vector<int> product(const std::vector<int>& v) const = 0;

    virtual std::shared_ptr<Matrix> product(const Matrix& m) const = 0;

    virtual std::shared_ptr<Matrix> transpose() const = 0;

    virtual int get(int row, int col) const = 0;

    inline int rowCount() const { return rows; }

    inline int columnCount() const { return cols; }
};

class DenseMatrix : public Matrix {
    std::vector<int> data;

public:
    DenseMatrix(int m, int n, std::vector<int>&& d) : Matrix(m, n), data(d) {}
    
    std::vector<int> product(const std::vector<int>& x) const {
        std::vector<int> b(rowCount());
        for (auto i=0; i<rowCount(); ++i) {
            for (auto j=0; j<columnCount(); ++j) {
                b[i] += get(i,j) * x[j];
            }
        }
        return b;
    }

    std::shared_ptr<Matrix> product(const Matrix& B) const {
        const auto& A = *this;
        std::vector<int> C(A.rowCount() * B.columnCount());
        for (auto i=0; i<A.rowCount(); ++i) {
            for (auto j=0; j<B.columnCount(); ++j) {
                for (auto k=0; k<B.rowCount(); ++k) {
                    C[B.columnCount()*i+j] += A.get(i,k) * B.get(k,j);
                }
            }
        }
        return std::make_shared<DenseMatrix>(A.rowCount(), B.columnCount(), std::move(C));
    }

    std::shared_ptr<Matrix> transpose() const {
        const auto& A = *this;
        std::vector<int> At(A.rowCount() * A.columnCount());
        for (auto i=0; i<A.rowCount(); ++i) {
            for (auto j=0; j<A.columnCount(); ++j) {
                At[A.rowCount()*j+i] = A.get(i, j);
            }
        }
        return std::make_shared<DenseMatrix>(A.columnCount(), A.rowCount(), std::move(At));
    }

    int get(int row, int col) const { return data[ columnCount() * row + col ]; }
};

class SparseMatrix : public Matrix {
public:
    SparseMatrix(int m, int n) : Matrix(m, n) {}
};

class CompressedSparseRowMatrix : public SparseMatrix {
    std::vector<int> values; // len NNZ; non-zero values of M left-to-right
    std::vector<int> row_ptr; // 0th = 0, ..., N-1th = NNZ
    std::vector<int> col_idx; // len NNZ; column index of each element in values

public:
    CompressedSparseRowMatrix(int m, int n, const std::vector<int>& d) : SparseMatrix(m, n) {
        auto nnz_total = 0,
             nnz_on_curr_row = 0;

        row_ptr.resize(m+1);
        row_ptr[0] = 0;

        for (auto i=0; i<rowCount(); ++i) {
            nnz_on_curr_row = 0;
            for (auto j=0; j<columnCount(); ++j) {
                const auto val = d[columnCount() * i + j];
                if (val != 0) {
                    ++nnz_total;
                    ++nnz_on_curr_row;
                    values.push_back(val);
                    col_idx.push_back(j);
                }
            }
            row_ptr[i+1] = row_ptr[i] + nnz_on_curr_row;
        }
    }
    
    std::vector<int> product(const std::vector<int>& v) const {
        std::vector<int> b(rowCount());
        for (auto i=0; i<rowCount(); ++i) {  
            for (auto j=row_ptr[i]; j<row_ptr[i+1]; ++j) {
                b[i] += values[j]*v[col_idx[j]];
            }  
        } 
        return b;
    }

    std::shared_ptr<Matrix> product(const Matrix& B) const {
        const auto& A = *this;
        std::vector<int> C(A.rowCount() * B.columnCount());
        for (auto i=0; i<A.rowCount(); ++i) {
            for (auto j=0; j<B.columnCount(); ++j) {
                for (auto k=0; k<B.rowCount(); ++k) {
                    C[B.columnCount()*i+j] += A.get(i,k) * B.get(k,j);
                }
            }
        }
        return std::make_shared<CompressedSparseRowMatrix>(A.rowCount(), B.columnCount(), std::move(C));
    }

    std::shared_ptr<Matrix> transpose() const {
        std::vector<int> At(columnCount()*rowCount());

        for (auto i=0; i<rowCount(); ++i) {  
            for (auto j=row_ptr[i]; j<row_ptr[i+1]; ++j) {
                const auto row = i;
                const auto col = col_idx[j];
                At[rowCount()*col+row] = values[j];
            }  
        } 

        return std::make_shared<CompressedSparseRowMatrix>(columnCount(), rowCount(), std::move(At));
    }

    int get(int row, int col) const {
        if (0 <= row && row <= row_ptr.size()-1) {
            for (auto j=row_ptr[row]; j<row_ptr[row+1]; ++j) {
                if (col == col_idx[j]) {
                    return values[j];
                }
            }
        }
        return 0;
    }
};

class DictionaryOfKeysMatrix : public SparseMatrix {
    typedef std::pair<int, int> Key;
    std::map<Key, int> data;

public:
    DictionaryOfKeysMatrix(int m, int n, const std::vector<int>& d) : SparseMatrix(m, n) {
        for (auto i=0; i<rowCount(); ++i) {
            for (auto j=0; j<columnCount(); ++j) {
                const auto val = d[columnCount() * i + j];
                if (val != 0) {
                    data[std::make_pair(i,j)] = val;
                }
            }
        }
    }
    
    std::vector<int> product(const std::vector<int>& x) const {
        std::vector<int> b(rowCount());
        for (auto i=0; i<rowCount(); ++i) {
            for (auto j=0; j<columnCount(); ++j) {
                b[i] += get(i,j) * x[j];
            }
        }
        return b;
    }

    std::shared_ptr<Matrix> product(const Matrix& B) const {
        const auto& A = *this;
        std::vector<int> C(A.rowCount() * B.columnCount());
        for (auto i=0; i<A.rowCount(); ++i) {
            for (auto j=0; j<B.columnCount(); ++j) {
                for (auto k=0; k<B.rowCount(); ++k) {
                    C[B.columnCount()*i+j] += A.get(i,k) * B.get(k,j);
                }
            }
        }
        return std::make_shared<DictionaryOfKeysMatrix>(A.rowCount(), B.columnCount(), std::move(C));
    }

    std::shared_ptr<Matrix> transpose() const {
        const auto& A = *this;
        std::vector<int> At(A.rowCount() * A.columnCount());
        for (auto i=0; i<A.rowCount(); ++i) {
            for (auto j=0; j<A.columnCount(); ++j) {
                At[A.rowCount()*j+i] = A.get(i, j);
            }
        }
        return std::make_shared<DictionaryOfKeysMatrix>(A.columnCount(), A.rowCount(), std::move(At));
    }

    int get(int row, int col) const {
        const auto itr = data.find(std::make_pair(row,col));
        return itr != data.end() ? itr->second : 0;
    }
};

enum MatrixForm {
    DENSE_MATRIX,
    CSR_MATRIX,
    DOK_MATRIX
};

struct MatrixFactory {
    static std::shared_ptr<Matrix> build(MatrixForm form, int rows, int cols, std::vector<int>&& data) {
        switch (form) {
            case DENSE_MATRIX: return std::make_shared<DenseMatrix>(rows, cols, std::move(data));
            case CSR_MATRIX:   return std::make_shared<CompressedSparseRowMatrix>(rows, cols, data);
            case DOK_MATRIX:   return std::make_shared<DictionaryOfKeysMatrix>(rows, cols, data);
            default:           return nullptr;
        }
    }
};

class MatrixFileIO {
     std::ifstream in;

public:
     MatrixFileIO(const std::string& filename) : in(filename) {
     }

     std::shared_ptr<Matrix> readFile() {
         int m, n, value;
         in >> m >> n;
         std::vector<int> d(m*n);
         for (int i=0; i<m*n; ++i) {
             in >> value;
             d[i] = value;
         }
         return std::make_shared<DenseMatrix>(m, n, std::move(d));
     }
};

class VectorFileIO {
     std::ifstream in;

public:
     VectorFileIO(const std::string& filename) : in(filename) {
     }

     std::vector<int> readFile() {
         int n, value;
         in >> n;
         std::vector<int> d(n);
         for (int i=0; i<n; ++i) {
             in >> value;
             d[i] = value;
         }
         return d;
     }
};

TEST(FileIO, TestOne) { 
    MatrixFileIO fileIO("matrixTestOne.txt");
    auto A = fileIO.readFile();
    ASSERT_EQ(2, A->rowCount());
    ASSERT_EQ(3, A->columnCount());
    
    VectorFileIO vecIO("vectorTestOne.txt");
    std::vector<int> x = vecIO.readFile();
    std::vector<int> b = A->product(x);
    ASSERT_EQ(2, b.size());
    ASSERT_EQ(1, b[0]);
    ASSERT_EQ(-3, b[1]);
}

TEST(DenseMatrix, FactoryConstruction) { 
    auto m = MatrixFactory::build(DENSE_MATRIX, 2,3, {1, -1, 2, 
                                                      0, -3, 1});
    ASSERT_EQ(2, m->rowCount());
    ASSERT_EQ(3, m->columnCount());
    ASSERT_EQ(1,  m->get(0,0));
    ASSERT_EQ(-1, m->get(0,1));
    ASSERT_EQ(2,  m->get(0,2));
    ASSERT_EQ(0,  m->get(1,0));
    ASSERT_EQ(-3, m->get(1,1));
    ASSERT_EQ(1,  m->get(1,2));
}

TEST(DenseMatrix, MatrixVectorProduct) { 
    auto m = MatrixFactory::build(DENSE_MATRIX, 2,3, {1, -1, 2, 
                                                      0, -3, 1});
    std::vector<int> x = {2, 1, 0}; // input vector
    std::vector<int> b = m->product(x);
    ASSERT_EQ(2, b.size());
    ASSERT_EQ(1, b[0]);
    ASSERT_EQ(-3, b[1]);
}

TEST(DenseMatrix, MatrixMatrixProduct) { 
    auto A = MatrixFactory::build(DENSE_MATRIX, 2,3, {0, 4, -2,
                                                      -4, -3, 0});
    auto B = MatrixFactory::build(DENSE_MATRIX, 3,2, {0, 1, 
                                                      1, -1,
                                                      2, 3});
    auto C = A->product(*B);
    ASSERT_EQ(2, C->rowCount());
    ASSERT_EQ(2, C->columnCount());
    ASSERT_EQ(0, C->get(0,0));
    ASSERT_EQ(-10, C->get(0,1));
    ASSERT_EQ(-3, C->get(1,0));
    ASSERT_EQ(-1, C->get(1,1));
}

TEST(DenseMatrix, MatrixTranspose) { 
    auto A = MatrixFactory::build(DENSE_MATRIX, 2,3, {1, 2, 3, 
                                                      4, 5, 6});
    ASSERT_EQ(2, A->rowCount());
    ASSERT_EQ(3, A->columnCount());
    ASSERT_EQ(1, A->get(0,0));
    ASSERT_EQ(2, A->get(0,1));
    ASSERT_EQ(3, A->get(0,2));
    ASSERT_EQ(4, A->get(1,0));
    ASSERT_EQ(5, A->get(1,1));
    ASSERT_EQ(6, A->get(1,2));

    auto At = A->transpose();
    ASSERT_EQ(3, At->rowCount());
    ASSERT_EQ(2, At->columnCount());
    ASSERT_EQ(1, At->get(0,0));
    ASSERT_EQ(4, At->get(0,1));
    ASSERT_EQ(2, At->get(1,0));
    ASSERT_EQ(5, At->get(1,1));
    ASSERT_EQ(3, At->get(2,0));
    ASSERT_EQ(6, At->get(2,1));
}

TEST(CSRMatrix, FactoryConstruction) { 
    auto A = MatrixFactory::build(CSR_MATRIX, 4,4, {0, 0, 0, 0,
                                                    5, 8, 0, 0,
                                                    0, 0, 3, 0,
                                                    0, 6, 0, 0});
    ASSERT_EQ(4, A->rowCount());
    ASSERT_EQ(4, A->columnCount());
    ASSERT_EQ(0, A->get(0,0));
    ASSERT_EQ(0, A->get(0,1));
    ASSERT_EQ(0, A->get(0,2));
    ASSERT_EQ(0, A->get(0,3));
    ASSERT_EQ(5, A->get(1,0));
    ASSERT_EQ(8, A->get(1,1));
    ASSERT_EQ(0, A->get(1,2));
    ASSERT_EQ(0, A->get(1,3));
    ASSERT_EQ(0, A->get(2,0));
    ASSERT_EQ(0, A->get(2,1));
    ASSERT_EQ(3, A->get(2,2));
    ASSERT_EQ(0, A->get(2,3));
    ASSERT_EQ(0, A->get(3,0));
    ASSERT_EQ(6, A->get(3,1));
    ASSERT_EQ(0, A->get(3,2));
    ASSERT_EQ(0, A->get(3,3));
}

TEST(CSRMatrix, MatrixVectorProduct) { 
    auto A = MatrixFactory::build(CSR_MATRIX, 4,4, {0, 0, 0, 0,
                                                    5, 8, 0, 0,
                                                    0, 0, 3, 0,
                                                    0, 6, 0, 0});
    std::vector<int> x = {4, 3, 2, 1}; // input vector
    std::vector<int> b = A->product(x);
    ASSERT_EQ(4, b.size());
    ASSERT_EQ(0, b[0]);
    ASSERT_EQ(44, b[1]);
    ASSERT_EQ(6, b[2]);
    ASSERT_EQ(18, b[3]);
}

TEST(CSRMatrix, MatrixMatrixProduct) { 
    auto A = MatrixFactory::build(DENSE_MATRIX, 2,3, {0, 4, -2,
                                                      -4, -3, 0});
    auto B = MatrixFactory::build(DENSE_MATRIX, 3,2, {0, 1, 
                                                      1, -1,
                                                      2, 3});
    auto C = A->product(*B);
    ASSERT_EQ(2, C->rowCount());
    ASSERT_EQ(2, C->columnCount());
    ASSERT_EQ(0, C->get(0,0));
    ASSERT_EQ(-10, C->get(0,1));
    ASSERT_EQ(-3, C->get(1,0));
    ASSERT_EQ(-1, C->get(1,1));
}

TEST(CSRMatrix, MatrixTranspose) { 
    auto A = MatrixFactory::build(CSR_MATRIX, 2,3, {1, 2, 3,
                                                    4, 5, 6});
    ASSERT_EQ(2, A->rowCount());
    ASSERT_EQ(3, A->columnCount());
    ASSERT_EQ(1, A->get(0,0));
    ASSERT_EQ(2, A->get(0,1));
    ASSERT_EQ(3, A->get(0,2));
    ASSERT_EQ(4, A->get(1,0));
    ASSERT_EQ(5, A->get(1,1));
    ASSERT_EQ(6, A->get(1,2));

    auto At = A->transpose();
    ASSERT_EQ(3, At->rowCount());
    ASSERT_EQ(2, At->columnCount());
    ASSERT_EQ(1, At->get(0,0));
    ASSERT_EQ(4, At->get(0,1));
    ASSERT_EQ(2, At->get(1,0));
    ASSERT_EQ(5, At->get(1,1));
    ASSERT_EQ(3, At->get(2,0));
    ASSERT_EQ(6, At->get(2,1));
}

void transposeTestFromBaseMatrixAPI(std::shared_ptr<Matrix> M) { 
    ASSERT_EQ(2, M->rowCount());
    ASSERT_EQ(3, M->columnCount());
    ASSERT_EQ(1, M->get(0,0));
    ASSERT_EQ(2, M->get(0,1));
    ASSERT_EQ(3, M->get(0,2));
    ASSERT_EQ(4, M->get(1,0));
    ASSERT_EQ(5, M->get(1,1));
    ASSERT_EQ(6, M->get(1,2));

    auto Mt = M->transpose();
    ASSERT_EQ(3, Mt->rowCount());
    ASSERT_EQ(2, Mt->columnCount());
    ASSERT_EQ(1, Mt->get(0,0));
    ASSERT_EQ(4, Mt->get(0,1));
    ASSERT_EQ(2, Mt->get(1,0));
    ASSERT_EQ(5, Mt->get(1,1));
    ASSERT_EQ(3, Mt->get(2,0));
    ASSERT_EQ(6, Mt->get(2,1));
}

TEST(DOKMatrix, FactoryConstruction) { 
    auto m = MatrixFactory::build(DOK_MATRIX, 2,3, {1, -1, 2, 
                                                    0, -3, 1});
    ASSERT_EQ(2, m->rowCount());
    ASSERT_EQ(3, m->columnCount());
    ASSERT_EQ(1,  m->get(0,0));
    ASSERT_EQ(-1, m->get(0,1));
    ASSERT_EQ(2,  m->get(0,2));
    ASSERT_EQ(0,  m->get(1,0));
    ASSERT_EQ(-3, m->get(1,1));
    ASSERT_EQ(1,  m->get(1,2));
}

TEST(DOKMatrix, MatrixVectorProduct) { 
    auto m = MatrixFactory::build(DOK_MATRIX, 2,3, {1, -1, 2, 
                                                    0, -3, 1});
    std::vector<int> x = {2, 1, 0}; // input vector
    std::vector<int> b = m->product(x);
    ASSERT_EQ(2, b.size());
    ASSERT_EQ(1, b[0]);
    ASSERT_EQ(-3, b[1]);
}

TEST(DOKMatrix, MatrixMatrixProduct) { 
    auto A = MatrixFactory::build(DOK_MATRIX, 2,3, {0, 4, -2,
                                                    -4, -3, 0});
    auto B = MatrixFactory::build(DOK_MATRIX, 3,2, {0, 1, 
                                                    1, -1,
                                                    2, 3});
    auto C = A->product(*B);
    ASSERT_EQ(2, C->rowCount());
    ASSERT_EQ(2, C->columnCount());
    ASSERT_EQ(0, C->get(0,0));
    ASSERT_EQ(-10, C->get(0,1));
    ASSERT_EQ(-3, C->get(1,0));
    ASSERT_EQ(-1, C->get(1,1));
}

TEST(DOKMatrix, MatrixTranspose) { 
    auto A = MatrixFactory::build(DOK_MATRIX, 2,3, {1, 2, 3, 
                                                    4, 5, 6});
    ASSERT_EQ(2, A->rowCount());
    ASSERT_EQ(3, A->columnCount());
    ASSERT_EQ(1, A->get(0,0));
    ASSERT_EQ(2, A->get(0,1));
    ASSERT_EQ(3, A->get(0,2));
    ASSERT_EQ(4, A->get(1,0));
    ASSERT_EQ(5, A->get(1,1));
    ASSERT_EQ(6, A->get(1,2));

    auto At = A->transpose();
    ASSERT_EQ(3, At->rowCount());
    ASSERT_EQ(2, At->columnCount());
    ASSERT_EQ(1, At->get(0,0));
    ASSERT_EQ(4, At->get(0,1));
    ASSERT_EQ(2, At->get(1,0));
    ASSERT_EQ(5, At->get(1,1));
    ASSERT_EQ(3, At->get(2,0));
    ASSERT_EQ(6, At->get(2,1));
}

TEST(PolymorphicMatrix, MatrixTranspose) { 
    auto A1 = MatrixFactory::build(DENSE_MATRIX, 2,3, {1, 2, 3,
                                                       4, 5, 6});
    auto A2 = MatrixFactory::build(CSR_MATRIX, 2,3, {1, 2, 3,
                                                     4, 5, 6});
    auto A3 = MatrixFactory::build(DOK_MATRIX, 2,3, {1, 2, 3,
                                                     4, 5, 6});
    transposeTestFromBaseMatrixAPI(A1);
    transposeTestFromBaseMatrixAPI(A2);
    transposeTestFromBaseMatrixAPI(A3);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
