#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>

template<class T>
class Matrix {
    T rows, cols;

protected: // disable copies
    Matrix() = default;
    ~Matrix() = default;

    Matrix(Matrix const &) = delete;
    void operator=(Matrix const &x) = delete;

public:
    Matrix(int m, int n) : rows(m), cols(n) {}

    virtual std::vector<T> product(const std::vector<T>& v) const = 0;

    virtual std::shared_ptr<Matrix> product(const Matrix& m) const = 0;

    virtual std::shared_ptr<Matrix> transpose() const = 0;

    virtual T get(int row, int col) const = 0;

    inline int rowCount() const { return rows; }

    inline int columnCount() const { return cols; }
};

template<class T>
class DenseMatrix : public Matrix<T> {
    std::vector<T> data;

public:
    DenseMatrix(int m, int n, std::vector<T>&& d) : Matrix<T>(m, n), data(d) {}
    
    std::vector<T> product(const std::vector<T>& x) const {
        std::vector<T> b(this->rowCount());
        for (auto i=0; i<this->rowCount(); ++i) {
            for (auto j=0; j<this->columnCount(); ++j) {
                b[i] += get(i,j) * x[j];
            }
        }
        return b;
    }

    std::shared_ptr<Matrix<T>> product(const Matrix<T>& B) const {
        const auto& A = *this;
        std::vector<T> C(A.rowCount() * B.columnCount());
        for (auto i=0; i<A.rowCount(); ++i) {
            for (auto j=0; j<B.columnCount(); ++j) {
                for (auto k=0; k<B.rowCount(); ++k) {
                    C[B.columnCount()*i+j] += A.get(i,k) * B.get(k,j);
                }
            }
        }
        return std::make_shared<DenseMatrix<T>>(A.rowCount(), B.columnCount(), std::move(C));
    }

    std::shared_ptr<Matrix<T>> transpose() const {
        const auto& A = *this;
        std::vector<T> At(A.rowCount() * A.columnCount());
        for (auto i=0; i<A.rowCount(); ++i) {
            for (auto j=0; j<A.columnCount(); ++j) {
                At[A.rowCount()*j+i] = A.get(i, j);
            }
        }
        return std::make_shared<DenseMatrix<T>>(A.columnCount(), A.rowCount(), std::move(At));
    }

    T get(int row, int col) const { return data[ this->columnCount() * row + col ]; }
};

template<class T>
class SparseMatrix : public Matrix<T> {
public:
    SparseMatrix(int m, int n) : Matrix<T>(m, n) {}
};

template<class T>
class CompressedSparseRowMatrix : public SparseMatrix<T> {
    std::vector<T> values; // len NNZ; non-zero values of M left-to-right
    std::vector<T> row_ptr; // 0th = 0, ..., N-1th = NNZ
    std::vector<T> col_idx; // len NNZ; column index of each element in values

public:
    CompressedSparseRowMatrix(int m, int n, const std::vector<T>& d) : SparseMatrix<T>(m, n) {
        auto nnz_total = 0,
             nnz_on_curr_row = 0;

        row_ptr.resize(m+1);
        row_ptr[0] = 0;

        for (auto i=0; i<this->rowCount(); ++i) {
            nnz_on_curr_row = 0;
            for (auto j=0; j<this->columnCount(); ++j) {
                const auto val = d[this->columnCount() * i + j];
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
    
    std::vector<T> product(const std::vector<T>& v) const {
        std::vector<T> b(this->rowCount());
        for (auto i=0; i<this->rowCount(); ++i) {  
            for (auto j=row_ptr[i]; j<row_ptr[i+1]; ++j) {
                b[i] += values[j]*v[col_idx[j]];
            }  
        } 
        return b;
    }

    std::shared_ptr<Matrix<T>> product(const Matrix<T>& B) const {
        const auto& A = *this;
        std::vector<T> C(A.rowCount() * B.columnCount());
        for (auto i=0; i<A.rowCount(); ++i) {
            for (auto j=0; j<B.columnCount(); ++j) {
                for (auto k=0; k<B.rowCount(); ++k) {
                    C[B.columnCount()*i+j] += A.get(i,k) * B.get(k,j);
                }
            }
        }
        return std::make_shared<CompressedSparseRowMatrix<T>>(A.rowCount(), B.columnCount(), std::move(C));
    }

    std::shared_ptr<Matrix<T>> transpose() const {
        std::vector<T> At(this->columnCount()*this->rowCount());

        for (auto i=0; i<this->rowCount(); ++i) {  
            for (auto j=row_ptr[i]; j<row_ptr[i+1]; ++j) {
                const auto row = i;
                const auto col = col_idx[j];
                At[this->rowCount()*col+row] = values[j];
            }  
        } 

        return std::make_shared<CompressedSparseRowMatrix<T>>(this->columnCount(), this->rowCount(), std::move(At));
    }

    T get(int row, int col) const {
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

template<class T>
class DictionaryOfKeysMatrix : public SparseMatrix<T> {
    typedef std::pair<int, int> Key;
    std::map<Key, T> data;

public:
    DictionaryOfKeysMatrix(int m, int n, const std::vector<T>& d) : SparseMatrix<T>(m, n) {
        for (auto i=0; i<this->rowCount(); ++i) {
            for (auto j=0; j<this->columnCount(); ++j) {
                const auto val = d[this->columnCount() * i + j];
                if (val != 0) {
                    data[std::make_pair(i,j)] = val;
                }
            }
        }
    }
    
    std::vector<T> product(const std::vector<T>& x) const {
        std::vector<T> b(this->rowCount());
        for (auto i=0; i<this->rowCount(); ++i) {
            for (auto j=0; j<this->columnCount(); ++j) {
                b[i] += get(i,j) * x[j];
            }
        }
        return b;
    }

    std::shared_ptr<Matrix<T>> product(const Matrix<T>& B) const {
        const auto& A = *this;
        std::vector<T> C(A.rowCount() * B.columnCount());
        for (auto i=0; i<A.rowCount(); ++i) {
            for (auto j=0; j<B.columnCount(); ++j) {
                for (auto k=0; k<B.rowCount(); ++k) {
                    C[B.columnCount()*i+j] += A.get(i,k) * B.get(k,j);
                }
            }
        }
        return std::make_shared<DictionaryOfKeysMatrix<T>>(A.rowCount(), B.columnCount(), std::move(C));
    }

    std::shared_ptr<Matrix<T>> transpose() const {
        const auto& A = *this;
        std::vector<T> At(A.rowCount() * A.columnCount());
        for (auto i=0; i<A.rowCount(); ++i) {
            for (auto j=0; j<A.columnCount(); ++j) {
                At[A.rowCount()*j+i] = A.get(i, j);
            }
        }
        return std::make_shared<DictionaryOfKeysMatrix<T>>(A.columnCount(), A.rowCount(), std::move(At));
    }

    T get(int row, int col) const {
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
    template<class T>
    static std::shared_ptr<Matrix<T>> build(MatrixForm form, int rows, int cols, std::vector<T>&& data) {
        switch (form) {
            case DENSE_MATRIX: return std::make_shared<DenseMatrix<T>>(rows, cols, std::move(data));
            case CSR_MATRIX:   return std::make_shared<CompressedSparseRowMatrix<T>>(rows, cols, data);
            case DOK_MATRIX:   return std::make_shared<DictionaryOfKeysMatrix<T>>(rows, cols, data);
            default:           return nullptr;
        }
    }
};

template<class T>
class MatrixFileIO {
public:
     virtual std::shared_ptr<Matrix<T>> readFile() = 0;
};

template<class T>
class DenseMatrixFileIO : public MatrixFileIO<T> {
     std::ifstream in;

public:
     DenseMatrixFileIO(const std::string& filename) : in(filename) {}

     std::shared_ptr<Matrix<T>> readFile() {
         int m, n;
         in >> m >> n;
         const auto size = m*n;
         std::vector<T> d(size);
         for (int i=0; i<size; ++i) {
             T value;
             in >> value;
             d[i] = value;
         }
         return std::make_shared<DenseMatrix<T>>(m, n, std::move(d));
     }
};

template<class T>
class SparseMatrixFileIO : public MatrixFileIO<T> {
     std::ifstream in;

public:
     SparseMatrixFileIO(const std::string& filename) : in(filename) {}

     std::shared_ptr<Matrix<T>> readFile() {
         int m, n;
         in >> m >> n;
         const auto size = m*n;
         std::vector<T> d(size);
         for (int i=0; i<size; ++i) {
             T value;
             in >> value;
             d[i] = value;
         }
         return std::make_shared<DictionaryOfKeysMatrix<T>>(m, n, std::move(d));
     }
};

template<class T>
class VectorFileIO {
     std::ifstream in;

public:
     VectorFileIO(const std::string& filename) : in(filename) {
     }

     std::vector<int> readFile() {
         int n;
         in >> n;
         std::vector<T> d(n);
         for (int i=0; i<n; ++i) {
             T value;
             in >> value;
             d[i] = value;
         }
         return d;
     }
};

TEST(FileIO, DenseMatVecProd) { 
    DenseMatrixFileIO<int> fileIO("matrixTestOne.txt");
    auto A = fileIO.readFile();
    ASSERT_EQ(2, A->rowCount());
    ASSERT_EQ(3, A->columnCount());
    
    VectorFileIO<int> vecIO("vectorTestOne.txt");
    std::vector<int> x = vecIO.readFile();
    std::vector<int> b = A->product(x);
    ASSERT_EQ(2, b.size());
    ASSERT_EQ(1, b[0]);
    ASSERT_EQ(-3, b[1]);
}

TEST(FileIO, SparseMatVecProd) { 
    SparseMatrixFileIO<int> fileIO("matrixTestOne.txt");
    auto A = fileIO.readFile();
    ASSERT_EQ(2, A->rowCount());
    ASSERT_EQ(3, A->columnCount());
    
    VectorFileIO<int> vecIO("vectorTestOne.txt");
    std::vector<int> x = vecIO.readFile();
    std::vector<int> b = A->product(x);
    ASSERT_EQ(2, b.size());
    ASSERT_EQ(1, b[0]);
    ASSERT_EQ(-3, b[1]);
}

TEST(DenseMatrix, FactoryConstruction) { 
    auto m = MatrixFactory::build<int>(DENSE_MATRIX, 2,3, {1, -1, 2, 
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
    auto m = MatrixFactory::build<int>(DENSE_MATRIX, 2,3, {1, -1, 2, 
                                                           0, -3, 1});
    const std::vector<int> x = {2, 1, 0}; // input vector
    auto b = m->product(x);
    ASSERT_EQ(2, b.size());
    ASSERT_EQ(1, b[0]);
    ASSERT_EQ(-3, b[1]);
}

TEST(DenseMatrix, MatrixVectorProductFloat) { 
    auto m = MatrixFactory::build<float>(DENSE_MATRIX, 2,3, {1.0, -1.0, 2.0, 
                                                             0.0, -3.0, 1.0});
    const std::vector<float> x = {2.0, 1.0, 0.0}; // input vector
    auto b = m->product(x);
    ASSERT_EQ(2.0, b.size());
    ASSERT_EQ(1.0, b[0]);
    ASSERT_EQ(-3.0, b[1]);
}

TEST(DenseMatrix, MatrixVectorProductDouble) { 
    auto m = MatrixFactory::build<double>(DENSE_MATRIX, 2,3, {1.0, -1.0, 2.0, 
                                                              0.0, -3.0, 1.0});
    const std::vector<double> x = {2.0, 1.0, 0.0}; // input vector
    auto b = m->product(x);
    ASSERT_EQ(2.0, b.size());
    ASSERT_EQ(1.0, b[0]);
    ASSERT_EQ(-3.0, b[1]);
}

TEST(DenseMatrix, MatrixMatrixProduct) { 
    auto A = MatrixFactory::build<int>(DENSE_MATRIX, 2,3, {0, 4, -2,
                                                           -4, -3, 0});
    auto B = MatrixFactory::build<int>(DENSE_MATRIX, 3,2, {0, 1, 
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
    auto A = MatrixFactory::build<int>(DENSE_MATRIX, 2,3, {1, 2, 3, 
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
    auto A = MatrixFactory::build<int>(CSR_MATRIX, 4,4, {0, 0, 0, 0,
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
    auto A = MatrixFactory::build<int>(CSR_MATRIX, 4,4, {0, 0, 0, 0,
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
    auto A = MatrixFactory::build<int>(DENSE_MATRIX, 2,3, {0, 4, -2,
                                                           -4, -3, 0});
    auto B = MatrixFactory::build<int>(DENSE_MATRIX, 3,2, {0, 1, 
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
    auto A = MatrixFactory::build<int>(CSR_MATRIX, 2,3, {1, 2, 3,
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

template<class T>
void transposeTestFromBaseMatrixAPI(std::shared_ptr<Matrix<T>> M) { 
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
    auto m = MatrixFactory::build<int>(DOK_MATRIX, 2,3, {1, -1, 2, 
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
    auto m = MatrixFactory::build<int>(DOK_MATRIX, 2,3, {1, -1, 2, 
                                                         0, -3, 1});
    std::vector<int> x = {2, 1, 0}; // input vector
    std::vector<int> b = m->product(x);
    ASSERT_EQ(2, b.size());
    ASSERT_EQ(1, b[0]);
    ASSERT_EQ(-3, b[1]);
}

TEST(DOKMatrix, MatrixMatrixProduct) { 
    auto A = MatrixFactory::build<int>(DOK_MATRIX, 2,3, {0, 4, -2,
                                                         -4, -3, 0});
    auto B = MatrixFactory::build<int>(DOK_MATRIX, 3,2, {0, 1, 
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
    auto A = MatrixFactory::build<int>(DOK_MATRIX, 2,3, {1, 2, 3, 
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
    auto A1 = MatrixFactory::build<int>(DENSE_MATRIX, 2,3, {1, 2, 3,
                                                            4, 5, 6});
    auto A2 = MatrixFactory::build<int>(CSR_MATRIX, 2,3, {1, 2, 3,
                                                          4, 5, 6});
    auto A3 = MatrixFactory::build<int>(DOK_MATRIX, 2,3, {1, 2, 3,
                                                          4, 5, 6});
    transposeTestFromBaseMatrixAPI<int>(A1);
    transposeTestFromBaseMatrixAPI<int>(A2);
    transposeTestFromBaseMatrixAPI<int>(A3);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
