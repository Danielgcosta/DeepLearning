/* 
 * File: CUDAMatrix.h
 * 
 * Data structure for dense matrix storage and manipulation in gpu using
 * cuBLAS library.
 * 
 * Author: Eder Perez
 */

#ifndef CUDAMATRIX_H
#define	CUDAMATRIX_H

#include <iomanip>
#include <algorithm>
#include <vector>
#include <curand.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "CUDAErrorMessage.h"
#include "CUDAVector.h"


namespace gpu
{

template<typename T> class CUDAMatrix
{
    public:
        
        /**
         * Create a matrix on gpu with the given dimensions. The matrix values
         * must be filled using the setData method before using it.
         * WARNING: All CUDAMatrix matrices store data in column major order.
         * 
         * @param handle A valid cuBLAS context.
         * @param rows Number of rows.
         * @param columns Number of columns.
         */
        CUDAMatrix( const cublasHandle_t& handle, int rows, int columns );

        /**
         * Create a matrix on gpu with the given dimensions and buffer data.
         * 
         * @param rows Number of rows.
         * @param columns Number of columns.
         * @param buffer Buffer containing data in column major order.
         * @param rowMajor If true, the buffer represents a matrix in row major
         * order and a conversion must be carried out and this operation is more slow.
         */
        CUDAMatrix( const cublasHandle_t& handle, int rows, int columns, const std::vector<T>& buffer, bool rowMajor );
        
        /**
         * Copy constructor.
         */
        CUDAMatrix( const CUDAMatrix& copy );

        /**
         * Release device memory
         */
        ~CUDAMatrix();
        
        /**
         * Get the number of rows.
         */
        int rows() const;
        
        /**
         * Get the number of columns.
         */
        int cols() const;
        
        /**
         * Set matrix values.
         * 
         * @param buffer Buffer containing data in row major order.
         * @param rowMajor If true, the buffer represents a matrix in row major
         * order and a conversion must be carried out and this operation is more slow.
         */
        void setData( const std::vector<T>& buffer, bool rowMajor );
        
        /**
         * Retrieve matrix data.
         * 
         * @param rowMajor If true, the buffer represents a matrix in row major
         * order and a conversion must be carried out and this operation is more slow.
         */
        void getData( std::vector<T>& buffer, bool rowMajor ) const;
        
        /**
         * Get the pointer to device allocated memory.
         */
        T* getDevicePtr();
        const T* getDevicePtr() const;
        
        /**
         * Matrix-vector multiplication.
         * 
         * v = a*(m*u) + b*v
         * 
         * In the second version, a = 1.0 and b = 0.0.
         */
        static void mult( T a, const CUDAMatrix& m, bool transpose, const CUDAVector<T>& u, T b, CUDAVector<T>& v );
        static void mult( const CUDAMatrix& m, bool transpose, const CUDAVector<T>& u, CUDAVector<T>& v );
        
        /**
         * Matrix-matrix multiplication.
         * 
         * m3 = a*m1*m2 + b*m3
         * 
         * In the second version, a = 1.0 and b = 0.0.
         */
        static void mult( T a, const CUDAMatrix& m1, bool transposeM1, const CUDAMatrix& m2, bool transposeM2, T b, CUDAMatrix& m3 );
        static void mult( const CUDAMatrix& m1, bool transposeM1, const CUDAMatrix& m2, bool transposeM2, CUDAMatrix& m3 );
        
        /**
         * Matrix-matrix Hadamard product (elementwise multiplication)
         */
        static void hadamard( const CUDAMatrix& m1, const CUDAMatrix& m2, CUDAMatrix& m3 );
        
        /**
         * Matrix-matrix addition and subtraction.
         * 
         * m3 = a*m1 + b*m2
         * 
         * In the second version, a = 1.0 and b = 1.0.
         * In sub, a = 1.0 and b = -1.0.
         */
        static void add( T a, const CUDAMatrix& m1, bool transposeM1, T b, const CUDAMatrix& m2, bool transposeM2, CUDAMatrix& m3 );
        static void add( const CUDAMatrix& m1, bool transposeM1, const CUDAMatrix& m2, bool transposeM2, CUDAMatrix& m3 );
        static void sub( const CUDAMatrix& m1, bool transposeM1, const CUDAMatrix& m2, bool transposeM2, CUDAMatrix& m3 );
        
        /**
         * Computes the Euclidean norm of matrix.
         */
        static T norm( const CUDAMatrix& m );
        
        /**
         * Scale a matrix.
         */
        static void scale( T s, CUDAMatrix& m );
        
        /**
         * Copy the values of a matrix into another. I matrices dimensions don't
         * match, copy either a submatrix of src or the entire src into dst.
         */
        static void copy( const CUDAMatrix& src, CUDAMatrix& dst );
        
        /**
         * Fill columns and rows with values.
         */
        static void fillCol( CUDAMatrix& m, int col, T value );
        static void fillCol( CUDAMatrix& m, int col, const std::vector<T>& values );
        static void fillRow( CUDAMatrix& m, int row, T value );
        static void fillRow( CUDAMatrix& m, int row, const std::vector<T>& values );
        
        /**
         * Fill matrix with uniform pseudo-random values in range (0, 1].
         */
        static void fillRandom( CUDAMatrix& m );
        
        /**
         * Print matrix values in standard output.
         */
        static void print( const CUDAMatrix& m );
        
        /**
         * Overloading attribution operator. It copies all contents of m to @this.
         * If m and this has different size, @this is reallocated.
         */
        CUDAMatrix& operator=( const CUDAMatrix& m );

    private:
        
        /**
         * Wrapper to cublas functions (in order to make template work).
         */
        static cublasStatus_t cublasCopy( cublasHandle_t handle, int n, const T* x, int incx, T* y, int incy );        
        static cublasStatus_t cublasgemv( cublasHandle_t handle, cublasOperation_t trans, int m, int n, const T* alpha, const T* A, int lda, const T* x, int incx, const T* beta, T* y, int incy );
        static cublasStatus_t cublasgemm( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const T* alpha, const T* A, int lda, const T* B, int ldb, const T* beta, T* C, int ldc );
        static cublasStatus_t cublasgeam( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const T* alpha, const T* A, int lda, const T* beta, const T* B, int ldb, T* C, int ldc );
        static cublasStatus_t cublasnrm2( cublasHandle_t handle, int n, const T* x, int incx, T* result );
        static cublasStatus_t cublasscal( cublasHandle_t handle, int n, const T* alpha, T* x, int incx );
        
        
        cublasHandle_t mCUBLASHandle;
        int mRows;
        int mCols;
        T* mDevicePtr;
};


#define IDX2C(i,j,ld) (((j)*(ld))+(i))

template<typename T>
CUDAMatrix<T>::CUDAMatrix( const cublasHandle_t& handle, int rows, int columns ) :
    mCUBLASHandle(handle),
    mRows(rows),
    mCols(columns),
    mDevicePtr(0)
{
    // Allocates memory on device
    cudaError_t cudaStatus = cudaMalloc( (void**)&mDevicePtr, mRows*mCols*sizeof(T) );
    if (cudaStatus != cudaSuccess)
    {
        CUDAErrorMessage::printErrorMessage( cudaStatus, std::string(__FILE__), __LINE__ );
        mRows = 0;
        mCols = 0;
    }
}


template<typename T>
CUDAMatrix<T>::CUDAMatrix( const cublasHandle_t& handle, int rows, int columns, const std::vector<T>& buffer, bool rowMajor ) :
    CUDAMatrix( handle, rows, columns )
{
    setData( buffer, rowMajor );
}


template<typename T>
CUDAMatrix<T>::CUDAMatrix( const CUDAMatrix& copy ) :
    CUDAMatrix( copy.mCUBLASHandle, copy.mRows, copy.mCols )
{
    // Allocates memory on device
    CUDAMatrix::copy( copy, *this );
}


template<typename T>
CUDAMatrix<T>::~CUDAMatrix()
{
    // Release device memory
    cudaFree(mDevicePtr);
}


template<typename T>
int CUDAMatrix<T>::rows() const
{
    return mRows;
}


template<typename T>
int CUDAMatrix<T>::cols() const
{
    return mCols;
}


template<typename T>
void CUDAMatrix<T>::setData( const std::vector<T>& buffer, bool rowMajor )
{
    if ( buffer.size() < static_cast<size_t>( mRows*mCols ) )
    {
        return;
    }
    
    T* b = 0;
    std::vector<T> colMajorBuffer;

    // If conversion to column major must be executed...
    if ( rowMajor )
    {
        colMajorBuffer.resize( buffer.size() );
        for ( int i = 0; i < mRows; i++ )
        {
            for ( int j = 0; j < mCols; j++ )
            {
                colMajorBuffer[IDX2C(i, j, mRows)] = buffer[i*mCols + j];
            }
        }
        
        b = colMajorBuffer.data();
    }
    else
    {
        b = const_cast<T*>( buffer.data() );
    }
    
    cublasStatus_t status = cublasSetMatrix( mRows, mCols, sizeof(T), b, mRows, mDevicePtr, mRows );
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        CUDAErrorMessage::printErrorMessage( status, __FILE__, __LINE__ );
    }
}


template<typename T>
void CUDAMatrix<T>::getData( std::vector<T>& buffer, bool rowMajor ) const
{
    buffer.resize( mRows*mCols );
    
    cublasStatus_t status = cublasGetMatrix( mRows, mCols, sizeof(T), mDevicePtr, mRows, buffer.data(), mRows);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        CUDAErrorMessage::printErrorMessage( status, __FILE__, __LINE__ );
        buffer.clear();
        return;
    }

    // Convert to row major if necessary
    if ( rowMajor )
    {
        if ( mRows == mCols )
        {
            for ( int i = 0; i < mRows; i++ )
            {
                for ( int j = 0; j < i; j++ )
                {
                    std::swap( buffer[i*mRows + j], buffer[j*mRows + i] );
                }
            }
        }
        else
        {
            std::vector<T> rowMajorBuffer;
            rowMajorBuffer.resize( buffer.size() );
            for ( int i = 0; i < mRows; i++ )
            {
                for ( int j = 0; j < mCols; j++ )
                {
                    rowMajorBuffer[i*mCols + j] = buffer[IDX2C(i, j, mRows)];
                }
            }

            std::swap(rowMajorBuffer, buffer);
        }
    }
}


template<typename T>
T* CUDAMatrix<T>::getDevicePtr()
{
    return mDevicePtr;
}


template<typename T>
const T* CUDAMatrix<T>::getDevicePtr() const
{
    return mDevicePtr;
}


template<typename T>
void CUDAMatrix<T>::mult( T a, const CUDAMatrix& m, bool transpose, const CUDAVector<T>& u, T b, CUDAVector<T>& v )
{
    // Check dimensions
    int mRow = transpose ? m.mCols : m.mRows;
    int mCol = transpose ? m.mRows : m.mCols;
    if ( mCol != u.dim() || mRow != v.dim() )
    {
        std::cerr << "Checking dimensions failed at " << __FILE__ << ":" << __LINE__ << "\n";
        return;
    }

    cublasOperation_t trans = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasStatus_t cublasStatus = CUDAMatrix::cublasgemv( m.mCUBLASHandle, trans, m.mRows, m.mCols, &a, m.mDevicePtr, m.mRows, u.mDevicePtr, 1, &b, v.mDevicePtr, 1 );
    
    if ( cublasStatus != CUBLAS_STATUS_SUCCESS )
    {
        CUDAErrorMessage::printErrorMessage( cublasStatus, __FILE__, __LINE__ );
    }
}


template<typename T>
void CUDAMatrix<T>::mult( const CUDAMatrix& m, bool transpose, const CUDAVector<T>& u, CUDAVector<T>& v )
{
    mult( 1.0, m, transpose, u, 0.0, v );
}


template<typename T>
void CUDAMatrix<T>::mult( T a, const CUDAMatrix& m1, bool transposeM1, const CUDAMatrix& m2, bool transposeM2, T b, CUDAMatrix& m3 )
{
    cublasOperation_t transM1 = transposeM1 ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transM2 = transposeM2 ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    int m1Rows = transposeM1 ? m1.mCols : m1.mRows;
    int m1Cols = transposeM1 ? m1.mRows : m1.mCols;
    int m2Rows = transposeM2 ? m2.mCols : m2.mRows;
    int m2Cols = transposeM2 ? m2.mRows : m2.mCols;
    
    // Check dimensions
    if ( m1Cols != m2Rows || m3.mRows != m1Rows || m3.mCols != m2Cols )
    {
        std::cerr << "Checking dimensions failed at " << __FILE__ << ":" << __LINE__ << "\n";
        return;
    }

    cublasStatus_t cublasStatus = CUDAMatrix::cublasgemm( m1.mCUBLASHandle, transM1, transM2, m3.mRows, m3.mCols, m1Cols, &a, m1.mDevicePtr, m1.mRows, m2.mDevicePtr, m2.mRows, &b, m3.mDevicePtr, m3.mRows );

    if ( cublasStatus != CUBLAS_STATUS_SUCCESS )
    {
        CUDAErrorMessage::printErrorMessage( cublasStatus, __FILE__, __LINE__ );
    }
}


template<typename T>
void CUDAMatrix<T>::mult( const CUDAMatrix& m1, bool transposeM1, const CUDAMatrix& m2, bool transposeM2, CUDAMatrix& m3 )
{
    mult( 1.0, m1, transposeM1, m2, transposeM2, 0.0, m3 );
}


void gpuHadamard( int rows, int cols, const float* m1, const float* m2, float* m3 );
void gpuHadamard( int rows, int cols, const double* m1, const double* m2, double* m3 );
template<typename T>
void CUDAMatrix<T>::hadamard( const CUDAMatrix& m1, const CUDAMatrix& m2, CUDAMatrix& m3 )
{
    if ( m1.rows() != m2.rows() || m1.cols() != m2.cols() ||
         m1.rows() != m3.rows() || m1.cols() != m3.cols() )
    {
        std::cerr << "Checking dimensions failed at " << __FILE__ << ":" << __LINE__ << "\n";
        return;
    }
    
    gpuHadamard( m1.rows(), m1.cols(), m1.mDevicePtr, m2.mDevicePtr, m3.mDevicePtr );
}


template<typename T>
void CUDAMatrix<T>::add( T a, const CUDAMatrix& m1, bool transposeM1, T b, const CUDAMatrix& m2, bool transposeM2, CUDAMatrix& m3 )
{
    cublasOperation_t transM1 = transposeM1 ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transM2 = transposeM2 ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    int m1Rows = transposeM1 ? m1.mCols : m1.mRows;
    int m1Cols = transposeM1 ? m1.mRows : m1.mCols;
    int m2Rows = transposeM2 ? m2.mCols : m2.mRows;
    int m2Cols = transposeM2 ? m2.mRows : m2.mCols;
    
    // Check dimensions
    if ( m1Rows != m2Rows || m1Cols != m2Cols || m3.mRows != m1Rows || m3.mCols != m1Cols )
    {
        std::cerr << "Checking dimensions failed at " << __FILE__ << ":" << __LINE__ << "\n";
        return;
    }

    cublasStatus_t cublasStatus = CUDAMatrix::cublasgeam( m1.mCUBLASHandle, transM1, transM2, m3.mRows, m3.mCols, &a, m1.mDevicePtr, m1.mRows, &b, m2.mDevicePtr, m2.mRows, m3.mDevicePtr, m3.mRows );

    if ( cublasStatus != CUBLAS_STATUS_SUCCESS )
    {
        CUDAErrorMessage::printErrorMessage( cublasStatus, __FILE__, __LINE__ );
    }
}


template<typename T>
void CUDAMatrix<T>::add( const CUDAMatrix& m1, bool transposeM1, const CUDAMatrix& m2, bool transposeM2, CUDAMatrix& m3 )
{
    add( 1.0, m1, transposeM1, 1.0, m2, transposeM2, m3 );
}


template<typename T>
void CUDAMatrix<T>::sub( const CUDAMatrix& m1, bool transposeM1, const CUDAMatrix& m2, bool transposeM2, CUDAMatrix& m3 )
{
    add( 1.0, m1, transposeM1, -1.0, m2, transposeM2, m3 );
}


template<typename T>
T CUDAMatrix<T>::norm( const CUDAMatrix& m )
{
    T result = 0;
    cublasStatus_t cublasStatus = CUDAMatrix::cublasnrm2( m.mCUBLASHandle, m.mRows*m.mCols, m.mDevicePtr, 1, &result );
    
    if ( cublasStatus != CUBLAS_STATUS_SUCCESS )
    {
        CUDAErrorMessage::printErrorMessage( cublasStatus, __FILE__, __LINE__ );
    }
    
    return result;
}


template<typename T>
void CUDAMatrix<T>::scale( T s, CUDAMatrix& m )
{
    cublasStatus_t cublasStatus = CUDAMatrix::cublasscal( m.mCUBLASHandle, m.mRows*m.mCols, &s, m.mDevicePtr, 1 );

    if ( cublasStatus != CUBLAS_STATUS_SUCCESS )
    {
        CUDAErrorMessage::printErrorMessage( cublasStatus, __FILE__, __LINE__ );
    }
}


template<typename T>
void CUDAMatrix<T>::copy( const CUDAMatrix& src, CUDAMatrix& dst )
{
    // If dimensions are the same copy using cublas copy function
    if ( src.mRows == dst.mRows && src.mCols == dst.mCols )
    {
        cublasStatus_t cublasStatus = CUDAMatrix::cublasCopy( src.mCUBLASHandle, src.mRows*src.mCols, src.mDevicePtr, 1, dst.mDevicePtr, 1);

        if ( cublasStatus != CUBLAS_STATUS_SUCCESS )
        {
            CUDAErrorMessage::printErrorMessage( cublasStatus, __FILE__, __LINE__ );
        }
    }
    // ... else use summation to make the submatrix copy
    else
    {
        int rows = std::min( src.mRows, dst.mRows );
        int cols = std::min( src.mCols, dst.mCols );
        T a = 1;
        T b = 0;
        cublasStatus_t cublasStatus = CUDAMatrix::cublasgeam( src.mCUBLASHandle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, &a, src.mDevicePtr, src.mRows, &b, src.mDevicePtr, src.mRows, dst.mDevicePtr, dst.mRows );

        if ( cublasStatus != CUBLAS_STATUS_SUCCESS )
        {
            CUDAErrorMessage::printErrorMessage( cublasStatus, __FILE__, __LINE__ );
        }
    }
}


template<typename T>
void CUDAMatrix<T>::fillCol( CUDAMatrix& m, int col, T value )
{
    std::vector<T> values( m.mRows );
    std::fill( values.begin(), values.end(), value ); 
    fillCol( m, col, values );
}


template<typename T>
void CUDAMatrix<T>::fillCol( CUDAMatrix& m, int col, const std::vector<T>& values )
{
    // Check parameters
    if ( col < 0 || col >= m.mCols || values.size() == 0 )
    {
        std::cerr << "Checking parameters error at " << __FILE__ << ":" << __LINE__ << "\n";
        return;
    }
    
    T* devicePtr = m.mDevicePtr + col*m.mRows;
    int size = std::min( static_cast<int>( values.size() ), m.mRows );
    CUDAVector<T> v( m.mCUBLASHandle, values.size(), values );
    cublasStatus_t cublasStatus = CUDAMatrix::cublasCopy( m.mCUBLASHandle, size, v.mDevicePtr, 1, devicePtr, 1 );

    if ( cublasStatus != CUBLAS_STATUS_SUCCESS )
    {
        CUDAErrorMessage::printErrorMessage( cublasStatus, __FILE__, __LINE__ );
    }
}


template<typename T>
void CUDAMatrix<T>::fillRow( CUDAMatrix& m, int row, T value )
{
    std::vector<T> values( m.mCols );
    std::fill( values.begin(), values.end(), value ); 
    fillRow( m, row, values );
}


template<typename T>
void CUDAMatrix<T>::fillRow( CUDAMatrix& m, int row, const std::vector<T>& values )
{
    // Check parameters
    if ( row < 0 || row >= m.mRows || values.size() == 0 )
    {
        CUDAErrorMessage::printErrorMessage( CUBLAS_STATUS_INVALID_VALUE, __FILE__, __LINE__ );
        return;
    }
    
    T* devicePtr = m.mDevicePtr + row;
    int size = std::min( static_cast<int>( values.size() ), m.mCols );
    CUDAVector<T> v( m.mCUBLASHandle, values.size(), values );
    cublasStatus_t cublasStatus = CUDAMatrix::cublasCopy( m.mCUBLASHandle, size, v.mDevicePtr, 1, devicePtr, m.mRows );

    if ( cublasStatus != CUBLAS_STATUS_SUCCESS )
    {
        CUDAErrorMessage::printErrorMessage( cublasStatus, __FILE__, __LINE__ );
    }
}

// cuRAND API overload
curandStatus_t CURANDAPI generateUniform(curandGenerator_t generator, float* outputPtr, size_t num);
curandStatus_t CURANDAPI generateUniform(curandGenerator_t generator, double* outputPtr, size_t num);

template<typename T>
void CUDAMatrix<T>::fillRandom( CUDAMatrix& m )
{
    curandGenerator_t prng;
    curandStatus_t status;
    
    status = curandCreateGenerator( &prng, CURAND_RNG_PSEUDO_XORWOW );
    if (status != CURAND_STATUS_SUCCESS)
    {
        CUDAErrorMessage::printErrorMessage( status, std::string(__FILE__), __LINE__ );
    }
    
    status = curandSetPseudoRandomGeneratorSeed( prng, time(0) );
    if (status != CURAND_STATUS_SUCCESS)
    {
        CUDAErrorMessage::printErrorMessage( status, std::string(__FILE__), __LINE__ );
    }
    
    // Generates mRows * mCols uniform pseudo-random numbers in range (0, 1]
    status = generateUniform(prng, m.mDevicePtr, m.mRows * m.mCols );
    if (status != CURAND_STATUS_SUCCESS)
    {
        CUDAErrorMessage::printErrorMessage( status, std::string(__FILE__), __LINE__ );
    }
}


template<typename T>
void CUDAMatrix<T>::print( const CUDAMatrix& m )
{
    std::vector<T> buffer;
    m.getData( buffer, false );
    
    for ( int i = 0; i < m.mRows; i++ )
    {
        for ( int j = 0; j < m.mCols; j++ )
        {
            T value = buffer[IDX2C(i, j, m.mRows)];
            std::cout << std::setprecision(10) << value << std::setw(20);
        }
        std::cout << "\n";
    }
}


template<typename T>
CUDAMatrix<T>& CUDAMatrix<T>::operator=( const CUDAMatrix& m )
{
    mCUBLASHandle = m.mCUBLASHandle;

    if ( mRows*mCols != m.mRows*m.mCols )
    {
        // Release device memory
        cudaFree(mDevicePtr);

        // Allocates memory on device
        cudaError_t cudaStatus = cudaMalloc( (void**)&mDevicePtr, m.mRows*m.mCols*sizeof(T) );
        if (cudaStatus != cudaSuccess)
        {
            CUDAErrorMessage::printErrorMessage( cudaStatus, std::string(__FILE__), __LINE__ );
            mRows = 0;
            mCols = 0;
        }
    }
    
    mRows = m.mRows;
    mCols = m.mCols;
    
    CUDAMatrix::copy( m, *this );
    
    return *this;
}

}

#endif	/* CUDAMATRIX_H */
