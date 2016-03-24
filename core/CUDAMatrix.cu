/* 
 * File: CUDAMatrix.cu
 * 
 * Data structure for dense matrix storage and manipulation in gpu using
 * cuBLAS library.
 * 
 * Author: Eder Perez
 */

#include "CUDAMatrix.h"

namespace gpu
{

__global__ void _gpuHadamard( int rows, int cols, const float* m1, const float* m2, float* m3 )
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int index = j * cols + i;
    
    if( i < cols && j < rows )
    {
        m3[index] = m1[index] * m2[index];
    }
}


__global__ void _gpuHadamard( int rows, int cols, const double* m1, const double* m2, double* m3 )
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int index = j * cols + i;
    
    if( i < cols && j < rows )
    {
        m3[index] = m1[index] * m2[index];
    }
}


int iDivUp( int a, int b )
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}


void gpuHadamard( int rows, int cols, const float* m1, const float* m2, float* m3 )
{
    dim3 threadsPerBlock(32, 16);
    dim3 blockSize( iDivUp( cols, threadsPerBlock.x ), iDivUp( rows, threadsPerBlock.y ) );
    _gpuHadamard<<<blockSize, threadsPerBlock>>>( rows, cols, m1, m2, m3 );
    cudaDeviceSynchronize();
}


void gpuHadamard( int rows, int cols, const double* m1, const double* m2, double* m3 )
{
    dim3 threadsPerBlock(32, 16);
    dim3 blockSize( iDivUp( cols, threadsPerBlock.x ), iDivUp( rows, threadsPerBlock.y ) );
    _gpuHadamard<<<blockSize, threadsPerBlock>>>( rows, cols, m1, m2, m3 );
    cudaDeviceSynchronize();
}

curandStatus_t CURANDAPI generateUniform(curandGenerator_t generator, float* outputPtr, size_t num)
{
    return curandGenerateUniform( generator, outputPtr, num );
}

curandStatus_t CURANDAPI generateUniform(curandGenerator_t generator, double* outputPtr, size_t num)
{
    return curandGenerateUniformDouble( generator, outputPtr, num );
}

template<>
cublasStatus_t CUDAMatrix<float>::cublasCopy( cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy )
{
    return cublasScopy( handle, n, x, incx, y, incy );
}


template<>
cublasStatus_t CUDAMatrix<double>::cublasCopy( cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy )
{
    return cublasDcopy( handle, n, x, incx, y, incy );
}


template<>
cublasStatus_t CUDAMatrix<float>::cublasgemv( cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy )
{
    return cublasSgemv( handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy );
}


template<>
cublasStatus_t CUDAMatrix<double>::cublasgemv( cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy )
{
    return cublasDgemv( handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy );
}


template<>
cublasStatus_t CUDAMatrix<float>::cublasgemm( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc )
{
    return cublasSgemm( handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}


template<>
cublasStatus_t CUDAMatrix<double>::cublasgemm( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc )
{
    return cublasDgemm( handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}


template<>
cublasStatus_t CUDAMatrix<float>::cublasgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc )
{
    return cublasSgeam( handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc );
}


template<>
cublasStatus_t CUDAMatrix<double>::cublasgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* C, int ldc )
{
    return cublasDgeam( handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc );
}


template<>
cublasStatus_t CUDAMatrix<float>::cublasnrm2(cublasHandle_t handle, int n, const float* x, int incx, float* result )
{
    return cublasSnrm2( handle, n, x, incx, result );
}


template<>
cublasStatus_t CUDAMatrix<double>::cublasnrm2(cublasHandle_t handle, int n, const double* x, int incx, double* result )
{
    return cublasDnrm2( handle, n, x, incx, result );
}


template<>
cublasStatus_t CUDAMatrix<float>::cublasscal( cublasHandle_t handle, int n, const float* alpha, float* x, int incx )
{
    return cublasSscal( handle, n, alpha, x, incx );
}


template<>
cublasStatus_t CUDAMatrix<double>::cublasscal( cublasHandle_t handle, int n, const double* alpha, double* x, int incx )
{
    return cublasDscal( handle, n, alpha, x, incx );
}

}
