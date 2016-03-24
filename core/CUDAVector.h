/* 
 * File: CUDAVector.h
 * 
 * Data structure for vector storage and manipulation in gpu using cuBLAS library.
 * 
 * Author: Eder Perez
 */

#ifndef CUDAVECTOR_H
#define	CUDAVECTOR_H

#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "CUDAErrorMessage.h"


namespace gpu
{

template<typename T> class CUDAVector
{
    template<typename U> friend class CUDAMatrix;

    public:
        
        /**
         * Create a vector on gpu with the given dimension. The vector values
         * must be filled using the setData method before using it.
         * 
         * @param handle A valid cuBLAS context.
         * @param dim Vector dimension.
         */
        CUDAVector( const cublasHandle_t& handle, int dim );
        
        /**
         * Create a vector on gpu with the given dimensions and buffer data.
         * 
         * @param dim Vector dimension.
         * @param buffer Buffer containing data.
         */
        CUDAVector( const cublasHandle_t& handle, int dim, const std::vector<T>& buffer );
        
        /**
         * Release device memory.
         */
        ~CUDAVector();
        
        /**
         * Get the vector dimension.
         */
        int dim() const;
        
        /**
         * Set vector values.
         * 
         * @param buffer Buffer containing data.
         */
        void setData( const std::vector<T>& buffer );
        
        /**
         * Retrieve vector data.
         */
        void getData( std::vector<T>& buffer );
        
    private:
        
        cublasHandle_t mCUBLASHandle;
        int mDim;
        T* mDevicePtr;
};


template<typename T>
CUDAVector<T>::CUDAVector( const cublasHandle_t& handle, int dim ) :
    mCUBLASHandle(handle),
    mDim(dim),
    mDevicePtr(0)
{
    // Allocates memory on device
    cudaError_t cudaStatus = cudaMalloc( (void**)&mDevicePtr, mDim*sizeof(T) );
    if (cudaStatus != cudaSuccess)
    {
        CUDAErrorMessage::printErrorMessage( cudaStatus, __FILE__, __LINE__ );
    }
}


template<typename T>
CUDAVector<T>::CUDAVector( const cublasHandle_t& handle, int dim, const std::vector<T>& buffer ) :
    CUDAVector( handle, dim )
{
    setData( buffer );
}


template<typename T>
CUDAVector<T>::~CUDAVector()
{
    // Release device memory
    cudaFree(mDevicePtr);
}


template<typename T>
int CUDAVector<T>::dim() const
{
    return mDim;
}


template<typename T>
void CUDAVector<T>::setData( const std::vector<T>& buffer )
{
    if ( buffer.size() < static_cast<size_t>( mDim ) )
    {
        return;
    }
    
    cublasStatus_t status = cublasSetVector( mDim, sizeof(T), buffer.data(), 1, mDevicePtr, 1 );
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        CUDAErrorMessage::printErrorMessage( status, __FILE__, __LINE__ );
    }
}


template<typename T>
void CUDAVector<T>::getData( std::vector<T>& buffer )
{
    buffer.resize( mDim );
    
    cublasStatus_t status = cublasGetVector( mDim, sizeof(T), mDevicePtr, 1, buffer.data(), 1);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        CUDAErrorMessage::printErrorMessage( status, __FILE__, __LINE__ );
        buffer.clear();
        return;
    }
}

}

#endif	/* CUDAVECTOR_H */
