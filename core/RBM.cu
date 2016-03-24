/* 
 * File: RBM.cu
 * 
 * Restricted Boltzmann Machine.
 * 
 * Author: Eder Perez
 */

#include "RBM.h"
#include "CUDAErrorMessage.h"

namespace gpu
{

////////////////////////////////////////////////////////////////////////////////
/////                      GPU kernels and functions                       /////
////////////////////////////////////////////////////////////////////////////////

__global__ void _seedRandomGenerator( int rows, int cols, curandState* state, unsigned long seed )
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int id = j * cols + i;
    
    // Each thread gets same seed, a different sequence number, no offset
    if( i < cols && j < rows )
    {
        curand_init( seed, id, 0, &state[id] );
    }
}


__global__ void _gpuLogistic( int rows, int cols, const float* src, float* dst )
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int index = j * cols + i;
    
    if( i < cols && j < rows )
    {
        dst[index] = 1.0 / ( 1.0 + exp(-src[index]) );
    }
}


__global__ void _gpuLogistic( int rows, int cols, const double* src, double* dst )
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int index = j * cols + i;
    
    if( i < cols && j < rows )
    {
        dst[index] = 1.0 / ( 1.0 + exp(-src[index]) );
    }
}


__global__ void _gpuActivateState( curandState* randState, int rows, int cols, const float* src, float* dst )
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int index = j * cols + i;

    if( i < cols && j < rows )
    {
        dst[index] = src[index] > curand_uniform( &randState[index] ) ? 1 : 0;
    }
}


__global__ void _gpuActivateState( curandState* randState, int rows, int cols, const double* src, double* dst )
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int index = j * cols + i;

    if( i < cols && j < rows )
    {
        dst[index] = src[index] > curand_uniform( &randState[index] ) ? 1 : 0;
    }
}


////////////////////////////////////////////////////////////////////////////////
/////                 Auxiliary functions implementation                   /////
////////////////////////////////////////////////////////////////////////////////

inline int iDivUp( int a, int b )
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void seedRandomGenerator( int rows, int cols, curandState* randState, unsigned long seed )
{
    dim3 threadsPerBlock(32, 16);
    dim3 blockSize( iDivUp( cols, threadsPerBlock.x ), iDivUp( rows, threadsPerBlock.y ) );
    _seedRandomGenerator<<<blockSize, threadsPerBlock>>>( rows, cols, randState, time(0) );
    cudaDeviceSynchronize();
}


void gpuLogistic( int rows, int cols, const float* src, float* dst )
{
    dim3 threadsPerBlock(32, 16);
    dim3 blockSize( iDivUp( cols, threadsPerBlock.x ), iDivUp( rows, threadsPerBlock.y ) );
    _gpuLogistic<<<blockSize, threadsPerBlock>>>( rows, cols, src, dst );
    cudaDeviceSynchronize();
}


void gpuLogistic( int rows, int cols, const double* src, double* dst )
{
    dim3 threadsPerBlock(32, 16);
    dim3 blockSize( iDivUp( cols, threadsPerBlock.x ), iDivUp( rows, threadsPerBlock.y ) );
    _gpuLogistic<<<blockSize, threadsPerBlock>>>( rows, cols, src, dst );
    cudaDeviceSynchronize();
}


void gpuActivateState( curandState* randState, int rows, int cols, const float* src, float* dst )
{
    dim3 threadsPerBlock(32, 16);
    dim3 blockSize( iDivUp( cols, threadsPerBlock.x ), iDivUp( rows, threadsPerBlock.y ) );
    _gpuActivateState<<<blockSize, threadsPerBlock>>>( randState, rows, cols, src, dst );
    cudaDeviceSynchronize();
}


void gpuActivateState( curandState* randState, int rows, int cols, const double* src, double* dst )
{
    dim3 threadsPerBlock(32, 16);
    dim3 blockSize( iDivUp( cols, threadsPerBlock.x ), iDivUp( rows, threadsPerBlock.y ) );
    _gpuActivateState<<<blockSize, threadsPerBlock>>>( randState, rows, cols, src, dst );
    cudaDeviceSynchronize();
}

}
