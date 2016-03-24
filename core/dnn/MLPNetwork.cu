/* 
 * File:   MLPNetwork.cu
 * Author: ederperez
 */

namespace dnn
{

////////////////////////////////////////////////////////////////////////////////
/////                      GPU kernels and functions                       /////
////////////////////////////////////////////////////////////////////////////////

__global__ void _gpuSigmoid( int rows, int cols, const float* src, float* dst )
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int index = j * cols + i;
    
    if( i < cols && j < rows )
    {
        dst[index] = 1.0 / ( 1.0 + exp(-src[index]) );
    }
}


__global__ void _gpuSigmoid( int rows, int cols, const double* src, double* dst )
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int index = j * cols + i;
    
    if( i < cols && j < rows )
    {
        dst[index] = 1.0 / ( 1.0 + exp(-src[index]) );
    }
}


__global__ void _gpuSigmoidDerivative( int rows, int cols, const float* src, float* dst )
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int index = j * cols + i;
    
    if( i < cols && j < rows )
    {
        float s = 1.0 / ( 1.0 + exp(-src[index]) );
        dst[index] = s * (1.0 - s);
    }
}


__global__ void _gpuSigmoidDerivative( int rows, int cols, const double* src, double* dst )
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int index = j * cols + i;
    
    if( i < cols && j < rows )
    {
        double s = 1.0 / ( 1.0 + exp(-src[index]) );
        dst[index] = s * (1.0 - s);
    }
}


////////////////////////////////////////////////////////////////////////////////
/////                 Auxiliary functions implementation                   /////
////////////////////////////////////////////////////////////////////////////////

int iDivUp( int a, int b )
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}


void gpuSigmoid( int rows, int cols, const float* src, float* dst )
{
    dim3 threadsPerBlock(32, 16);
    dim3 blockSize( iDivUp( cols, threadsPerBlock.x ), iDivUp( rows, threadsPerBlock.y ) );
    _gpuSigmoid<<<blockSize, threadsPerBlock>>>( rows, cols, src, dst );
    cudaDeviceSynchronize();
}


void gpuSigmoid( int rows, int cols, const double* src, double* dst )
{
    dim3 threadsPerBlock(32, 16);
    dim3 blockSize( iDivUp( cols, threadsPerBlock.x ), iDivUp( rows, threadsPerBlock.y ) );
    _gpuSigmoid<<<blockSize, threadsPerBlock>>>( rows, cols, src, dst );
    cudaDeviceSynchronize();
}


void gpuSigmoidDerivative( int rows, int cols, const float* src, float* dst )
{
    dim3 threadsPerBlock(32, 16);
    dim3 blockSize( iDivUp( cols, threadsPerBlock.x ), iDivUp( rows, threadsPerBlock.y ) );
    _gpuSigmoidDerivative<<<blockSize, threadsPerBlock>>>( rows, cols, src, dst );
    cudaDeviceSynchronize();
}


void gpuSigmoidDerivative( int rows, int cols, const double* src, double* dst )
{
    dim3 threadsPerBlock(32, 16);
    dim3 blockSize( iDivUp( cols, threadsPerBlock.x ), iDivUp( rows, threadsPerBlock.y ) );
    _gpuSigmoidDerivative<<<blockSize, threadsPerBlock>>>( rows, cols, src, dst );
    cudaDeviceSynchronize();
}

}
