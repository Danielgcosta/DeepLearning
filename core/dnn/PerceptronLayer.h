/* 
 * File:   PerceptronLayer.h
 * Author: ederperez
 */

#ifndef PERCEPTRON_LAYER_H
#define PERCEPTRON_LAYER_H

#include "Layer.h"


namespace dnn
{


template<typename T>
class PerceptronLayer : public Layer<T>
{
    public:
        
        PerceptronLayer( const LayerParameters& parameters, cublasHandle_t handle );
        
        /**
         * Executes the forward pass.
         * WARNING: No activation function is executed in the output.
         * 
         * @param deviceInput A pointer to a gpu::CUDAMatrix object with @batchSize rows and
         * LayerParameters.inputs neurons.
         * @param deviceOutput A pointer to a gpu::CUDAMatrix object with @batchSize rows and
         * LayerParameters.outputs neurons.
         * @param batchSize This parameter is ignored.
         */
        void forwardPass( const void* deviceInput, void* deviceOutput, int batchSize = 1 );
        
        /**
         * Executes the backward pass.
         * WARNING: No derivative of activation function is executed in the output.
         * 
         * @param deviceInput A pointer to a gpu::CUDAMatrix object with N rows and
         * LayerParameters.inputs values.
         * @param deviceOutput A pointer to a gpu::CUDAMatrix object with N rows and
         * LayerParameters.outputs values.
         * @param batchSize This parameter is ignored.
         */
        void backwardPass( const void* deviceInput, void* deviceOutput, int batchSize = 1 );
        
        /**
         * Gets the weights matrix.
         */
        const gpu::CUDAMatrix<T>& getWeights();

        /**
         * Sets new weights if @weights and @mWeights has same dimensions.
         */
        bool setWeights( const gpu::CUDAMatrix<T>& weights );
        
    private:

        cublasHandle_t mCUBLASHandle;
        
        /**
         * +1 column for bias
         * +1 row for one extra 1-valued element in neuron vector
         */
        gpu::CUDAMatrix<T> mWeights;
};


template<typename T>
PerceptronLayer<T>::PerceptronLayer( const LayerParameters& parameters, cublasHandle_t handle ) :
    Layer<T>::Layer( parameters ),
    mCUBLASHandle( handle ),
    mWeights( mCUBLASHandle, parameters.outputs + 1, parameters.inputs + 1 )
{
    // Initialize weights with random values in (-1, 1]
    gpu::CUDAMatrix<T>::fillRandom( mWeights );
    gpu::CUDAMatrix<T> ones( mCUBLASHandle, mWeights.rows(), mWeights.cols() );
    for ( int i = 0; i < ones.cols(); i++ )
    {
        gpu::CUDAMatrix<T>::fillCol( ones, i, 1.0 );
    }
    gpu::CUDAMatrix<T>::add( 2.0, mWeights, false, -1.0, ones, false, mWeights );
    
    // Adjust bias components
    gpu::CUDAMatrix<T>::fillRow( mWeights, mWeights.rows() - 1, 0);
}


template<typename T>
void PerceptronLayer<T>::forwardPass( const void* deviceInput, void* deviceOutput, int batchSize )
{
    // Check parameters
    if ( deviceInput == nullptr || deviceOutput == nullptr )
    {
        return;
    }

    // deviceInput and deviceOutput must be a pointer to a gpu::CUDAMatrix
    const gpu::CUDAMatrix<T>* input = static_cast<const gpu::CUDAMatrix<T>*>( deviceInput );
    gpu::CUDAMatrix<T>* output = static_cast<gpu::CUDAMatrix<T>*>( deviceOutput );
    
    // Computes input * mWeights^T = output
    gpu::CUDAMatrix<T>::mult( *input, false, mWeights, true, *output );
    
    // Adjust bias component in output
    gpu::CUDAMatrix<T>::fillCol( *output, output->cols() - 1, 1);
}


template<typename T>
void PerceptronLayer<T>::backwardPass( const void* deviceInput, void* deviceOutput, int batchSize )
{
    // Check parameters
    if ( deviceInput == nullptr || deviceOutput == nullptr )
    {
        return;
    }

    // deviceInput and deviceOutput must be a pointer to a gpu::CUDAMatrix
    const gpu::CUDAMatrix<T>* input = static_cast<const gpu::CUDAMatrix<T>*>( deviceInput );
    gpu::CUDAMatrix<T>* output = static_cast<gpu::CUDAMatrix<T>*>( deviceOutput );
    
    // Computes input * mWeights = output
    gpu::CUDAMatrix<T>::mult( *input, false, mWeights, false, *output );
    
    // Adjust bias component in output
    gpu::CUDAMatrix<T>::fillCol( *output, output->cols() - 1, 0);
}


template<typename T>
const gpu::CUDAMatrix<T>& PerceptronLayer<T>::getWeights()
{
    return mWeights;
}


template<typename T>
bool PerceptronLayer<T>::setWeights( const gpu::CUDAMatrix<T>& weights )
{
    if ( mWeights.rows() == weights.rows() && mWeights.cols() == weights.cols() )
    {
        mWeights = weights;
        return true;
    }
    
    return false;
}

}

#endif
