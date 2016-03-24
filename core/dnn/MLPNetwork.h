/* 
 * File:   MLPNetwork.h
 * Author: ederperez
 */

#ifndef MLP_NETWORK_H
#define MLP_NETWORK_H

#include <iostream>
#include <cstring>
#include <random>
#include <cublas_v2.h>
#include "CUDAMatrix.h"
#include "CUDAErrorMessage.h"
#include "PerceptronLayer.h"


namespace dnn
{

struct DNNParameters
{
    float learningRate;
    float maxError;
    int epochs;
    
    DNNParameters( float learningRate, float maxError, int epochs ) :
        learningRate( learningRate ),
        maxError(maxError),
        epochs( epochs )
    {
    }
};

template<typename T>
class MLPNetwork
{
    public:
        
        MLPNetwork( const DNNParameters& parameters );
        MLPNetwork( const DNNParameters& parameters, const std::vector<int>& networkStructure );

        /**
         * Inserts a new layer at the end of the network. The layer is only created
         * and inserted if parameters.inputs equals last layer output size.
         * @param parameters Parameters necessary for the creation of a PerceptronLayer.
         * @return False if layer is incompatible with the current network, true otherwise.
         */
        
        bool pushLayer( const LayerParameters& parameters );

        /**
         * Removes the last layer.
         * @return True if layer was removed, false otherwise.
         */
        bool popLayer();
        
        /**
         * Gets the weights of a specific layer.
         * @param inputLayer Index of input layer for the required weights.
         * @return A reference to weights.
         */
        const gpu::CUDAMatrix<T>* getWeights( int inputLayer );

        /**
         * Sets values of weights. The dimension of a matrix of weights is given by:
         * (M+1)x(N+1), where M is the output layer size and N is the input layer
         * size.
         * @param inputLayer Index of input layer for the required weights.
         * @param weights New weights values.
         * @return True if weights are updated successfully, false otherwise.
         */
        bool setWeights( int inputLayer, const gpu::CUDAMatrix<T>& weights );

        /**
         * Executes a batch training in the network.
         * @param input Array of input samples in host memory.
         * @param target Array of targets samples in host memory.
         * @param numberOfSamples Total number of samples.
         * @param inputSampleSize Dimension of an input sample.
         * @param targetSampleSize Dimension of a target sample.
         */
        void train( const T* input, const T* target, int numberOfSamples, int inputSampleSize, int targetSampleSize );
        
        /**
         * Executes the network in a batch of inputs.
         * @param input Array of input samples.
         * @param output Array of computed samples.
         * @param numberOfSamples Total number of samples.
         * @param inputSampleSize Dimension of an input sample.
         * @param outputSampleSize Dimension of an output sample.
         */
        void run( const T* input, T* output, int numberOfSamples, int inputSampleSize, int outputSampleSize );

    private:

        /**
         * Compute a good batch size to be sent to GPU.
         * @param totalInputSize Total number of input samples.
         * @return Batch size.
         */
        int computeBatchSize( int totalInputSize );
        
        /**
         * Copy host data into GPU.
         * @param input Array of input samples.
         * @param target Array of targets samples.
         * @param numberOfSamples Total number of samples.
         * @param gpuInput Matrix to be filled with input samples.
         * @param gpuTarget Matrix to be filled with target samples.
         * @param rndGenerator Mersenne Twister random number generator.
         */
        void fillBatch( const T* input, const T* target, int numberOfSamples, gpu::CUDAMatrix<T>& gpuInput, gpu::CUDAMatrix<T>& gpuTarget, std::mt19937_64& rndGenerator );

        /**
         * Copy host data into GPU.
         * @param input Array of input samples.
         * @param startRow Start row in input.
         * @param gpuInput Matrix to be filled with input samples.
         */
        void fillBatch( const T* input, int& startRow, gpu::CUDAMatrix<T>& gpuInput );
        
        void fillOutputBatch( const gpu::CUDAMatrix<T>& gpuOutput, T* output, int startRow );
        
        /**
         * Executes the backpropagation algorithm and returns the current error
         * between output and target.
         * @param gpuInput Matrix with input samples.
         * @param gpuTarget Matrix with target samples.
         * @return The error between outputs and targets.
         */
        T backPropagation( const gpu::CUDAMatrix<T>& gpuInput, const gpu::CUDAMatrix<T>& gpuTarget );
        
        DNNParameters mParameters;
        std::vector< PerceptronLayer<T> > mNetwork;
        cublasHandle_t mCUBLASHandle;
        
};


template<typename T>
MLPNetwork<T>::MLPNetwork( const DNNParameters& parameters ) :
    mParameters( parameters )
{
    // Creates a cublas handle
    cublasStatus_t status = cublasCreate( &mCUBLASHandle );
    if ( status != CUBLAS_STATUS_SUCCESS )
    {
        gpu::CUDAErrorMessage::printErrorMessage( status, __FILE__, __LINE__ );
    }
}


template<typename T>
MLPNetwork<T>::MLPNetwork( const DNNParameters& parameters, const std::vector<int>& networkStructure ) :
    MLPNetwork( parameters )
{
    for ( size_t i = 1; i < networkStructure.size(); i++ )
    {
        pushLayer( LayerParameters( networkStructure[i-1], networkStructure[i] ) );
    }
}


template<typename T>
bool MLPNetwork<T>::pushLayer( const LayerParameters& parameters )
{
    if ( mNetwork.empty() || parameters.inputs == mNetwork.back().outputs() )
    {
        mNetwork.push_back( PerceptronLayer<T>( parameters, mCUBLASHandle ) );
        return true;
    }
    
    return false;
}


template<typename T>
bool MLPNetwork<T>::popLayer()
{
    if ( !mNetwork.empty() )
    {
        mNetwork.erase( mNetwork.end() - 1 );
        return true;
    }
    
    return false;
}


template<typename T>
const gpu::CUDAMatrix<T>* MLPNetwork<T>::getWeights( int inputLayer )
{
    if ( !mNetwork.empty() )
    {
        inputLayer = std::min( static_cast<int>( mNetwork.size() ) - 1, inputLayer );
        inputLayer = std::max( 0, inputLayer );
        
        return &mNetwork[inputLayer].getWeights();
    }

    return 0;
}


template<typename T>
bool MLPNetwork<T>::setWeights( int inputLayer, const gpu::CUDAMatrix<T>& weights )
{
    if ( !mNetwork.empty() )
    {
        inputLayer = std::min( static_cast<int>( mNetwork.size() ) - 1, inputLayer );
        inputLayer = std::max( 0, inputLayer );
        
        return mNetwork[inputLayer].setWeights( weights );
    }

    return false;
}


template<typename T>
void MLPNetwork<T>::train( const T* input, const T* target, int numberOfSamples, int inputSampleSize, int targetSampleSize )
{
    // Check parameters
    if ( input == nullptr || target == nullptr ||
         numberOfSamples <= 0 || inputSampleSize <= 0 || targetSampleSize <= 0 ||
         mNetwork.empty() ||
         mNetwork[0].inputs() != inputSampleSize ||
         mNetwork.back().outputs() != targetSampleSize )
    {
        std::cerr << "MLPNetwork: invalid parameters.\n";
        return;
    }
    
    // Find a good batch size to be sent to GPU
    int batchSize = computeBatchSize( numberOfSamples );
    
    std::cout << "MLPNetwork training batch size: " << batchSize << "\n";
    
    // Creates gpu matrices to fill with data (+1 in the number of columns for bias).
    gpu::CUDAMatrix<T> gpuInput( mCUBLASHandle, batchSize, inputSampleSize + 1 );
    gpu::CUDAMatrix<T> gpuTarget( mCUBLASHandle, batchSize, targetSampleSize + 1);
    
    // Initialize random number generator for the method fillBatch
    std::random_device randomDevice;
    std::mt19937_64 rndGenerator( randomDevice() );

    bool canStop = false;
    for ( int i = 0; i < mParameters.epochs; i++ )
    {
        // Train a number of samples equivalent to the total number of samples
        for ( int j = 0; j < numberOfSamples/batchSize + 1; j++ )
        {
            // Populates gpu matrices with random samples
            fillBatch( input, target, numberOfSamples, gpuInput, gpuTarget, rndGenerator );

            // Executes the backpropagation algorithm updating all weights
            T error = backPropagation( gpuInput, gpuTarget );
            
            std::cout << "MLPNetwork error: " << error << std::endl;
            
            if ( error < static_cast<T>( mParameters.maxError ) )
            {
                canStop = true;
                break;
            }
        }
        
        if ( canStop )
        {
            break;
        }
    }
}


// Declaration of activation functions
void gpuSigmoid( int rows, int cols, const float* src, float* dst );
void gpuSigmoid( int rows, int cols, const double* src, double* dst );
void gpuSigmoidDerivative( int rows, int cols, const float* src, float* dst );
void gpuSigmoidDerivative( int rows, int cols, const double* src, double* dst );


template<typename T>
void MLPNetwork<T>::run( const T* input, T* output, int numberOfSamples, int inputSampleSize, int outputSampleSize )
{
    // Check parameters
    if ( input == nullptr || output == nullptr ||
         numberOfSamples <= 0 || inputSampleSize <= 0 || outputSampleSize <= 0 ||
         mNetwork.empty() ||
         mNetwork[0].inputs() != inputSampleSize ||
         mNetwork.back().outputs() != outputSampleSize )
    {
        std::cerr << "MLPNetwork: invalid parameters.\n";
        return;
    }
    
    // Find a good batch size to be sent to GPU
    int batchSize = computeBatchSize( numberOfSamples );
    
    std::cout << "MLPNetwork batch size: " << batchSize << "\n";
    
    // Creates gpu matrices to fill with data (+1 in the number of columns for bias).
    gpu::CUDAMatrix<T> gpuInput( mCUBLASHandle, batchSize, inputSampleSize + 1 );
    
    // Create neurons
    std::vector< gpu::CUDAMatrix<T> > actNeurons;
    actNeurons.reserve( mNetwork.size() );
    for ( auto& layer : mNetwork )
    {
        actNeurons.push_back( gpu::CUDAMatrix<T>( mCUBLASHandle, batchSize, layer.outputs() + 1 ) );
    }

    // Starts neural network computation
    int startRow = 0;
    for ( int k = 0; k < numberOfSamples/batchSize; k++ )
    {
        // Populates gpu matrices with input samples
        fillBatch( input, startRow, gpuInput );
        
        // Feedforward the network activating all neurons
        const gpu::CUDAMatrix<T>* inputNeurons = &gpuInput;
        for ( size_t i = 0; i < mNetwork.size(); i++ )
        {
            mNetwork[i].forwardPass( static_cast<const void*>( inputNeurons ), static_cast<void*>( &actNeurons[i] ) );

            // Activate neurons
            gpuSigmoid( actNeurons[i].rows(), actNeurons[i].cols(), actNeurons[i].getDevicePtr(), actNeurons[i].getDevicePtr() );

            // Adjust bias component
            gpu::CUDAMatrix<T>::fillCol( actNeurons[i], actNeurons[i].cols() - 1, 1.0 );

            inputNeurons = &actNeurons[i];
        }

        // Copy output neuron to output array
        fillOutputBatch( actNeurons.back(), output, startRow - batchSize );
    }
    
    // Tail of data
    if ( startRow < numberOfSamples )
    {
        numberOfSamples = numberOfSamples - startRow;
        batchSize = numberOfSamples;

        std::cout << "MLPNetwork last batch size: " << batchSize << "\n";

        gpuInput = gpu::CUDAMatrix<T>( mCUBLASHandle, batchSize, inputSampleSize + 1 );

        actNeurons.clear();
        for ( auto& layer : mNetwork )
        {
            actNeurons.push_back( gpu::CUDAMatrix<T>( mCUBLASHandle, batchSize, layer.outputs() + 1 ) );
        }

        fillBatch( input, startRow, gpuInput );

        const gpu::CUDAMatrix<T>* inputNeurons = &gpuInput;
        for ( size_t i = 0; i < mNetwork.size(); i++ )
        {
            mNetwork[i].forwardPass( static_cast<const void*>( inputNeurons ), static_cast<void*>( &actNeurons[i] ) );

            // Activate neurons
            gpuSigmoid( actNeurons[i].rows(), actNeurons[i].cols(), actNeurons[i].getDevicePtr(), actNeurons[i].getDevicePtr() );

            // Adjust bias component
            gpu::CUDAMatrix<T>::fillCol( actNeurons[i], actNeurons[i].cols() - 1, 1.0 );

            inputNeurons = &actNeurons[i];
        }

        // Copy output neuron to output array
        fillOutputBatch( actNeurons.back(), output, startRow - batchSize );
    }
}


template<typename T>
int MLPNetwork<T>::computeBatchSize( int totalInputSize )
{
    // The ideal size would be one that balances the use of GPU memory and
    // computing time but ...
    // ... let's just guess for now (or ever:)
//    return std::min<int>( 10000, totalInputSize );
    return 0.05*totalInputSize;
}


template<typename T>
void MLPNetwork<T>::fillBatch( const T* input, const T* target, int numberOfSamples, gpu::CUDAMatrix<T>& gpuInput, gpu::CUDAMatrix<T>& gpuTarget, std::mt19937_64& rndGenerator )
{
    std::uniform_int_distribution<unsigned long long> distribution( 0, numberOfSamples - 1 );
    
    int batchSize = gpuInput.rows();
    int inputSampleSize = gpuInput.cols() - 1;
    int targetSampleSize = gpuTarget.cols() - 1;

    // This vector will be used several times with the same size, so
    // let it be static to avoid unnecessary memory allocation operations.
    static std::vector<T> hostInput;
    static std::vector<T> hostTarget;
    hostInput.resize( gpuInput.rows() * gpuInput.cols() );
    hostTarget.resize( gpuTarget.rows() * gpuTarget.cols() );
    
    int hostInputIndex = 0;
    int hostTargetIndex = 0;

    for ( int i = 0; i < batchSize; i++ )
    {
        int randomIndex = distribution( rndGenerator );
        memcpy( hostInput.data() + hostInputIndex, &input[randomIndex*inputSampleSize], sizeof(T)*inputSampleSize );
        hostInputIndex += inputSampleSize;
        hostInput[hostInputIndex++] = 1.0;
        
        memcpy( hostTarget.data() + hostTargetIndex, &target[randomIndex*targetSampleSize], sizeof(T)*targetSampleSize );
        hostTargetIndex += targetSampleSize;
        hostTarget[hostTargetIndex++] = 1.0;
    }

    gpuInput.setData( hostInput, true );
    gpuTarget.setData( hostTarget, true );
}


template<typename T>
void MLPNetwork<T>::fillBatch( const T* input, int& startRow, gpu::CUDAMatrix<T>& gpuInput )
{
    int batchSize = gpuInput.rows();
    int inputSampleSize = gpuInput.cols() - 1;

    // This vector will be used several times with the same size, so
    // let it be static to avoid unnecessary memory allocation operations.
    static std::vector<T> hostInput( gpuInput.rows() * gpuInput.cols() );
    int hostInputIndex = 0;

    for ( int i = 0; i < batchSize; i++ )
    {
        memcpy( hostInput.data() + hostInputIndex, &input[startRow*inputSampleSize], sizeof(T)*inputSampleSize );
        hostInputIndex += inputSampleSize;
        hostInput[hostInputIndex++] = 1.0;
        startRow++;
    }

    gpuInput.setData( hostInput, true );
}


template<typename T>
void MLPNetwork<T>::fillOutputBatch( const gpu::CUDAMatrix<T>& gpuOutput, T* output, int startRow )
{
    int batchSize = gpuOutput.rows();
    int outputSampleSize = gpuOutput.cols() - 1;

    // This vector will be used several times with the same size, so
    // let it be static to avoid unnecessary memory allocation operations.
    static std::vector<T> hostOutput( gpuOutput.rows() * gpuOutput.cols() );
    
    gpuOutput.getData( hostOutput, true );
    int hostOutputIndex = 0;

    for ( int i = 0; i < batchSize; i++ )
    {
        memcpy( output + startRow*outputSampleSize, hostOutput.data() + hostOutputIndex, sizeof(T)*outputSampleSize );
        hostOutputIndex += outputSampleSize + 1;
        startRow++;
    }
}


template<typename T>
T MLPNetwork<T>::backPropagation( const gpu::CUDAMatrix<T>& gpuInput, const gpu::CUDAMatrix<T>& gpuTarget )
{
    // Auxiliary matrices for, respectively, activated neurons, derivative of activated neurons,
    // deltas from backward pass, gradients of weights.
    std::vector< gpu::CUDAMatrix<T> > actNeurons; actNeurons.reserve( mNetwork.size() );
    std::vector< gpu::CUDAMatrix<T> > devActNeurons; devActNeurons.reserve( mNetwork.size() );
    std::vector< gpu::CUDAMatrix<T> > deltas; deltas.reserve( mNetwork.size() );
    std::vector< gpu::CUDAMatrix<T> > gradWeights; gradWeights.reserve( mNetwork.size() );
    
    // Create auxiliary matrices
    int batchSize = gpuInput.rows();
    for ( auto& layer : mNetwork )
    {
        actNeurons.push_back( gpu::CUDAMatrix<T>( mCUBLASHandle, batchSize, layer.outputs() + 1 ) );
        devActNeurons.push_back( gpu::CUDAMatrix<T>( mCUBLASHandle, batchSize, layer.outputs() + 1 ) );
        deltas.push_back( gpu::CUDAMatrix<T>( mCUBLASHandle, batchSize, layer.outputs() + 1 ) );
        gradWeights.push_back( gpu::CUDAMatrix<T>( mCUBLASHandle, layer.getWeights().rows(), layer.getWeights().cols() ) );
    }

    // Feedforward the network activating all neurons and computing its derivative
    const gpu::CUDAMatrix<T>* input = &gpuInput;
    for ( size_t i = 0; i < mNetwork.size(); i++ )
    {
        mNetwork[i].forwardPass( static_cast<const void*>( input ), static_cast<void*>( &actNeurons[i] ) );
        
        // Activate neurons
        gpuSigmoidDerivative( actNeurons[i].rows(), actNeurons[i].cols(), actNeurons[i].getDevicePtr(), devActNeurons[i].getDevicePtr() );
        gpuSigmoid( actNeurons[i].rows(), actNeurons[i].cols(), actNeurons[i].getDevicePtr(), actNeurons[i].getDevicePtr() );

        // Adjust bias component
        gpu::CUDAMatrix<T>::fillCol( actNeurons[i], actNeurons[i].cols() - 1, 1.0 );
        gpu::CUDAMatrix<T>::fillCol( devActNeurons[i], devActNeurons[i].cols() - 1, 1.0 );
        
        input = &actNeurons[i];
    }
    
    // Computes output deltas based on the error in output layer
    gpu::CUDAMatrix<T>::sub( actNeurons.back(), false, gpuTarget, false, deltas.back() );
    gpu::CUDAMatrix<T>::fillCol( deltas.back(), deltas.back().cols() - 1, 0.0 );
    
//    //////////////////////////////////////////////////////////////////////
//    std::vector<T> buffer;
//    const gpu::CUDAMatrix<T>* matrix = &actNeurons.back();
//    matrix->getData( buffer, true );
//    T err = 0.0;
//    for ( int i = 0; i < matrix->rows(); i++)
//    {
//        float perr = 0.0;
//        for ( int j = 0; j < matrix->cols(); j++)
//        {
//            std::cout << buffer[i*matrix->cols() + j] << "\t";
//            perr += buffer[i*matrix->cols() + j]*buffer[i*matrix->cols() + j];
//        }
//        err += std::sqrt(perr);
//        std::cout << "\n";
//    }
//    std::cout << "\n";
//    std::cout << "Aiai: " << err/matrix->rows() << "\n";
//    //////////////////////////////////////////////////////////////////////
    
    T error = gpu::CUDAMatrix<T>::norm( deltas.back() );
    error *= error;
    
    if ( error < mParameters.maxError )
    {
        return error;
    }
    
    gpu::CUDAMatrix<T>::hadamard( deltas.back(), devActNeurons.back(), deltas.back() );

    // Computes all remaining deltas
    for ( int i = static_cast<int>( deltas.size() ) - 2; i >= 0; i-- )
    {
        mNetwork[i+1].backwardPass( static_cast<void*>( &deltas[i+1] ), static_cast<void*>( &deltas[i] ) );
        gpu::CUDAMatrix<T>::hadamard( deltas[i], devActNeurons[i], deltas[i] );
    }
    
    // Computes derivatives and update weights
    input = &gpuInput;
    for ( size_t i = 0; i < deltas.size(); i++ )
    {
        // Computes gradients
        gpu::CUDAMatrix<T>::mult( deltas[i], true, *input, false, gradWeights[i] );

        // Update weights
        gpu::CUDAMatrix<T>& weights = const_cast< gpu::CUDAMatrix<T>& >( mNetwork[i].getWeights() );
        gpu::CUDAMatrix<T>::add( 1.0, weights, false, -mParameters.learningRate/T(batchSize), gradWeights[i], false, weights );

        input = &actNeurons[i];
    }
    
    error = gpu::CUDAMatrix<T>::norm( deltas.back() );
    error *= error;
    return error;
}

}

#endif
