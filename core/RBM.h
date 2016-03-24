/* 
 * File: RBM.h
 * 
 * Restricted Boltzmann Machine.
 * 
 * Author: Eder Perez
 */

#ifndef RBM_H
#define	RBM_H

#include <curand_kernel.h>
#include "CUDAMatrix.h"
#include <limits>

namespace gpu
{

/**
 * Overload to device kernel code.
 */
void seedRandomGenerator( int rows, int cols, curandState* randState, unsigned long seed );
void gpuLogistic( int rows, int cols, const float* src, float* dst );
void gpuLogistic( int rows, int cols, const double* src, double* dst );
void gpuActivateState( curandState* randState, int rows, int cols, const float* src, float* dst );
void gpuActivateState( curandState* randState, int rows, int cols, const double* src, double* dst );


/**
 * This class is used to set RBM parameters.
 */
class RBMParameters
{
    public:
        RBMParameters() :
            learningRate(0.1),
            maxEpochs(1000),
            maxError(1e-3),
            statesAsProbabilities(true)
        {
                
        }
            
        RBMParameters( const RBMParameters& copy ) :
        learningRate( copy.learningRate ),
        maxEpochs( copy.maxEpochs ),
        maxError( copy.maxError ),
        statesAsProbabilities( copy.statesAsProbabilities )
        {
                
        }

        float learningRate;
        int maxEpochs;
        int maxError;
        
        /**
         * The value of the states can be either on/off or
         * the probabilities of being on.
         */
        bool statesAsProbabilities;
};


/**
 * Restricted Boltzmann Machine class.
 */
template<typename T> class RBM
{
    public:

        /**
         * Default constructor with default machine parameters.
         * 
         * @param visibleSize Number of visible units.
         * @param hiddenSize Number of hidden units.
         */
        RBM( const cublasHandle_t& handle, int visibleSize, int hiddenSize );

        /**
         * Constructor with custom machine parameters.
         * 
         * @param visibleSize Number of visible units.
         * @param hiddenSize Number of hidden units.
         * @param parameters Machine parameters.
         */
        RBM( const cublasHandle_t& handle, int visibleSize, int hiddenSize, const RBMParameters& parameters );

        ~RBM();

        /**
         * Get the number of visible units.
         */
        int getVisibleSize();

        /**
         * Get the number of hidden units.
         */
        int getHiddenSize();

        /**
         * Get machine parameters.
         */
        const RBMParameters& getParameters();

        /**
         * Get the weight matrix.
         */
        const CUDAMatrix<T>& getWeights();
        
        /**
         * Train the machine. This method should be called only after the weights
         * were initialized.
         * @param data A matrix where each row is a training example.
         * @param maxError Used to stop the loop.
         */
        void train( const CUDAMatrix<T>& data );

        /**
         * Run the machine on a set of visible units to get a set of hidden units.
         * @param data A matrix where each row consists of the states of the
         * visible units.
         * @param hidden A matrix where each row consists of the hidden units
         * activated from the visible units in the data matrix passed in.
         */
        void runVisible( const CUDAMatrix<T>& data, CUDAMatrix<T>& hidden );

        /**
         * Run the machine on a set of hidden units to get a set of visible units.
         * @param data A matrix where each row consists of the states of the
         * hidden units.
         * @param visible A matrix where each row consists of the visible units
         * activated from the hidden units in the data matrix passed in.
         */
        void runHidden( const CUDAMatrix<T>& data, CUDAMatrix<T>& visible );
        
        /**
         * Computes the mean square error for a give @data.
         * @param data Input data.
         * @return The error.
         */
        double computeMSEError( const CUDAMatrix<T>& data );


    private:

        /**
         * Initializes the weights based on the set parameters.
         */
        void initWeights();
        
        /**
         * Seed the gpu random number generator.
         */
        void setupRandState( int rows, int cols );

        cublasHandle_t mCUBLASHandle;

        int mVisibleSize;
        int mHiddenSize;
        RBMParameters mParameters;

        CUDAMatrix<T> mWeights;

        /**
         * Used to generate pseudo-random numbers in gpu.
         */
        curandState* mCURANDStates;
};


template<typename T>
RBM<T>::RBM( const cublasHandle_t& handle, int visibleSize, int hiddenSize ) :
    mCUBLASHandle( handle ),
    mVisibleSize( visibleSize ),
    mHiddenSize( hiddenSize ),
    mWeights( mCUBLASHandle, visibleSize + 1, hiddenSize + 1 ),
    mCURANDStates( 0 )
{
    initWeights();
}


template<typename T>
RBM<T>::RBM( const cublasHandle_t& handle, int visibleSize, int hiddenSize, const RBMParameters& parameters ) :
    RBM( handle, visibleSize, hiddenSize )
{
    mParameters = parameters;
}


template<typename T>
RBM<T>::~RBM()
{
    cudaError_t cudaStatus = cudaFree( mCURANDStates );
    if ( cudaStatus != cudaSuccess )
    {
        CUDAErrorMessage::printErrorMessage( cudaStatus, __FILE__, __LINE__ );
    }
}


template<typename T>
int RBM<T>::getVisibleSize()
{
    return mVisibleSize;
}


template<typename T>
int RBM<T>::getHiddenSize()
{
    return mHiddenSize;
}


template<typename T>
const RBMParameters& RBM<T>::getParameters()
{
    return mParameters;
}


template<typename T>
const CUDAMatrix<T>& RBM<T>::getWeights()
{
    return mWeights;
}


template<typename T>
void RBM<T>::initWeights()
{
    // Initialize a weight matrix with random numbers
    CUDAMatrix<T>::fillRandom( mWeights );
    CUDAMatrix<T>::scale( 0.20, mWeights );

    // Insert weights for the bias units into the last row and last column
    CUDAMatrix<T>::fillRow( mWeights, mWeights.rows() - 1, 1);
    CUDAMatrix<T>::fillCol( mWeights, mWeights.cols() - 1, 1);
}


template<typename T>
void RBM<T>::train( const CUDAMatrix<T>& data )
{
    CUDAMatrix<T> biasedData( mCUBLASHandle, data.rows(), data.cols() + 1 );

    // Copy the values of data to biasedData (the last column is filled with 1)
    CUDAMatrix<T>::copy( data, biasedData );
    CUDAMatrix<T>::fillCol( biasedData, biasedData.cols() - 1, 1 );

    // Create intermediate matrices
    CUDAMatrix<T> weights( mWeights );
    CUDAMatrix<T> posHiddenActivations( mCUBLASHandle, biasedData.rows(), mWeights.cols() );
    CUDAMatrix<T>& posHiddenProbs = posHiddenActivations;//( mCUBLASHandle, biasedData.rows(), mWeights.cols() );
    CUDAMatrix<T>& posHiddenStates = posHiddenProbs;//( mCUBLASHandle, biasedData.rows(), mWeights.cols() );
    CUDAMatrix<T> posAssociations( mCUBLASHandle, biasedData.cols(), mWeights.cols() );
    CUDAMatrix<T> negVisibleActivations( mCUBLASHandle, biasedData.rows(), mWeights.rows() );
    CUDAMatrix<T>& negVisibleProbs = negVisibleActivations;//( mCUBLASHandle, biasedData.rows(), mWeights.rows() );
    CUDAMatrix<T>& negHiddenActivations = posHiddenStates;//( mCUBLASHandle, biasedData.rows(), mWeights.cols() );
    CUDAMatrix<T>& negHiddenProbs = negHiddenActivations;//( mCUBLASHandle, biasedData.rows(), mWeights.cols() );
    CUDAMatrix<T> negAssociations( mCUBLASHandle, mWeights.rows(), mWeights.cols() );

    // Initialize gpu random number generator
    if ( !mParameters.statesAsProbabilities )
    {
        setupRandState( biasedData.rows(), mWeights.cols() );
    }
    
//#ifdef _DEBUG
    CUDAMatrix<T> errorMatrix( mCUBLASHandle, data.rows(), data.cols() );
//#endif

    T bestError = std::numeric_limits<T>::max();
//    T prevError = std::numeric_limits<T>::max();
//    int accumulatedErrorCount = 0;
    for (int epoch = 0; epoch < mParameters.maxEpochs; epoch++)
    {
        // Clamp to the data and sample from the hidden units.
        // (This is the "positive CD phase", aka the reality phase.)
        CUDAMatrix<T>::mult( biasedData, false, weights, false, posHiddenActivations );
        
        // Compute logistic function
        gpuLogistic( posHiddenActivations.rows(), posHiddenActivations.cols(), posHiddenActivations.getDevicePtr(), posHiddenProbs.getDevicePtr() );
        
        // Note that we're using the activation *probabilities* of the hidden
        // states, not the hidden states themselves, when computing associations.
        // We could also use the states; see section 3 of Hinton's
        // "A Practical Guide to Training Restricted Boltzmann Machines" for more.
        CUDAMatrix<T>::mult( biasedData, true, posHiddenProbs, false, posAssociations );

        // Activate hidden states
        if ( !mParameters.statesAsProbabilities )
        {
            gpuActivateState( mCURANDStates, posHiddenProbs.rows(), posHiddenProbs.cols(), posHiddenProbs.getDevicePtr(), posHiddenStates.getDevicePtr() );
        }
        else
        {
            CUDAMatrix<T>::copy( posHiddenProbs, posHiddenStates );
        }

        // Reconstruct the visible units and sample again from the hidden units.
        // (This is the "negative CD phase", aka the daydreaming phase.)
        CUDAMatrix<T>::mult( posHiddenStates, false, weights, true, negVisibleActivations );

        // Compute logistic function
        gpuLogistic( negVisibleActivations.rows(), negVisibleActivations.cols(), negVisibleActivations.getDevicePtr(), negVisibleProbs.getDevicePtr() );

        // Fix the bias unit
        CUDAMatrix<T>::fillCol( negVisibleProbs, negVisibleProbs.cols() - 1, 1 );

        // Activate hidden states
        CUDAMatrix<T>::mult( negVisibleProbs, false, weights, false, negHiddenActivations );

        // Compute logistic function
        gpuLogistic( negHiddenActivations.rows(), negHiddenActivations.cols(), negHiddenActivations.getDevicePtr(), negHiddenProbs.getDevicePtr() );

        // Note, again, that we're using the activation *probabilities* when
        // computing associations, not the states themselves.
        CUDAMatrix<T>::mult( negVisibleProbs, true, negHiddenProbs, false, negAssociations);

        // Update weights.
        // mWeights += learningRate * ((posAssociations - negAssociations) / samplesSize)
        int samplesSize = data.rows();
        CUDAMatrix<T>::sub( posAssociations, false, negAssociations, false, posAssociations);
        CUDAMatrix<T>::scale( 1.0 / samplesSize, posAssociations );
        CUDAMatrix<T>::scale( mParameters.learningRate, posAssociations );
        CUDAMatrix<T>::add( weights, false, posAssociations, false, weights );

        CUDAMatrix<T>::copy( negVisibleProbs, errorMatrix );

        CUDAMatrix<T>::sub( data, false, errorMatrix, false, errorMatrix );

        T error = CUDAMatrix<T>::norm( errorMatrix );

               
        // Check if the actual weight is the best till now and update.
        if ( error < bestError )
        {
            CUDAMatrix<T>::copy( weights, mWeights );
            bestError = error;
        }

        // If 5 consecutive errors was less than maxError then stop.
//        accumulatedErrorCount = fabs( prevError - error ) <= mParameters.maxError ? accumulatedErrorCount + 1 : 0;
//        if( accumulatedErrorCount == 5 )
//        {
//            break;
//        }
        
//        std::cout << "Epoch " << epoch << ": MSE error = " << (error*error) / errorMatrix.rows()  << "\n";
        
        
//        prevError = error;
    }
}


template<typename T>
void RBM<T>::runVisible( const CUDAMatrix<T>& data, CUDAMatrix<T>& hidden )
{
    // Insert bias units of 1 into the last column of data.
    CUDAMatrix<T> biasedData( mCUBLASHandle, data.rows(), data.cols() + 1 );
    CUDAMatrix<T>::copy( data, biasedData );
    CUDAMatrix<T>::fillCol( biasedData, biasedData.cols() - 1, 1 );

    // Calculate the activations of the hidden units.
    CUDAMatrix<T> hiddenUnits( mCUBLASHandle, biasedData.rows(), mWeights.cols() );
    CUDAMatrix<T>::mult( biasedData, false, mWeights, false, hiddenUnits );
    
    // Calculate the probabilities of turning the hidden units on.
    gpuLogistic( hiddenUnits.rows(), hiddenUnits.cols(), hiddenUnits.getDevicePtr(), hiddenUnits.getDevicePtr() );
    
    if ( !mParameters.statesAsProbabilities )
    {
        setupRandState( biasedData.rows(), mWeights.cols() );

        // Turn the hidden units on with their specified probabilities.
        gpuActivateState( mCURANDStates, hiddenUnits.rows(), hiddenUnits.cols(), hiddenUnits.getDevicePtr(), hiddenUnits.getDevicePtr() );
    }

    // Ignore the bias units.
    CUDAMatrix<T>::copy( hiddenUnits, hidden );
}


template<typename T>
void RBM<T>::runHidden( const CUDAMatrix<T>& data, CUDAMatrix<T>& visible )
{
    // Insert bias units of 1 into the last column of data.
    CUDAMatrix<T> biasedData( mCUBLASHandle, data.rows(), data.cols() + 1 );
    CUDAMatrix<T>::copy( data, biasedData );
    CUDAMatrix<T>::fillCol( biasedData, biasedData.cols() - 1, 1 );

    // Calculate the activations of the visible units.
    CUDAMatrix<T> visibleUnits( mCUBLASHandle, biasedData.rows(), mWeights.rows() );
    CUDAMatrix<T>::mult( biasedData, false, mWeights, true, visibleUnits );
    
    // Calculate the probabilities of turning the visible units on.
    gpuLogistic( visibleUnits.rows(), visibleUnits.cols(), visibleUnits.getDevicePtr(), visibleUnits.getDevicePtr() );
    
    if ( !mParameters.statesAsProbabilities )
    {
        setupRandState( visibleUnits.rows(), visibleUnits.cols() );

        // Turn the visible units on with their specified probabilities.
        gpuActivateState( mCURANDStates, visibleUnits.rows(), visibleUnits.cols(), visibleUnits.getDevicePtr(), visibleUnits.getDevicePtr() );
    }

    // Ignore the bias units.
    CUDAMatrix<T>::copy( visibleUnits, visible );
}


template<typename T>
void RBM<T>::setupRandState( int rows, int cols )
{
    if ( rows < 1 || cols < 1 )
    {
        return;
    }
    
    if ( mCURANDStates == 0 )
    {
        cudaError_t cudaStatus = cudaMalloc( (void **)&mCURANDStates, rows * cols * sizeof(mCURANDStates) );
        if ( cudaStatus != cudaSuccess )
        {
            CUDAErrorMessage::printErrorMessage( cudaStatus, __FILE__, __LINE__ );
            return;
        }
    }
    
    seedRandomGenerator( rows, cols, mCURANDStates, time(0) );
}


template<typename T>
double RBM<T>::computeMSEError( const CUDAMatrix<T>& data )
{
    CUDAMatrix<T> errorMatrix( mCUBLASHandle, data.rows(), data.cols() );
    CUDAMatrix<T> hidden( mCUBLASHandle, data.rows(), mHiddenSize );
    
    runVisible( data, hidden );
    runHidden( hidden, errorMatrix );
    
    CUDAMatrix<T>::sub( data, false, errorMatrix, false, errorMatrix );
    float error = CUDAMatrix<T>::norm( errorMatrix );
        
    return (error * error) / errorMatrix.rows();
}

}
#endif	// RBM_H
