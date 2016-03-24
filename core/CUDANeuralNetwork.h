/* 
 * File:   CUDANeuralNetwork.h
 * Author: ederperez
 */

#ifndef CUDANEURALNETWORK_H
#define	CUDANEURALNETWORK_H

#include <vector>
#include "CUDAMatrix.h"
#include "CUDAVector.h"


namespace gpu
{


/**
 * This class is used to set network parameters.
 */
class CUDANeuralNetworkParameters
{
    public:
        CUDANeuralNetworkParameters() :
            learningRate(0.1),
            maxEpochs(1000),
            maxError(1e-3),
            momentum()
        {
        }

        float learningRate;
        int maxEpochs;
        int maxError;
        float momentum;
};


template<typename T>
class CUDANeuralNetwork
{
    public:

        /**
         * Constructor.
         * @param layers Array containing the number of neurons on each layer.
         * @param parameters Parameters for training neural network.
         */
        CUDANeuralNetwork( const std::vector<int>& layers );
        CUDANeuralNetwork( const std::vector<int>& layers, const CUDANeuralNetworkParameters& parameters );
        
        /**
	 * Inserts random numbers in range [0.0, 1) as weight values.
	 */
        /**@OLD*/void init();
        
        /**
         * Insert values into weight matrix.
	 * @param inputLayer The layer that corresponds to the input for applying
         * the weights to in order to compute the next layer.
         * @param weights Corresponds to the weights values.
         */
	void setWeights( int inputLayer, const CUDAMatrix& weights );
        
        /**
	 * Get values from the weight.
	 */
	void getWeights( int inputLayer, CUDAMatrix& weights );
        
        /**
         * Get the network structure.
         */
        const std::vector<int>& getNetworkStructure()
        {
            return mLayers;
        }
        
        /**
	 * Example: for a 3-Layer Net, if I want to train the third Layer
	 * I just need to call activateCalculus( Net, 2 ); -> the index begins with zero!!!
	 * The first Layer (Layer zero) doesn't have Weights before it, right???
	 */
	/**@OLD*/void activateCalculus( int layer );
        
        /**
	 * Function used to calculate the activation function value for any input.
	 */
	/**@OLD*/T activationFunction( const CUDAVector& src, CUDAVector& dst );
        
        /** 
	 * Function used to find the activation outputs for all the neurons of the Net.
	 */
	/**@OLD*/void activateAll();
        
        /**
	 * Function that calculates the Net's output error
	 * @param targets Neuron's targeted output.
	 * @return Returns the resulting error.
	 */
	/**@OLD*/T outputError( const std::vector<T>& targets );
        
        /**
	 * Function that inserts entry values on the input Layer of the Net.
	 */
	/**@OLD*/int setEntry( const std::vector<T>& entry );
        
        /**
	 * To train the neural Net, we'll need the activation function derivative's value of any
	 * point. This function returns this derivative.
	 * @param f Input value. The activation function derivative will be calculated for this value.
	 * @return Returns the value of the activation function derivative.
	 */
	/**@OLD*/T activationDerivative( T f )
        {
            return f * (1 - f);
        }
        
        /**
	 * Function used to calculate the gradients of the last Layer of neurons.
	 * @param PtNet Net Pointer to the Net in use.
	 * @param targets Target vector (desired outputs).
	 * @param dimTargets Dimension of the target vector.
	 * @return Returns -2 if Net points to NULL, -1 if dimTargets is less than zero, or 0 if ok.
	 */
	/**@OLD*/void outGradient( const std::vector<T>& targets );
        
        /**
	 * Function used to train the hidden Layers's neurons.
	 * @param Net Net being used.
	 * @param Layer Layer to receive the Gradients based on the already calculated Weights.
	 * @return Returns -2 if Net points to NULL, -1 if dimTargets is less than zero, or 0 if ok.
	 */
	/**@OLD*/void hiddenGradient( int layer );
        
        /**
	 * Function used to calculate the gradients of the last Layer of neurons.
	 * @param targets Target vector (desired outputs).
	 * @param dimTargets Dimension of the target vector.
	 * @param MFactor Momentum factor.
	 */
	/**@OLD*/void outGradientMtm( const std::vector<T>& targets, T MFactor );
        
        /**
	 * Function used to train the hidden Layers's neurons.
	 * @param Layer Layer to receive the Gradients based on the already calculated Weights.
	 * @param MFactor Momentum factor.
	 * @return Returns -2 if Net points to NULL, -1 if dimTargets is less than zero, or 0 if ok.
	 */
	/**@OLD*/int hiddenGradientMtm( int layer, T MFactor );
        
        /**
	 * Function that update the Weights on the Net.
	 * @param Net Net being used.
	 * @param rate The Learning rate.
	 * @return Returns -2 if Net points to NULL, -1 if the value of rate is not between 0.0 and 1.0, or 0.
	 */
	/**@OLD*/int updateWeights( T rate );
        
        /**
	 * Recebe um vetor de tamanho n e desloca os valores 1 posicao a esquerda.
	 * O primeiro valor (o valor contido na posicao de indice 0 do vetor) sera deletado.
	 */
	/**@OLD*/bool shiftLeft( CUDAVector<T>& vector, T newValue, int newValuePosition );
        
        /**
	 * Function used to train the net.
	 * @param size The size of entry vector.
	 * @param entry CUDAVector<T> that contains values of entries.
	 * @param size2 The size of out vector.
	 * @param out CUDAVector<T> that contains values of outs.
	 * @param l_rate The learning rate.
	 * @param momentum Momentum factor.
	 * @return returns 1 if ok.
	 */
	/**@OLD*/int train( const std::vector<T>& entry, std::vector<T>& out, T l_rate, T momentum );
        
        /**
	 * Once the net is ready, we need a function to use it.
	 * @param size Length of the entry vector.
	 * @param entry The entries vector.
	 * @return It returns 0 if ok.
	 */
	/**@OLD*/int use( const std::vector<T>& entry );
        
        /**
	 * Function that gets out values from the Net.
	 */
	/**@OLD*/int getOut( std::vector<T>& out );
        
        /**
	 * Obtem o erro medio quadratico de um conjunto de amostras fornecido.
	 */
	/**@OLD*/void RMSError( CUDAMatrix<T>& inMatrix, CUDAMatrix<T>& outMatrix, T* retRMS_Error );

    private:
        
        /**
         * Execute a forward pass in the neural network given a batch of inputs.
         * 
         * @param input A matrix in which each row is an input.
         * @param input A matrix in which each row is an output.
         */
        void forwardPass( const CUDAMatrix& input, CUDAMatrix& output );
        
        /**
         * Parameters for training.
         */
        CUDANeuralNetworkParameters mParameters;

        /**
         * Function used to create a new neural Net.
         */
        /**@OLD*/void create( const std::vector<int>& layers );
       
        cublasHandle_t mCUBLASHandle;
        
        /**
         * Layer structure containing the number of neurons on each layer.
         */
        std::vector<int> mLayers;
        
        /**
         * Output value of each neuron.
         */
        /**@OLD*/std::vector< CUDAVector<T> > mNeurons;
        
        /**
         * Auxiliary vector to save temporary deltas while training the net.
         */
        /**@OLD*/std::vector< CUDAVector<T> > mDeltas;
        
        /**
         * Connection Weights of each neuron from Layer n to Layer n+1.
         */
        std::vector< CUDAMatrix<T> > mWeights;
        
        /**
         * Gradients of each Weight.
         */
        /**@OLD*/std::vector< CUDAMatrix<T> > mGradients;
        
        /**
         * Temporary Gradients of each Weight (Momentum).
         */
        /**@OLD*/std::vector< CUDAMatrix<T> > mGradientsMt;
        
        /**
         * Auxiliary flag to indicate this is not the first train.
         */
        /**@OLD*/int mFlagMomentum;
};


template<typename T>
CUDANeuralNetwork<T>::CUDANeuralNetwork( const std::vector<int>& layers ) :
    mLayers(layers),
    mFlagMomentum(0)
{
    create( layers );
    init();
}


template<typename T>
CUDANeuralNetwork<T>::CUDANeuralNetwork( const std::vector<int>& layers, const CUDANeuralNetworkParameters& parameters ) :
    CUDANeuralNetwork( const std::vector<int>& layers ) :
{
    mParameters = parameters;
}


template<typename T>
void CUDANeuralNetwork<T>::init()
{
    if ( mLayers.size() < 2 )
    {
        return;
    }

    // Initializing Weights and Gradients
    for ( size_t i = 0; i < mWeights.size(); i++ )
    {
        CUDAMatrix::fillRandom( mWeights[i] );
        CUDAMatrix::scale( 0.0, mGradients[i] );
        CUDAMatrix::scale( 0.0, mGradientsMt[i] );
    }

    // Inserting random in the neurons's output values
    for ( size_t i = 0; i < mNeurons.size(); i++ )
    {
        CUDAVector::fillRandom( mNeurons[i] );
        CUDAVector::fillRange( 1.0, mNeurons[i], mNeurons[i].size() - 1, mNeurons[i].size() );
    }
}


template<typename T>
void CUDANeuralNetwork<T>::setWeights( int inputLayer, const CUDAMatrix& weights )
{
    if( inputLayer >= 0 && inputLayer < mLayers.size() )
    {
        CUDAMatrix::copy( weights, mWeights[inputLayer] );
    }
}


template<typename T>
void CUDANeuralNetwork<T>::getWeights( int inputLayer, CUDAMatrix& weights )
{
    if( inputLayer >= 0 && inputLayer < mLayers.size() )
    {
        CUDAMatrix::copy( mWeights[inputLayer], weights );
    }
}


template<typename T>
void CUDANeuralNetwork::activateCalculus( int layer )
{
    if( layer > 0 && layer < mLayers.size() )
    {
        CUDAVector n( mNeurons[layer].size() );
        
        CUDAMatrix::mult( mWeights[layer - 1], false, mNeurons[layer - 1], n );
        activationFunction( n, mNeurons[layer] );
    }
}


template<typename T>
void CUDANeuralNetwork<T>::activateAll()
{
    // Calculates the neuron's activations in each Layer
    for( size_t layer = 1; layer < mLayers.size(); layer++ )
    {
        activateCalculus( layer );
    }
}


template<typename T>
T CUDANeuralNetwork<T>::outputError( const std::vector<T>& targets )
{
    //@TODO
//    // Initializing counter
//    T sum = 0.0;
//    
//    // outLayerId will receive the Net's number of Layers: 
//    int outLayerId = mLayers.size() - 1;
//    std::vector<T> neurons;
//    mNeurons[outLayerId].getData( neurons );
//    
//    for ( size_t i = 0; i < neurons.size(); i++ )
//    {
//        T d = targets[i] - neurons[i];
//        sum += d * d;
//    }
//
//    // Medium error
//    return sum / static_cast<T(neurons.size());
}


template<typename T>
int CUDANeuralNetwork<T>::setEntry( const std::vector<T>& entry )
{
    if( entry.size() == static_cast<size_t>(mLayers[0]) )
    {
        std::vector<T> entryBiased( entry.size() + 1 );
        
        std::copy( entry.begin(), entry.end(), entryBiased.begin() );
        entryBiased.back() = 1.0;
        mNeurons[0].setData( entryBiased );
    }
}


template<typename T>
void CUDANeuralNetwork<T>::outGradient( const std::vector<T>& targets )
{
    int lastLayer = static_cast<int>(mLayers.size()) - 1;
    int l = static_cast<int>(mLayers.size()) - 2;
    for ( int i = 0; i < mLayers.back(); i++ )
    {
        // Calculates the delta value for the index i neuron
        T delta = ( targets[i] - mNeurons[lastLayer](i) ) * activationDerivative( mNeurons[lastLayer](i) );

        mDeltas[lastLayer](i) = delta;

        for ( int c = 0; c < mNeurons[l].size(); c++ )
        {
            mGradients[l](i, c) = mNeurons[l](c) * delta;
        }
    }
}


template<typename T>
void CUDANeuralNetwork<T>::train( const CUDAMatrix& input, const CUDAMatrix& targets, CUDAMatrix& output )
{
    // Check parameters
    if ( mLayers.size() < 2 || input.cols() != mLayers[0] || output.cols() != mLayers.back() )
    {
        return;
    }
    
    CUDAMatrix<T> deltas( mCUBLASHandle, output.rows(), output.cols() );
    
    int epochs = mParameters.maxEpochs;
    while ( epochs >= 0 )
    {
        forwardPass( input, output );
        
        // Compute deltas
        // @TODO: computeDeltasForLastLayer( targets, output, deltas );
        //          CUDAMatrix<T>::sub( targets, false, output, false, deltas );
        //          deltas = deltas * ( output * (1 - output) );
        
        // Training the layer before the out layer
        if ( fabs(mParameters.momentum) > 1e-10 )
        {
            int lastLayer = mLayers.size() - 1;

            // Loop for each of the Net's output neurons
            for( int i = 0; i < mLayers.back(); i++ )
            {
                // Stores the delta value in the delta vector
                _deltas[lastLayer](i) = delta;

                if( _flagMomentum == 0)
                {
                    // Loop for each of the bindings to the previous Layer
                    for ( int c = 0; c < _neurons[lastLayer-1].size(); c++ )
                    {
                        _gradients[lastLayer-1](i, c) = _neurons[lastLayer-1](c) * delta;
                        _gradientsMt[lastLayer-1](i, c) = _neurons[lastLayer-1](c) * delta;
                    }

                    _flagMomentum = 1;
                }
                else // In this case, it's not the first train
                {
                    // Loop for each of the bindings to the previous Layer
                    for ( int c = 0; c < _neurons[lastLayer-1].size(); c++ )
                    {
                        FLOAT auxGrad = _neurons[lastLayer-1](c) * delta;
                        _gradients[lastLayer-1](i, c) = (MFactor * auxGrad) + ( _gradientsMt[lastLayer-1](i, c) * (1 - MFactor) );
                        _gradientsMt[lastLayer-1](i, c) = auxGrad;
                    }
                }
            }
        }
        else
        {
            outGradient( out );
        }
        
        // Training the net for each hidden layer
        for( int i = _layers.size() - 2; i > 0; i-- )
        {
            if( momentum != 0)
            {
                hiddenGradientMtm( i, momentum );
            }
            else
            {
                hiddenGradient( i );
            }
        }

        // Updating the weights
        updateWeights( learningRate );
        
        --epochs;
    }
}


template<typename T>
void CUDANeuralNetwork<T>::forwardPass( const CUDAMatrix& input, CUDAMatrix& output )
{
    // Check parameters
    if ( mLayers.size() < 2 || input.cols() != mLayers[0] || output.cols() != mLayers.back() )
    {
        return;
    }
    
    // Copy the values of input to biasedInput (the last column is filled with 1)
    CUDAMatrix<T> biasedInput = new CUDAMatrix<T>( mCUBLASHandle, input.rows(), input.cols() + 1 );
    CUDAMatrix<T>::copy( input, *biasedInput );
    CUDAMatrix<T>::fillCol( *biasedInput, biasedInput->cols() - 1, 1 );
    
    // mapA and mapB holds matrices representing layers for each input data.
    // In order to reduce gpu memory allocation, up to 2 matrices with same dimensions
    // are allocated.
    std::map<CUDAMatrix*> mapLayersA, mapLayersB;
    
    for ( auto& it : mLayers )
    {
        mapLayersA[it] = nullptr;
        mapLayersB[it] = nullptr;
    }
    
    mapLayersA[mLayers[0]] = biasedInput;
    mapLayersB[mLayers[1]] = new CUDAMatrix( mCUBLASHandle, biasedInput->rows(), mLayers[1] + 1 );
    
    CUDAMatrix<T>* prevLayer = mapLayersA[ mLayers[0] ];
    CUDAMatrix<T>* nextLayer = mapLayersB[ mLayers[1] ];
    
    // Compute activation units for each layer
    for ( int i = 0; i < static_cast<int>(mLayers.size()) - 1; i++ )
    {
        CUDAMatrix<T>::mult( *prevLayer, true, mWeights[i], true, *nextLayer );
        
        // Apply sigmoid function
        gpuLogistic( nextLayer->rows(), nextLayer->cols(), nextLayer->getDevicePtr(), nextLayer->getDevicePtr() );
        
        prevLayer = nextLayer;
        
        // This test avoids unnecessary computation at last iteration
        if ( i != mLayers.size() - 2)
        {
            CUDAMatrix<T>::fillCol( prevLayer, prevLayer->cols() - 1, 1 );

            // Get the next available matrix
            
            // If mapLayersA is not available, use mapLayersB
            if ( mapLayersA[mLayers[i+1]] == prevLayer )
            {
                if ( mapLayersB[mLayers[i+1]] == nullptr )
                {
                    mapLayersB[mLayers[i+1]] = new CUDAMatrix( mCUBLASHandle, biasedInput->rows(), mLayers[i+1] + 1 );
                }
                nextLayer = mapLayersB[mLayers[i+1]];
            }
            // ... else, check if mapLayersB is available or create a new matrix in mapLayersA
            else
            {
                if ( mapLayersB[mLayers[i+1]] != prevLayer && mapLayersB[mLayers[i+1]] != nullptr )
                {
                    nextLayer = mapLayersB[mLayers[i+1]];
                }
                else if ( mapLayersA[mLayers[i+1]] != nullptr )
                {
                    nextLayer = mapLayersA[mLayers[i+1]];
                }
                else
                {
                    mapLayersA[mLayers[i+1]] = new CUDAMatrix( mCUBLASHandle, biasedInput->rows(), mLayers[i+1] + 1 );
                    nextLayer = mapLayersA[mLayers[i+1]];
                }
            }
        }
    }

    // At this point, nextLayer points to the output with bias.
    // Copy nextLayer to output excluding bias.
    CUDAMatrix<T>::copy( *nextLayer, output );
    
    // Release matrices
    for ( auto& it : mapLayersA )
    {
        delete it.second;
    }
    
    for ( auto& it : mapLayersB )
    {
        delete it.second;
    }
}

}

#endif	/* CUDANEURALNETWORK_H */
