/* 
 * File:   CUDANeuralNetwork.cpp
 * Author: ederperez
 *
 * Created on November 12, 2015, 1:43 PM
 */

#include <iostream>
#include <vector>
#include "CUDANeuralNetwork.h"


//namespace gpu
//{
//
//__global__ void _gpuSigmoid( int size, const float* src, float* dst )
//{
//    const int i = blockIdx.x*blockDim.x + threadIdx.x;
//    
//    if( i < cols && j < rows )
//    {
//        dst[i] = 1.0 / ( 1.0 + exp(-src[index]) );
//    }
//}
//
//
//}





























void CUDANeuralNetwork::hiddenGradient( int layer )
{
    if ( layer < 1 || layer >= static_cast<int>(_layers.size()) - 1 )
    {
        return;
    }

    // Loop for each of the neurons in the Net's selected Layer
    for( int k = 0 ; k < _layers[layer]; k++ )
    {
        T sum = 0.0;

        // Loop for each of the bindings between the selected neuron and the neurons of the next Layer
        for( int l = 0; l < _layers[layer+1]; l++ )
        {
            sum += _weights[layer](l, k) * _deltas[layer+1](l);
        }

        T delta = sum * activationDerivative( _neurons[layer](k) );

        // Saving this Layer's deltas to train the previous Layer, if there is one
        _deltas[layer](k) = delta;
        
        // Updating Gradients
        for ( int c = 0; c < _neurons[layer-1].size(); c++ )
        {
            _gradients[layer-1](k, c) = _neurons[layer-1](c) * delta;
        }
    }
}


void CUDANeuralNetwork::outGradientMtm( const std::vector<T>& targets, T MFactor )
{
    int lastLayer = _layers.size() - 1;
    
    // Loop for each of the Net's output neurons
    for( int i = 0; i < _layers.back(); i++ )
    {
        // Calculates the delta value for the index i neuron
        T delta = ( targets[i] - _neurons[lastLayer](i) ) * activationDerivative( _neurons[lastLayer](i) );
        
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
                T auxGrad = _neurons[lastLayer-1](c) * delta;
                _gradients[lastLayer-1](i, c) = (MFactor * auxGrad) + ( _gradientsMt[lastLayer-1](i, c) * (1 - MFactor) );
                _gradientsMt[lastLayer-1](i, c) = auxGrad;
            }
        }
    }
}


int CUDANeuralNetwork::hiddenGradientMtm( int layer, T MFactor )
{
    if ( layer < 1 || layer >= static_cast<int>(_layers.size()) - 1 )
    {
        return -1;
    }

    // Loop for each of the neurons in the Net's selected Layer
    for( int k = 0; k < _layers[layer]; k++ )
    {
        T sum = 0.0;
        
        // Loop for each of the bindings between the selected neuron and the neurons of the next Layer
        for( int l = 0; l < _layers[layer+1]; l++ )
        {
            sum += _weights[layer](l, k) * _deltas[layer+1](l);
        }
        
        T delta = sum * activationDerivative( _neurons[layer](k) );
        
        // Saving this Layer's deltas to train the previous Layer, if there is one
        _deltas[layer](k) = delta;
        
        // Updating Gradients
        if( _flagMomentum == 0 )
        {
            // Loop for each of the bindings between the selected neuron and the neurons of the next Layer
            for ( int c = 0; c < _neurons[layer-1].size(); c++ )
            {
                _gradients[layer-1](k, c) = _neurons[layer-1](c) * delta;
            }
        }
        else
        {
            for ( int c = 0; c < _neurons[layer-1].size(); c++ )
            {
                T auxGrad = delta * _neurons[layer-1](c);
                _gradients[layer-1](k, c) =  (MFactor * auxGrad) + ( _gradientsMt[layer-1](k, c) * (1 - MFactor) );
                _gradientsMt[layer-1](k, c) = auxGrad;
            }
        }
    }
    
    return 0;
}


int CUDANeuralNetwork::updateWeights( T rate )
{
    if ( rate <= 0.0 || rate > 1.0 )
    {
        return -1;
    }

    for( int layerId = _layers.size()-2; layerId >= 0; layerId-- )
    {
        for( int neuronId = 0; neuronId < _layers[layerId+1]; neuronId++ )
        {
            for( int weightId = 0; weightId <= _layers[layerId]; weightId++ )
            {
                _weights[layerId](neuronId, weightId) += rate * _gradients[layerId](neuronId, weightId);
            }
        }
    }
    
    return 0;
}


int CUDANeuralNetwork::train( const std::vector<T>& entry, std::vector<T>& out, T l_rate, T momentum )
{
    // Inserting the entries: */
    if( setEntry( entry ) == -1 )
    {
            return -1;
    }

    // Getting the wrong out values
    activateAll();

    // Training the layer before the out layer
    if( momentum != 0)
    {
        outGradientMtm( out, momentum );
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
    updateWeights( l_rate );

    return 0;
}


int CUDANeuralNetwork::use( const std::vector<T>& entry )
{
    // Inserting the entries
    if( setEntry( entry) != 0 )
    {
        return -1;
    }

    // Getting the wrong out values
    activateAll();

    return 0;
}


int CUDANeuralNetwork::getOut( std::vector<T>& out )
{
    out.clear();
    out.reserve( _layers.back() );
    int lastLayer = _layers.size() - 1;
    for ( int count = 0; count < _layers.back(); count++ )
    {
        out.push_back( _neurons[lastLayer](count) );
    }
    
    return 0;
}

void CUDANeuralNetwork::RMSError( MatrixNxM& inMatrix, MatrixNxM& outMatrix, T* retRMS_Error )
{
    std::vector<T> inSample;
    inSample.resize(inMatrix.cols());
    std::vector<T> outSample;
    outSample.resize(outMatrix.cols());

    T thisOutput = 0;
    
    // Executando os treinamentos obrigatorios
    for( unsigned int thisSample = 0; thisSample < inMatrix.rows(); thisSample++ )
    {
        for( unsigned int cont = 0; cont < inMatrix.cols(); cont++ )
        {
            inSample[cont] = (T)(inMatrix( thisSample, cont ));
        }

        for( unsigned int cont = 0; cont < outMatrix.cols(); cont++ )
        {
            outSample[cont] = (T)(outMatrix( thisSample, cont ));
        }

        use( inSample );
        thisOutput += outputError( outSample );
    }

    *retRMS_Error = (thisOutput/inMatrix.rows());
}


void CUDANeuralNetwork::create( const std::vector<int>& layers )
{
    _neurons.clear();
    _deltas.clear();
    _weights.clear();
    _gradients.clear();
    _gradientsMt.clear();
    
    // Allocating space to store neurons's output values of each layer, and to deltas
    _neurons.reserve( layers.size() );
    _deltas.reserve( layers.size() );
    for ( size_t i = 0; i < layers.size(); i++ )
    {
        // +1 for bias
        _neurons.push_back( VectorN( layers[i] + 1) );
        _deltas.push_back( VectorN( layers[i] + 1) );
    }
    
    // Allocating space to store the neuron's binding Weights from Layer n to Layer n+1.
    // Allocating space to store the Gradients related to the Weights of the bindings between neurons
    _weights.reserve( layers.size() - 1 );
    _gradients.reserve( layers.size() - 1 );
    _gradientsMt.reserve( layers.size() - 1 );
    for( size_t i = 0; i < layers.size() - 1; i++ )
    {
        int nRows = layers[i + 1];
        int nColumns = layers[i] + 1; // +1 for bias
        
        _weights.push_back( MatrixNxM( nRows, nColumns ) );
        _gradients.push_back( MatrixNxM( nRows, nColumns ) );
        _gradientsMt.push_back( MatrixNxM( nRows, nColumns ) );
    }
}
