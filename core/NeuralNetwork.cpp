/* 
 * File:   NeuralNetwork.cpp
 * Author: ederperez
 *
 * Created on November 12, 2015, 1:43 PM
 */

#include <iostream>
#include <vector>
#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork( const std::vector<int>& layers ) :
    _layers(layers),
    _flagMomentum(0)
{
    create( layers );
    init();
}


NeuralNetwork::NeuralNetwork( const std::string& path )
{
//    if ( path.empty() )
//    {
//        return;
//    }
//    
//    int i, j;
//    int lixo;
//    int layers, *nLayers;
//    FILE *f;
//    char name[128];
//    char namebin[128];
//
//    /* Checking arguments: */
//    if ( filename == NULL )
//    {
//        return;
//    }
//
//    /* Inserting the correct extensions: */
//    sprintf( name, "%s.nnt", filename );
//    sprintf( namebin, "%s.bin", filename );
//
//    f = fopen(name, "r");
//    if ( f == NULL )
//    {
//        return;
//    }
//
//    /* Number of layers: */
//    fscanf( f, "Number of layers: %d\n", &layers);
//
//    /* Space to the number of neurons on each layer: */
//    nLayers = (int*)malloc( layers * sizeof(int) );
//    if( nLayers == NULL )
//    {
//        return;
//    }
//
//    /* Number of neurons on each layer: */
//    for( i = 0; i < layers; i++ )
//    {
//        fscanf( f, "Neurons on layer %d: %d\n", &lixo, &(nLayers[i]) );
//    }
//
//    //BKPNeuralNet( layers, nLayers );
//    CreateNet( layers, nLayers );
//
//    /* Closing the text file: */
//    fclose( f );
//
//    /* Opening the binary file: */
//    f = fopen( namebin, "rb" );
//    if( f == NULL )
//    {
//        return;
//    }
//
//    /* Values of the neurons: */
//    for( i = 0; i < _layers; i++ )
//    {
//        fread( _neurons[i] , sizeof(FLOAT), _nLayers[i], f );
//    }
//
//    /* Weight values */
//    for( i = _layers - 2; i >= 0; i-- )
//    {
//        for( j = 0; j <= _nLayers[i+1]; j++ )
//        {
//            fread( _weights[i][j], sizeof(FLOAT), _nLayers[i]+1, f );
//        }
//    }
//
//    /* Closing file: */
//    fclose( f );
}


void NeuralNetwork::init()
{
    if ( _layers.size() < 2 )
    {
        return;
    }
    
    // Gerando uma semente nova, para os numeros aleatorios
    srand( (unsigned)time( NULL ) );

    // Initializing Weights and Gradients
    for ( int k = static_cast<int>(_layers.size()) - 2; k >= 0; k-- )
    {
        int nRows = _layers[k + 1];
        int nColumns = _layers[k] + 1; // +1 for bias
        for ( int i = 0; i < nRows; i++ )
        {
            for ( int j = 0; j < nColumns; j++ )
            {
                // Inserting random values for the Weights
                _weights[k](i, j) = ( (FLOAT)rand()/(FLOAT)RAND_MAX ) / _layers[k];
                
                assert( !isnan( _weights[k](i, j) ) );
                
                // Inserting 0.0 in Gradients
                _gradients[k](i, j) = 0.0;
                _gradientsMt[k](i, j) = 0.0;
            }
        }
    }

    // Inserting random in the neurons's output values
    for ( size_t i = 0; i < _layers.size(); i++ )
    {
        for ( int j = 0; j < _layers[i]; j++ )
        {
            _neurons[i](j) = ((FLOAT)rand()/(FLOAT)RAND_MAX);
        }
        _neurons[i](_layers[i]) = 1.0;
    }
}


int NeuralNetwork::setWeights( int outLayer, int neuronId, int prevLayerNeuronId, FLOAT value )
{
    if( outLayer == 0 )
    {
        return -1;
    }

    _weights[outLayer-1](neuronId, prevLayerNeuronId) = value;
    return 0;
}


void NeuralNetwork::setWeights( int inputLayer, const MatrixNxM& weights )
{
    if ( ( inputLayer >= 0 && inputLayer < static_cast<int>(_layers.size()) - 1 ) &&
            weights.rows() == _weights[inputLayer].rows() && weights.cols() == _weights[inputLayer].cols() )
    {
        _weights[inputLayer] = weights;
    }
}


int NeuralNetwork::getWeights( int outLayer, int neuronId, int prevLayerNeuronId, FLOAT& value )
{
    if( outLayer == 0 )
    {
        return -1;
    }

    value = _weights[outLayer-1](neuronId, prevLayerNeuronId);
    return 0;
}


MatrixNxM NeuralNetwork::getWeights( int weightIndex )
{
    return _weights[weightIndex];
}


int NeuralNetwork::activateCalculus( int layer )
{
    if( layer == 0 )
    {
        return -1; 
    }
    
    VectorN n = _weights[layer - 1] * _neurons[layer - 1];
    
    for ( int i = 0; i < n.rows(); i++ )
    {
        assert( !isnan( n(i) ) );

        _neurons[layer](i) = activationFunc( n(i) );
    }
    
    return 0;
}


void NeuralNetwork::activateAll()
{
    // Calculates the neuron's activations in each Layer
    for( size_t layer = 0; layer < _layers.size(); layer++ )
    {
        activateCalculus( layer );
    }
}


FLOAT NeuralNetwork::outputError( const std::vector<FLOAT>& targets )
{
    // Initializing counter
    FLOAT sum = 0.0;
    
    // outLayerId will receive the Net's number of Layers: 
    int outLayerId = _layers.size() - 1;
    
    for ( int i = 0; i < _layers[outLayerId]; i++ )
    {
        FLOAT d = targets[i] - _neurons[outLayerId](i);
        sum += d * d;
    }

    // Medium error
    return sum / (FLOAT) _layers[outLayerId];
}


int NeuralNetwork::setEntry( const std::vector<FLOAT>& entry )
{
    if( (int)(entry.size()) != _layers[0] )
    {
        return -1;
    }

    for( size_t count = 0; count < entry.size(); count++ )
    {
        // Inserting correct values
        _neurons[0](count) = entry[count];
    }
    
    // For bias
    _neurons[0](_neurons[0].size()-1) = 1.0;

    return 0;
}


void NeuralNetwork::outGradient( const std::vector<FLOAT>& targets )
{
    int lastLayer = _layers.size() - 1;
    int l = _layers.size() - 2;
    for ( int i = 0; i < _layers.back(); i++ )
    {
        // Calculates the delta value for the index i neuron
        FLOAT delta = ( targets[i] - _neurons[lastLayer](i) ) * activationDerivative( _neurons[lastLayer](i) );

        _deltas[lastLayer](i) = delta;

        for ( int c = 0; c < _neurons[l].size(); c++ )
        {
            _gradients[l](i, c) = _neurons[l](c) * delta;
        }
    }
}


void NeuralNetwork::hiddenGradient( int layer )
{
    if ( layer < 1 || layer >= static_cast<int>(_layers.size()) - 1 )
    {
        return;
    }

    // Loop for each of the neurons in the Net's selected Layer
    for( int k = 0 ; k < _layers[layer]; k++ )
    {
        FLOAT sum = 0.0;

        // Loop for each of the bindings between the selected neuron and the neurons of the next Layer
        for( int l = 0; l < _layers[layer+1]; l++ )
        {
            sum += _weights[layer](l, k) * _deltas[layer+1](l);
        }

        FLOAT delta = sum * activationDerivative( _neurons[layer](k) );

        // Saving this Layer's deltas to train the previous Layer, if there is one
        _deltas[layer](k) = delta;
        
        // Updating Gradients
        for ( int c = 0; c < _neurons[layer-1].size(); c++ )
        {
            _gradients[layer-1](k, c) = _neurons[layer-1](c) * delta;
        }
    }
}


void NeuralNetwork::outGradientMtm( const std::vector<FLOAT>& targets, FLOAT MFactor )
{
    int lastLayer = _layers.size() - 1;
    
    // Loop for each of the Net's output neurons
    for( int i = 0; i < _layers.back(); i++ )
    {
        // Calculates the delta value for the index i neuron
        FLOAT delta = ( targets[i] - _neurons[lastLayer](i) ) * activationDerivative( _neurons[lastLayer](i) );
        
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


int NeuralNetwork::hiddenGradientMtm( int layer, FLOAT MFactor )
{
    if ( layer < 1 || layer >= static_cast<int>(_layers.size()) - 1 )
    {
        return -1;
    }

    // Loop for each of the neurons in the Net's selected Layer
    for( int k = 0; k < _layers[layer]; k++ )
    {
        FLOAT sum = 0.0;
        
        // Loop for each of the bindings between the selected neuron and the neurons of the next Layer
        for( int l = 0; l < _layers[layer+1]; l++ )
        {
            sum += _weights[layer](l, k) * _deltas[layer+1](l);
        }
        
        FLOAT delta = sum * activationDerivative( _neurons[layer](k) );
        
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
                FLOAT auxGrad = delta * _neurons[layer-1](c);
                _gradients[layer-1](k, c) =  (MFactor * auxGrad) + ( _gradientsMt[layer-1](k, c) * (1 - MFactor) );
                _gradientsMt[layer-1](k, c) = auxGrad;
            }
        }
    }
    
    return 0;
}


int NeuralNetwork::updateWeights( FLOAT rate )
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


bool NeuralNetwork::shiftLeft( VectorN& vector, FLOAT newValue, int newValuePosition )
{
    // Caso a posicao seja maior que a dimensao do vetor, "shifta" todos pra esquerda e insere o novo valor
    if( newValuePosition >= vector.rows() )
    {
        for( int i = 1; i < vector.rows(); i++ )
        {
            vector(i-1) = vector(i);
        }

        vector(vector.rows()-1) = newValue;
        return true;
    }

    // Caso contrario, apenas insere o valor na posicao adequada:
    vector(newValuePosition) = newValue;
    return false;
}


int NeuralNetwork::train( const std::vector<FLOAT>& entry, std::vector<FLOAT>& out, FLOAT l_rate, FLOAT momentum )
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


int NeuralNetwork::use( const std::vector<FLOAT>& entry )
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


int NeuralNetwork::getOut( std::vector<FLOAT>& out )
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

void NeuralNetwork::RMSError( MatrixNxM& inMatrix, MatrixNxM& outMatrix, FLOAT* retRMS_Error )
{
    std::vector<FLOAT> inSample;
    inSample.resize(inMatrix.cols());
    std::vector<FLOAT> outSample;
    outSample.resize(outMatrix.cols());

    FLOAT thisOutput = 0;
    
    // Executando os treinamentos obrigatorios
    for( unsigned int thisSample = 0; thisSample < inMatrix.rows(); thisSample++ )
    {
        for( unsigned int cont = 0; cont < inMatrix.cols(); cont++ )
        {
            inSample[cont] = (FLOAT)(inMatrix( thisSample, cont ));
        }

        for( unsigned int cont = 0; cont < outMatrix.cols(); cont++ )
        {
            outSample[cont] = (FLOAT)(outMatrix( thisSample, cont ));
        }

        use( inSample );
        thisOutput += outputError( outSample );
    }

    *retRMS_Error = (thisOutput/inMatrix.rows());
}


void NeuralNetwork::create( const std::vector<int>& layers )
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
