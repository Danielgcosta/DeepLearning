/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   AutoEncoder.h
 * Author: jnavarro
 *
 * Created on 26 de Fevereiro de 2016, 13:00
 */

#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include <vector>

#include "CUDAMatrix.h"
#include "RBM.h"
#include "NeuralNetwork.h"
//#include "cuda_runtime.h"
//#include "cublas_v2.h"

namespace gpu
{

template<typename T> class AutoEncoder
{    

public:

    AutoEncoder( const RBMParameters& rbmParameters, int mlpEpochs, double mlpLearningRate, const std::vector<int>& structure );

    virtual ~AutoEncoder();
                
    bool train( const std::vector< T >& data );
    
    void encode( const std::vector< T >& data, std::vector< T >& features );
    
    void decode( const std::vector< T >& data, std::vector< T >& features );
    
    void run( const std::vector< T >& data, std::vector< T >& features );

private:
    
    int getRBMNumberOfSamples( const std::vector< T >& data );
    
    bool pretrain( const std::vector< T >& data );
    
    bool fineTune( const std::vector< T >& data );
    
    bool pretrainByLayer( const std::vector< T >& data );
    
    bool fineTuneByLayer( const std::vector< T >& data );
    
    void getGPURandomSamples( int numSamples, const std::vector< T >& data, CUDAMatrix<T>& gpuData );
    
    NeuralNetwork* convertRBMToMLP( RBM<T>* rbm );
    
    std::vector<int> createNeuralNetworkStrucutre();
    
    RBMParameters _rbmParameters;
    
    int _mlpEpochs;
    double _mlpLearningRate;
    
    std::vector< int > _structure;
    
    std::vector< RBM<T>* > _rbmStack;
    
    NeuralNetwork _neuralNetwork;

    cublasHandle_t _handle;
    
};

template<typename T>
AutoEncoder<T>::AutoEncoder( const RBMParameters& rbmParameters, int mlpEpochs, double mlpLearningRate, const std::vector<int>& structure) :
    _rbmParameters( rbmParameters ),
    _mlpEpochs( mlpEpochs ),    
    _mlpLearningRate( mlpLearningRate ),
    _structure( structure ),
    _neuralNetwork( createNeuralNetworkStrucutre() )
{
    srand( (unsigned)time( NULL ) );
    cublasStatus_t cublasStatus = cublasCreate( &_handle );
    if ( cublasStatus != CUBLAS_STATUS_SUCCESS )
    {
        std::cout << "AutoEncoder::AutoEncoder - CUBLAS initialization failed\n";
        exit(-1);
    }
}


template<typename T>
AutoEncoder<T>::~AutoEncoder()
{
    for( int i = 0; i < (int) _rbmStack.size(); ++i )
    {
        delete _rbmStack[i];
    }
}

template<typename T>
bool AutoEncoder<T>::train( const std::vector<T>& data )
{
    if( pretrainByLayer( data ) )
    {
        return fineTuneByLayer( data );
    }
    else
    {
        return false;
    }
}


template<typename T>
bool AutoEncoder<T>::pretrain( const std::vector<T>& data )
{
    if( !_rbmStack.empty() )
    {
        for( int i = 0; i < _rbmStack.size(); ++i )
        {
            delete _rbmStack[i];
        }

        _rbmStack.clear();
    }

    for( int i = 0; i < _structure.size() - 1; ++i )
    {
        std::cout << "Training RBM " << i << " of " << _structure.size() - 1 << std::endl;
                
        if( _structure[i] == 0 )
        {
            std::cout << "AutoEncoder::pretrain - Invalid auto-encoder strcture\n";
            return false;
        }

        RBM<T>* rbm = new RBM<T>( _handle, _structure[i], _structure[i + 1], _rbmParameters );
        _rbmStack.push_back( rbm );
        
        int numSamples = getRBMNumberOfSamples( data );
        
        // Criar CUDAMatrix(numSamples, _structure[0])
        CUDAMatrix<T>* gpuData = new CUDAMatrix<T>( _handle, numSamples, _structure[0] );
        
        int dataBlocks = (data.size()/_structure[0]) / numSamples;
        for ( int j = 0; j < dataBlocks; j++ )
        {
            getGPURandomSamples( numSamples, data, *gpuData );

            // If i == 0 just train _rbmStack[0] ...
            if ( i == 0 )
            {
                _rbmStack[0]->train( *gpuData );
            }
            // ... else run all existing rbms up to _rbmStack[i] and use the last output as training input
            else
            {
                CUDAMatrix<T>* in = gpuData;
                CUDAMatrix<T>* out;
                for ( int k = 0; k < i; k++ )
                {
                    out = new CUDAMatrix<T>( _handle, numSamples, _structure[k+1]);
                    _rbmStack[k]->runVisible( *in, *out );
                    
                    // Assuring that the gpuData will not be deleted.
                    if( in != gpuData)
                    {
                        delete in;
                    }
                    
                    in = out;
                }

                _rbmStack[i]->train( *out );
                delete out;
            }
        }
    }
    
    return true;
}


template<typename T>
bool AutoEncoder<T>::pretrainByLayer( const std::vector< T >& data )
{
    if( !_rbmStack.empty() )
    {
        for( int i = 0; i < _rbmStack.size(); ++i )
        {
            delete _rbmStack[i];
        }

        _rbmStack.clear();
    }

    std::vector<T> currentData( data );
    for( int i = 0; i < _structure.size() - 1; ++i )
    {
        std::cout << "Training RBM " << i << " of " << _structure.size() - 1 << std::endl;
                
        if( _structure[i] == 0 )
        {
            std::cout << "AutoEncoder::pretrain - Invalid auto-encoder strcture\n";
            return false;
        }

        // Fase de treinamento da RBM        
        RBM<T>* rbm = new RBM<T>( _handle, _structure[i], _structure[i + 1], _rbmParameters );
        _rbmStack.push_back( rbm );
        
        int numSamples = getRBMNumberOfSamples( currentData );
        
        // Criar CUDAMatrix(numSamples, _structure[0])
        CUDAMatrix<T> gpuData( _handle, numSamples, _structure[i] );
        
        int dataBlocks = (currentData.size()/_structure[i]) / numSamples;
        for ( int j = 0; j < dataBlocks; j++ )
        {
            getGPURandomSamples( numSamples, currentData, gpuData );
            
            _rbmStack[i]->train( gpuData );
        }
        
        NeuralNetwork* mlp = convertRBMToMLP( _rbmStack[i] );

        for( int e = 0; e < _mlpEpochs; e++ )
        {
            for ( int j = 0; j < currentData.size() / _structure[i]; j++ )
            {
                std::vector<FLOAT> in( _structure[i] ), out( _structure[i] );
                int randomRow = rand() % (currentData.size() / _structure[i]);
                std::copy( currentData.begin() + randomRow * _structure[i], currentData.begin() + (randomRow + 1) * _structure[i], in.begin() );
                out = in;
                mlp->train( in, out, _mlpLearningRate, 0.1 );
            }
            
            MatrixNxM inputMatrix( currentData.size() / _structure[i], _structure[i] );
            MatrixNxM outputMatrix( currentData.size() / _structure[i], _structure[i] );
            for( int j = 0; j < inputMatrix.rows(); ++j )
            {
                for( int k = 0; k < inputMatrix.cols(); ++k )
                {
                    inputMatrix( j, k ) = currentData[ j * inputMatrix.cols() + k ];
                }
            }

            float value;
            _neuralNetwork.RMSError( inputMatrix, outputMatrix, &value );
            std::cout << "Fine Tune MSE error: " << value << std::endl;
        }
        
        MatrixNxM currentWeights = mlp->getWeights( i );
        _neuralNetwork.setWeights( i, currentWeights );
        
        // Dividir o dado de entrada, pois pode haver caso em que o conjunto de entrada não cabe na memória da placa.
        int totalSamples = (currentData.size()/_structure[i]);
        CUDAMatrix<T> gpuInputData( _handle, totalSamples, _structure[i] );
        CUDAMatrix<T> gpuOutputData( _handle, totalSamples, _structure[i + 1] );
        
        gpuInputData.setData( currentData, true );
        
        _rbmStack[i]->runVisible( gpuInputData, gpuOutputData );
        
        currentData.clear();
        gpuOutputData.getData( currentData, true );       
    }
    
    return true;
}


template<typename T>
bool AutoEncoder<T>::fineTune( const std::vector< T >& data )
{
    // Preenche a rede neural com os pesos da rbm
    int back = 2 * _rbmStack.size() - 1;
    for ( int i = 0; i < _rbmStack.size(); i++ )
    {
        const CUDAMatrix<T>& gpuWeights = _rbmStack[i]->getWeights();

        std::vector< T > buffer;
        gpuWeights.getData( buffer, true );

        // Converte para MatrixNxM
        MatrixNxM cpuWeights( gpuWeights.cols() - 1, gpuWeights.rows() ); // -1 for the bias
        MatrixNxM transpCpuWeights( gpuWeights.rows() - 1, gpuWeights.cols() ); // -1 for the bias
        for( int j = 0; j < gpuWeights.rows(); ++j )
        {
            for( int k = 0; k < gpuWeights.cols() - 1; ++k ) // -1 to ignore the bias of the RBM
            {
                cpuWeights( k, j ) = buffer[ j * gpuWeights.cols() + k ];
            }
        }
        
        for( int j = 0; j < gpuWeights.rows() - 1; ++j )
        {
            for( int k = 0; k < gpuWeights.cols(); ++k )
            {
                transpCpuWeights( j, k ) = buffer[ j * gpuWeights.cols() + k ];
            }
        }
                                                
        _neuralNetwork.setWeights( i, cpuWeights );
        _neuralNetwork.setWeights( back--, transpCpuWeights );
    }

    MatrixNxM inputMatrix( data.size() / _structure[0], _structure[0] );
    MatrixNxM outputMatrix( data.size() / _structure[0], _structure[0] );
    for( int j = 0; j < inputMatrix.rows(); ++j )
    {
        for( int k = 0; k < inputMatrix.cols(); ++k )
        {
            inputMatrix( j, k ) = data[ j * inputMatrix.cols() + k ];
        }
    }
    
    // Executa o treinamento da rede com o data
    for ( int i = 0; i < _mlpEpochs; i++ )
    {
        if( (i + 1) % (int) ceil( (float) _mlpEpochs / 10.0f ) == 0 )
        {
            std::cout << "Fine Tune part " << i + 1 << " of " << _mlpEpochs << std::endl;
        }
        
        float value;
        _neuralNetwork.RMSError( inputMatrix, outputMatrix, &value );
        std::cout << "Fine Tune MSE error: " << value << std::endl;
        
        for ( int j = 0; j < data.size() / _structure[0]; j++ )
        {
            std::vector<FLOAT> in(_structure[0]), out(_structure[0]);
            int randomRow = rand() % (data.size() / _structure[0]);
            std::copy( data.begin() + randomRow*_structure[0], data.begin() + (randomRow+1)*_structure[0], in.begin() );
            out = in;
            _neuralNetwork.train( in, out, _mlpLearningRate, 0.1 );
        }
    }
    
    return true;
}


template<typename T>
bool AutoEncoder<T>::fineTuneByLayer( const std::vector< T >& data )
{
    MatrixNxM inputMatrix( data.size() / _structure[0], _structure[0] );
    MatrixNxM outputMatrix( data.size() / _structure[0], _structure[0] );
    for( int j = 0; j < inputMatrix.rows(); ++j )
    {
        for( int k = 0; k < inputMatrix.cols(); ++k )
        {
            inputMatrix( j, k ) = data[ j * inputMatrix.cols() + k ];
        }
    }
    
    // Executa o treinamento da rede com o data
    for ( int i = 0; i < _mlpEpochs; i++ )
    {
        if( (i + 1) % (int) ceil( (float) _mlpEpochs / 10.0f ) == 0 )
        {
            std::cout << "Fine Tune part " << i + 1 << " of " << _mlpEpochs << std::endl;
        }
        
        float value;
        _neuralNetwork.RMSError( inputMatrix, outputMatrix, &value );
        std::cout << "Fine Tune MSE error: " << value << std::endl;
        
        for ( int j = 0; j < data.size() / _structure[0]; j++ )
        {
            std::vector<FLOAT> in(_structure[0]), out(_structure[0]);
            int randomRow = rand() % (data.size() / _structure[0]);
            std::copy( data.begin() + randomRow*_structure[0], data.begin() + (randomRow+1)*_structure[0], in.begin() );
            out = in;
            _neuralNetwork.train( in, out, _mlpLearningRate, 0.1 );
        }
    }
    
    return true;
}

template<typename T>
int AutoEncoder<T>::getRBMNumberOfSamples( const std::vector< T >& data )
{
    return 1000;
}


template<typename T>
void AutoEncoder<T>::getGPURandomSamples( int numSamples, const std::vector< T >& data, CUDAMatrix<T>& gpuData )
{
    // Criar um vector<T> de tamanho numSamples * _structure[0] aleatorio
    std::vector<T> randomData( numSamples * _structure[0] );
    for ( int i = 0; i < numSamples; i++ )
    {
        int randomRow = rand() % (data.size() / _structure[0]);

        for ( int k = 0; k < _structure[0]; k++ )
        {
            randomData[i*_structure[0] + k] = data[randomRow*_structure[0] + k];
        }
        
    }
    
    gpuData.setData( randomData, true );
}


template<typename T>
std::vector<int> AutoEncoder<T>::createNeuralNetworkStrucutre()
{
    std::vector<int> layers( _structure );
    layers.reserve( 2*_structure.size() - 1 );
    for ( std::vector<int>::reverse_iterator i = _structure.rbegin() + 1; i != _structure.rend(); i++ )
    {
        layers.push_back( *i );
    }
    
    return layers;
}

template<typename T>
void AutoEncoder<T>::run( const std::vector< T >& data, std::vector< T >& features )
{
    for( int i = 0; i < (int) (data.size() / _structure[0]); i++ )
    {
        std::vector<FLOAT> inputFeature;
        for( int j = 0; j < _structure[0]; j++ )
        {
            inputFeature.push_back( data[i * _structure[0] + j] );
        }
        
        _neuralNetwork.use( inputFeature );
        
        std::vector<FLOAT> outputFeature;
	_neuralNetwork.getOut( outputFeature );

        for( size_t j = 0; j < outputFeature.size(); j++ )
        {
            features.push_back( outputFeature[j] );
        }
    }
}


template<typename T>
NeuralNetwork* AutoEncoder<T>::convertRBMToMLP( RBM<T>* rbm )
{
    const CUDAMatrix<T>& gpuWeights = rbm->getWeights();

    std::vector< T > buffer;
    gpuWeights.getData( buffer, true );

    // Converte para MatrixNxM
    MatrixNxM cpuWeights( gpuWeights.cols() - 1, gpuWeights.rows() ); // -1 for the bias
    MatrixNxM transpCpuWeights( gpuWeights.rows() - 1, gpuWeights.cols() ); // -1 for the bias
    for( int j = 0; j < gpuWeights.rows(); ++j )
    {
        for( int k = 0; k < gpuWeights.cols() - 1; ++k ) // -1 to ignore the bias of the RBM
        {
            cpuWeights( k, j ) = buffer[ j * gpuWeights.cols() + k ];
        }
    }

    for( int j = 0; j < gpuWeights.rows() - 1; ++j )
    {
        for( int k = 0; k < gpuWeights.cols(); ++k )
        {
            transpCpuWeights( j, k ) = buffer[ j * gpuWeights.cols() + k ];
        }
    }

    std::vector< int > netStructure;
    netStructure.push_back( rbm->getVisibleSize() );
    netStructure.push_back( rbm->getHiddenSize() );
    netStructure.push_back( rbm->getVisibleSize() );
    NeuralNetwork* neuralNetwork = new NeuralNetwork( netStructure );
    
    neuralNetwork->setWeights( 0, cpuWeights );
    neuralNetwork->setWeights( 1, transpCpuWeights );
    
    return neuralNetwork;
}

}

#endif /* AUTOENCODER_H */

