/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   AutoEncoderGPU.h
 * Author: jnavarro
 *
 * Created on 26 de Fevereiro de 2016, 13:00
 */

#ifndef AUTOENCODERGPU_H
#define AUTOENCODERGPU_H

#include <vector>

#include "CUDAMatrix.h"
#include "RBM.h"
#include "MLPNetwork.h"
//#include "cuda_runtime.h"
//#include "cublas_v2.h"

namespace gpu
{

template<typename T> class AutoEncoderGPU
{    

public:

    AutoEncoderGPU( const RBMParameters& rbmParameters, int mlpEpochs, double mlpLearningRate, const std::vector<int>& structure );

    virtual ~AutoEncoderGPU();
                
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
    
    dnn::MLPNetwork<T>* convertRBMToMLP( RBM<T>* rbm );
    
    std::vector<int> createNeuralNetworkStrucutre();
    
    RBMParameters _rbmParameters;
    
    int _mlpEpochs;
    double _mlpLearningRate;
    
    std::vector< int > _structure;
    
    std::vector< RBM<T>* > _rbmStack;
    
    dnn::MLPNetwork<T> _neuralNetwork;

    cublasHandle_t _handle;
    
};

template<typename T>
AutoEncoderGPU<T>::AutoEncoderGPU( const RBMParameters& rbmParameters, int mlpEpochs, double mlpLearningRate, const std::vector<int>& structure) :
    _rbmParameters( rbmParameters ),
    _mlpEpochs( mlpEpochs ),    
    _mlpLearningRate( mlpLearningRate ),
    _structure( structure ),
    _neuralNetwork( dnn::DNNParameters( mlpLearningRate, 1e-6, mlpEpochs ), createNeuralNetworkStrucutre() )
{
    srand( (unsigned)time( NULL ) );
    cublasStatus_t cublasStatus = cublasCreate( &_handle );
    if ( cublasStatus != CUBLAS_STATUS_SUCCESS )
    {
        std::cout << "AutoEncoderGPU::AutoEncoderGPU - CUBLAS initialization failed\n";
        exit(-1);
    }
}


template<typename T>
AutoEncoderGPU<T>::~AutoEncoderGPU()
{
    for( int i = 0; i < (int) _rbmStack.size(); ++i )
    {
        delete _rbmStack[i];
    }
}

template<typename T>
bool AutoEncoderGPU<T>::train( const std::vector<T>& data )
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
bool AutoEncoderGPU<T>::pretrain( const std::vector<T>& data )
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
            std::cout << "AutoEncoderGPU::pretrain - Invalid auto-encoder structure\n";
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
bool AutoEncoderGPU<T>::pretrainByLayer( const std::vector< T >& data )
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
            std::cout << "AutoEncoderGPU::pretrain - Invalid auto-encoder structure\n";
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
        
        dnn::MLPNetwork<T>* mlp = convertRBMToMLP( _rbmStack[i] );
        
        mlp->train( currentData.data(), currentData.data(), numSamples, _structure[i], _structure[i] );

        const gpu::CUDAMatrix<T>* m = mlp->getWeights( i );
        if ( m != nullptr )
        {
            _neuralNetwork.setWeights( i, *m );
        }
        delete mlp;
        
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
bool AutoEncoderGPU<T>::fineTune( const std::vector< T >& data )
{
    // Preenche a rede neural com os pesos da rbm
    int back = 2 * _rbmStack.size() - 1;
    for ( int i = 0; i < _rbmStack.size(); i++ )
    {
        const CUDAMatrix<T>& gpuWeights = _rbmStack[i]->getWeights();
        CUDAMatrix<T> transpGpuWeights( _handle, gpuWeights.cols(), gpuWeights.rows() );

        std::vector< T > buffer;
        gpuWeights.getData( buffer, true );
        transpGpuWeights.setData( buffer, false );
                                                
        _neuralNetwork.setWeights( i, gpuWeights );
        _neuralNetwork.setWeights( back--, transpGpuWeights );
    }
    
    _neuralNetwork.train( data.data(), data.data(), data.size() / _structure[0], _structure[0], _structure[0] );

    return true;
}


template<typename T>
bool AutoEncoderGPU<T>::fineTuneByLayer( const std::vector< T >& data )
{
    _neuralNetwork.train( data.data(), data.data(), data.size() / _structure[0], _structure[0], _structure[0] );
    
    return true;
}

template<typename T>
int AutoEncoderGPU<T>::getRBMNumberOfSamples( const std::vector< T >& data )
{
//    return std::min<int>( 50000, data.size()/_structure[0] );
    return 0.05*(data.size()/_structure[0]);
}


template<typename T>
void AutoEncoderGPU<T>::getGPURandomSamples( int numSamples, const std::vector< T >& data, CUDAMatrix<T>& gpuData )
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
std::vector<int> AutoEncoderGPU<T>::createNeuralNetworkStrucutre()
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
void AutoEncoderGPU<T>::run( const std::vector< T >& data, std::vector< T >& features )
{
    features.clear();
    features.resize( (data.size()/_structure[0]) * _structure[0] );
    _neuralNetwork.run( data.data(), features.data(), data.size()/_structure[0], _structure[0], _structure[0] );
}


template<typename T>
dnn::MLPNetwork<T>* AutoEncoderGPU<T>::convertRBMToMLP( RBM<T>* rbm )
{
    const CUDAMatrix<T>& gpuWeights = rbm->getWeights();
    CUDAMatrix<T> transpGpuWeights( _handle, gpuWeights.cols(), gpuWeights.rows() );

    std::vector< T > buffer;
    gpuWeights.getData( buffer, true );
    transpGpuWeights.setData( buffer, false );

    std::vector< int > netStructure;
    netStructure.push_back( rbm->getVisibleSize() );
    netStructure.push_back( rbm->getHiddenSize() );
    netStructure.push_back( rbm->getVisibleSize() );
    dnn::MLPNetwork<T>* neuralNetwork = new dnn::MLPNetwork<T>( dnn::DNNParameters( _mlpLearningRate, 1e-6, _mlpEpochs ), netStructure );
    
    neuralNetwork->setWeights( 0, gpuWeights );
    neuralNetwork->setWeights( 1, transpGpuWeights );
    
    return neuralNetwork;
}

}

#endif /* AUTOENCODERGPU_H */

