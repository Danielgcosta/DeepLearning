/* 
 * File:   RBMStack.h
 * Author: jnavarro
 *
 * Created on January 7, 2016, 7:08 PM
 */

#ifndef RBMSTACK_H
#define	RBMSTACK_H

#include "RBM.h"
#include <iostream>

namespace gpu
{
    
template<typename T> class RBMStack
{
    public:
        
        RBMStack( const cublasHandle_t& handle, std::vector< int >& dimensions, const RBMParameters& parameters );
        
        ~RBMStack( );
        
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
        
    private:

    std::vector< RBM<T>* > _rbms;
    
    cublasHandle_t mCUBLASHandle;
};


template<typename T>
RBMStack<T>::RBMStack( const cublasHandle_t& handle, std::vector< int >& dimensions, const RBMParameters& parameters ) : mCUBLASHandle( handle )    
{
    _rbms.reserve( dimensions.size() - 1 );
        
    for( unsigned int i = 0; i < dimensions.size() - 1; ++i )
    {
        RBM<T>* rbm = new RBM<T>( handle, dimensions[ i ], dimensions[ i + 1 ], parameters );
       _rbms.push_back( rbm );
    }
}


template<typename T>
RBMStack<T>::~RBMStack()
{
    for( unsigned int i = 0; i < _rbms.size(); ++i )
    {
        delete _rbms[i];
    }
        
}


template<typename T>
void RBMStack<T>::train( const CUDAMatrix<T>& data )
{
    CUDAMatrix<T> currentData = data;
    for( int i = 0; i < _rbms.size(); ++i )
    {
        _rbms[i]->train( currentData );
        
        //std::cout << "RBM[" << i << "] - MSE Error: " << _rbms[i]->computeMSEError( currentData ) << std::endl;
        
        CUDAMatrix<T> hidden( mCUBLASHandle, data.rows(), _rbms[i]->getHiddenSize() );
        
        _rbms[i]->runVisible( currentData, hidden );
        
        currentData = hidden;
    }
    
    //std::cout << std::endl;
}


template<typename T>
void RBMStack<T>::runVisible( const CUDAMatrix<T>& data, CUDAMatrix<T>& hidden )
{
    CUDAMatrix<T> currentData = data;
    for( int i = 0; i < _rbms.size(); ++i )
    {
        CUDAMatrix<T> hidden( mCUBLASHandle, data.rows(), _rbms[i]->getHiddenSize() );
        
        _rbms[i]->runVisible( currentData, hidden );
        
        currentData = hidden;
    }
    
    hidden = currentData;
}


template<typename T>
void RBMStack<T>::runHidden( const CUDAMatrix<T>& data, CUDAMatrix<T>& visible )
{
    CUDAMatrix<T> currentData = data;
    for( int i = _rbms.size() - 1; i >= 0; ++i )
    {
        CUDAMatrix<T> visible( mCUBLASHandle, data.rows(), _rbms[i]->getVisibleSize() );
        
        _rbms[i]->runHidden( currentData, visible );
        
        currentData = visible;
    }
    
    visible = currentData;
}

}



#endif	/* RBMSTACK_H */

