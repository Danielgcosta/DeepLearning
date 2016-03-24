/*
 * RBM_Stack.h
 * Author: aurelio
 */

#ifndef DEEPLEARNING_RBM_STACK_H_
#define DEEPLEARNING_RBM_STACK_H_

#include <limits.h>

#include "Matrix.h"
#include "RBM.h"



class RBM_Stack
{
public:

	double learningRate;
	unsigned int epochs;
	std::vector<unsigned int> sizesVector;
	std::vector<RBM*> rbmsVector;
	double maxAcceptableError;


	RBM_Stack( std::vector<unsigned int>& szsVector, double lrate=0.2, unsigned int epcs=1500, double maxAccError=0.05 )
	{
		learningRate = lrate;
		epochs = epcs;
		sizesVector = szsVector;
		maxAcceptableError = maxAccError;

		if( szsVector.size() < 2 )
		{
			std::cout << "RBM_Stack constructor. szsVector.size() < 2. Erro." << std::endl;
			return;
		}

		// Creating each RBM:
		for( unsigned int cont=1 ; cont<sizesVector.size() ; cont++ )
		{
			RBM* thisRBM = new RBM( sizesVector[cont-1], sizesVector[cont] );
			rbmsVector.push_back( thisRBM );
		}
	}


	~RBM_Stack()
	{
		for( unsigned int cont=0 ; cont<rbmsVector.size() ; cont++ )
		{
			RBM* thisRBM = rbmsVector[cont];
			delete thisRBM;
		}
	}


	void getLayersSizes( std::vector<unsigned int>& layersSizes )
	{
		layersSizes = sizesVector;
	}


	double getWeightValue( int outLayer, int neuronId, int prevLayerNeuronId )
	{
		RBM* thisRBM = rbmsVector[outLayer-1];
		Matrix weightsCopyRet;
		thisRBM->getWeights( weightsCopyRet );

		unsigned int valR = weightsCopyRet.rows();
		prevLayerNeuronId++;
		prevLayerNeuronId = (prevLayerNeuronId%valR);

		unsigned int valC = weightsCopyRet.columns();
		neuronId++;
		neuronId = (neuronId%valC);

		return weightsCopyRet.get( (unsigned int)prevLayerNeuronId, (unsigned int)neuronId );
	}


	/**
	 * Once the RBM is trained, this function receives a matrix where each row consists of the states of the visible units
	 * to one of the networks from the stack. It returns a matrix where each row consists of the hidden units activated
	 * from the visible units received.
	 */
	void runLayer( unsigned int layerPos, Matrix& dataSet, Matrix& result )
	{
		if( layerPos >= rbmsVector.size() )
		{
			std::cout << "RBM_Stack constructor. szsVector.size() < 2. Erro." << std::endl;
			return;
		}

		RBM* thisRBM = rbmsVector[layerPos];
		if( dataSet.columns() != thisRBM->getVisibleSize() )
		{
			std::cout << "RBM_Stack constructor. dataSet.columns() != thisRBM.getVisibleSize(). Erro." << std::endl;
			return;
		}

		thisRBM->runVisible( dataSet, result );
	}


	void ContrastiveDivergence( double lrate, unsigned int epcs, Matrix& dataSet )
	{
		learningRate = lrate;
		epochs = epcs;
		learn( dataSet );
	}


	/**
	 * Once the RBM is trained, this function receives a matrix where each row consists of the states of the visible units
	 * and returns a matrix where each row consists of the hidden units activated from the visible units received.
	 */
	void runVisible( Matrix& dataSet, Matrix& result )
	{
		Matrix layerDataSet;
		dataSet.copy( layerDataSet );

		Matrix resultTmp;
		// Using each RBM:
		for( unsigned int cont=0 ; cont<rbmsVector.size() ; cont++ )
		{
			RBM* thisRBM = rbmsVector[cont];
			thisRBM->runVisible( layerDataSet, resultTmp );
			resultTmp.copy( layerDataSet );
		}
		resultTmp.copy( result );
	}


	/**
	 * Once the RBM is trained, this function receives a matrix where each row consists of hidden
	 * units activated from visible units previously received.
	 */
	void runHidden( Matrix& dataSet, Matrix& result )
	{
		Matrix layerDataSet;
		dataSet.copy( layerDataSet );

		// Using each RBM on reverse order:
		for( unsigned int cont=rbmsVector.size() ; cont>0 ; cont-- )
		{
			Matrix resultTmp;
			RBM* thisRBM = rbmsVector[cont-1];
			thisRBM->runHidden( layerDataSet, resultTmp );
			resultTmp.copy( layerDataSet );
		}
		layerDataSet.copy( result );
	}


	/**
	 * Learn a matrix of data. Each row represents a single training set.
	 * For example t = [[1,0],[0,1]] will train the rbm to recognize [1,0] and [0,1]
	 * @param dataSet A matrix of data. Each row represents a single training set.
	 */
	void learn( Matrix& dataSet )
	{
		Matrix layerDataSet;
		dataSet.copy( layerDataSet );

		Matrix result;
		// Training each RBM:
		for( unsigned int cont=0 ; cont<rbmsVector.size() ; cont++ )
		{
			RBM* thisRBM = rbmsVector[cont];
			thisRBM->ContrastiveDivergence( learningRate, epochs, maxAcceptableError, layerDataSet );
			thisRBM->runVisible( layerDataSet, result );
			result.copy( layerDataSet );
		}
	}

};

#endif /* DEEPLEARNING_RBM_STACK_H_ */
