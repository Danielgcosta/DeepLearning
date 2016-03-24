/*
 * RBM.h
 *
 *  Created on: Dec 26, 2014
 *      Author: aurelio
 */

#ifndef DEEPLEARNING_RBM_H_
#define DEEPLEARNING_RBM_H_

#include <limits.h>

#include "Matrix.h"

class RBM
{
public:

	double learningRate;
	unsigned int epochs;
	double maxAcceptableError;

	Matrix weights;
	Matrix weightsBest;



	RBM( unsigned int visibleSize, unsigned int hiddenSize, double lrate=0.1, unsigned int epcs=15000 )
	{
		learningRate = lrate;
		epochs = epcs;
		weights = Matrix( visibleSize, hiddenSize );

		for( unsigned int i = 0; i < weights.rows(); i++)
		{
			for( unsigned int j = 0; j < weights.columns(); j++)
			{
				weights.set( i, j, RandFloat() / 5.0 );
			}
		}
		// Insert weights for the bias units into the first row and first column:
		weights.insert( 0, 1.0 );
		weights.insert( 1, 1.0 );
		weights.copy( weightsBest );
	}


	unsigned int getVisibleSize()
	{
		return weights.rows();
	}


	unsigned int getHiddenSize()
	{
		return weights.columns();
	}


	void getWeights( Matrix& weightsCopyRet )
	{
		weights.copy( weightsCopyRet );
	}


	void setWeights( Matrix& weightsNew )
	{
		weights = weightsNew;
	}


	void setWeights( unsigned int i, unsigned int j, double val )
	{
		weights.set( i, j, val );
	}


	double sigmoid( double x )
	{
		return 1.0 / (1.0 + exp(-x));
	}


	void sigmoid( Matrix& m )
	{
		for( unsigned int i = 0; i < m.rows(); i++)
		{
			for( unsigned int j = 0; j < m.columns(); j++)
			{
				m.set( i, j, sigmoid( m.get(i, j) ) );
			}
		}
	}


	double activationState( double x, double y )
	{
		return (x >= y) ? 1.0 : 0.0;
	}


	Matrix activationState( Matrix& m, Matrix& c )
	{
		Matrix ret( m.rows(), m.columns() );

		unsigned int j=0;
		#pragma omp parallel for shared( ret ) private(j)
		for( unsigned int i = 0; i < m.rows(); i++)
		{
			for( j = 0; j < m.columns(); j++)
			{
				ret.set( i, j, activationState( m.get(i, j), c.get(i, j) ) );
			}
		}
		return ret;
	}


	void ContrastiveDivergence( double lrate, unsigned int epcs, double maxAccError, Matrix& dataSet )
	{
		learningRate = lrate;
		epochs = epcs;
		maxAcceptableError = maxAccError;

		// Insert bias units of 1 into the first column:
		Matrix datasetBias;
		dataSet.copy( datasetBias );
		datasetBias.insert( 1, 1.0 );

		learn( datasetBias );
	}


	/**
	 * Learn a matrix of data. Each row represents a single training set.
	 * For example t = [[1,0],[0,1]] will train the rbm to recognize [1,0] and [0,1]
	 * @param dataSet A matrix of data. Each row represents a single training set.
	 */
	void learn( Matrix& dataSet )
	{
		std::cout << "Trainning the RBM network:" << std::endl;

		int numberSamples = dataSet.rows();
		double bestError = std::numeric_limits<double>::max();

		// Now we reconstruct the visible neurons from the hidden outputs. It is known as negative CD phase (daydreaming phase):
		for( unsigned int epoch=0 ; epoch<epochs ; epoch++ )
		{
			// Read training data and sample from the hidden later, positive CD phase, (reality phase) -> Matrix positiveHiddenActivations = dataSet.dot( weights );
			// Producing a matrix: coordinate (n,m) contains dot product( sample n, hidden neuron m ):
			Matrix positiveHiddenActivations;
			dataSet.matrixMultiplication( weights, positiveHiddenActivations );

			// Applying the activation function to all the entry samples:
			Matrix positiveHiddenProbabilities;
			positiveHiddenActivations.copy( positiveHiddenProbabilities );
			sigmoid( positiveHiddenProbabilities );

			// positiveHiddenStates(i,j) is 1.0 if positiveHiddenProbabilities(i,j) > randMatrix(i.j), or otherwise 0.0:
			Matrix randMatrix( numberSamples, getHiddenSize() );
			randMatrix.random();
			Matrix positiveHiddenStates;
			positiveHiddenStates = activationState( positiveHiddenProbabilities, randMatrix );

			// Testando se podemos utilizar as proprias saidas como estados:
			positiveHiddenProbabilities.copy( positiveHiddenStates );

			// When computing associations we are using the activation probabilities of the hidden states instead of the states itself. In this case
			// we could also use the states itself, as described in the section 3 of Hinton's A Practical Guide to Training Restricted Boltzmann Machines"
			Matrix dataSetTransposed;
			dataSet.transpose( dataSetTransposed );
			Matrix positiveAssociations;
			dataSetTransposed.matrixMultiplication( positiveHiddenProbabilities, positiveAssociations );

			// Now we reconstruct the visible neurons from the hidden outputs. It is known as negative CD phase (daydreaming phase):
			Matrix weightsTransposed;
			weights.transpose( weightsTransposed );

			Matrix negativeVisibleActivations;
			positiveHiddenStates.matrixMultiplication( weightsTransposed, negativeVisibleActivations );
			Matrix negativeVisibleProbabilities;
			negativeVisibleActivations.copy( negativeVisibleProbabilities );
			sigmoid( negativeVisibleProbabilities );

			// Fixing the bias unit:
			for( unsigned int cont=0 ; cont< negativeVisibleProbabilities.rows() ; cont++ )
				negativeVisibleProbabilities.set( cont, 0, 1.0 );

			Matrix negativeHiddenActivations;
			negativeVisibleProbabilities.matrixMultiplication( weights, negativeHiddenActivations );
			Matrix negativeHiddenProbabilities;
			negativeHiddenActivations.copy( negativeHiddenProbabilities );
			sigmoid( negativeHiddenProbabilities );

			// Note, again, that we're using the activation *probabilities* when computing associations, not the states themselves.
			Matrix negativeVisibleProbabilitiesTransposed;
			negativeVisibleProbabilities.transpose( negativeVisibleProbabilitiesTransposed );
			Matrix negativeAssociations;
			negativeVisibleProbabilitiesTransposed.matrixMultiplication( negativeHiddenProbabilities, negativeAssociations );


			Matrix updates;
			positiveAssociations.apply( negativeAssociations, Matrix::SUBTRACT, updates );
			updates.apply( (double)numberSamples, Matrix::DIVIDE );
			updates.apply( learningRate, Matrix::MULTIPLY );
			// Update weights:
			Matrix weightsUpdated;
			weights.apply( updates, Matrix::ADD, weightsUpdated );
			weightsUpdated.copy( weights );

			Matrix dataSetCopy;
			dataSet.copy( dataSetCopy );
			Matrix dataSetCopyResult;

			dataSetCopy.apply( negativeVisibleProbabilities, Matrix::SUBTRACT, dataSetCopyResult );
			dataSetCopyResult.apply( 2.0, Matrix::POWER );
			double thisError = dataSetCopyResult.sum();

			// Erro medio por neuronio:
			thisError = thisError / dataSet.rows() ;
			// Armazenando a melhor rede:
			if( thisError < bestError )
			{
				bestError = thisError;
				weights.copy( weightsBest );
			}

			std::cout << "epoch: " << epoch << " from: " << epochs << ". error:" << thisError << std::endl;

			if( bestError < maxAcceptableError )
				break;
		}
		std::cout << std::endl;
		std::cout << std::endl;

		weightsBest.copy( weights );
	}


	/**
	 * Once the RBM is trained, this function receives a matrix where each row consists of the states of the visible units
	 * and returns a matrix where each row consists of the hidden units activated from the visible units received.
	 */
	void runVisible( Matrix& dataSet, Matrix& result )
	{
		// Insert bias units of 1 into the first column:
		Matrix datasetBias;
		dataSet.copy( datasetBias );
		datasetBias.insert( 1, 1.0 );

		// Calculate the activations of the hidden units:
		Matrix positiveHiddenActivations;
		datasetBias.matrixMultiplication( weights, positiveHiddenActivations );

		// Applying the activation function to all the entry samples:
		Matrix positiveHiddenProbabilities;
		positiveHiddenActivations.copy( positiveHiddenProbabilities );
		sigmoid( positiveHiddenProbabilities );

		result.resize( positiveHiddenProbabilities.rows(), positiveHiddenProbabilities.columns()-1 );
    	for( unsigned int r=0 ; r<positiveHiddenProbabilities.rows() ; r++ )
        	for( unsigned int c=1 ; c<positiveHiddenProbabilities.columns() ; c++ )
        		result.set( r, (c-1), positiveHiddenProbabilities.get( r, c ) );


    	//		positiveHiddenProbabilities.copy( result );
		// Turn the hidden units on with their specified probabilities:
		// positiveHiddenStates(i,j) is 1.0 if positiveHiddenProbabilities(i,j) > randMatrix(i.j), or otherwise 0.0:
//		int numberSamples = datasetBias.rows();
//		Matrix randMatrix( numberSamples, getHiddenSize() );
//		randMatrix.random();
//		Matrix positiveHiddenStates = activationState( positiveHiddenProbabilities, randMatrix );
//		positiveHiddenStates.copy( result );
	}


	/**
	 * Once the RBM is trained, this function receives a matrix where each row consists of hidden
	 * units activated from visible units previously received.
	 */
	void runHidden( Matrix& dataSet, Matrix& result )
	{
		// Insert bias units of 1 into the first column:
		Matrix datasetBias;
		dataSet.copy( datasetBias );
		datasetBias.insert( 1, 1.0 );

		// Calculate the activations of the hidden units:
		Matrix weightsTransposed;
		weights.transpose( weightsTransposed );

		Matrix visibleActivations;
		datasetBias.matrixMultiplication( weightsTransposed, visibleActivations );
		Matrix visibleProbabilities;
		visibleActivations.copy( visibleProbabilities );
		sigmoid( visibleProbabilities );

		result.resize( visibleProbabilities.rows(), visibleProbabilities.columns()-1 );
    	for( unsigned int r=0 ; r<visibleProbabilities.rows() ; r++ )
        	for( unsigned int c=1 ; c<visibleProbabilities.columns() ; c++ )
        		result.set( r, (c-1), visibleProbabilities.get( r, c ) );
	}

};

#endif /* DEEPLEARNING_RBM_H_ */
