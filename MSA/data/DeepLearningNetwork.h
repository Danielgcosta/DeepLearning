/*
 * DeepLearningNetwork.h
 * Author: aurelio
 */

#ifndef DEEPLEARNING_DEEPLEARNINGNETWORK_H_
#define DEEPLEARNING_DEEPLEARNINGNETWORK_H_

#include <limits.h>

#include "BKPNeuralNet.h"
#include "Matrix.h"
#include "RBM_Stack.h"


/**
 * Classe destinada a criacao de redes neurais baseadas em Deep Learning voltadas a feature extraction. O modelo implementado segue o que
 * foi descrito em Hinton, G. E.  and Salakhutdinov R. R., "Reducing the Dimensionality of Data with Neural Networks", SCIENCE, VOL 313, JULY 2006.
 */
class DeepLearningNetwork
{
public:

	std::vector<unsigned int> _layersSizes;
	std::vector<BKPNeuralNet*> _mlpsVector;
	std::vector<BKPNeuralNet*> _mlpsVectorTmp;

	float _mlpMomFactor;
	float _mlpLearningRate;
	unsigned int _mlpEpochs;
	TrainType _mlpTrainType;

	double _rbmLearningRate;
	unsigned int _rbmEpochs;
	double _rbmMaxAcceptableError;



	/**
	 * Auxiliary function. It receives a RBM and BKPNeuralNet.
	 */
	int RBM_Stack_to_BKPNeuralNet( RBM_Stack* inRBM_Stack, BKPNeuralNet*& outBKPNeuralNet )
	{
		std::vector<unsigned int> layersSizes;
		inRBM_Stack->getLayersSizes( layersSizes );
		if( layersSizes.size() > 4 )
			return -1;

		int* layers = new int[layersSizes.size()];
		for( unsigned int i=0 ; i<layersSizes.size() ; i++ )
			layers[i] = (unsigned int)(layersSizes[i]);

		outBKPNeuralNet = new BKPNeuralNet( (int)(layersSizes.size()), layers );
		outBKPNeuralNet->NetInit();

		for( unsigned int i=1 ; i<layersSizes.size() ; i++ )
			for( unsigned int j=0 ; j<(layersSizes[i])+1 ; j++ )
			{
				for( unsigned int k=0 ; k<(layersSizes[i-1])+1 ; k++ )
				{
					double weight = inRBM_Stack->getWeightValue( i, j, k );
					outBKPNeuralNet->setWeights( i, j, k, (float)weight );
				}
			}
		return 0;
	}


	/**
	 * It receives a set of training samples, the number of neurons into the hidden layer and
	 * training parameters. The function creates, trains and returns the network. It begins with
	 * a RBM, then its weights are copied into a MLP where the final training is fine tuned.
	 */
	void trainOneDeepLayer( Matrix& dataSet, unsigned int hiddenLayerSize,
			BKPNeuralNet* &outputMLP )
	{
		// Creating the RBM:
		std::vector<unsigned int> szsVector;
		szsVector.push_back( dataSet.columns() );
		szsVector.push_back( hiddenLayerSize );
		RBM_Stack* thisRBM_Stack = new RBM_Stack( szsVector, _rbmLearningRate, _rbmEpochs );
		// Training the RBM:
		thisRBM_Stack->ContrastiveDivergence( _rbmLearningRate, _rbmEpochs, dataSet );

		// Creating the equivalent MLP, mirroring the RBM outputs:
		int* layersSizes = new int[3];
		layersSizes[0] = (int)(dataSet.columns());
		layersSizes[1] = hiddenLayerSize;
		layersSizes[2] = (int)(dataSet.columns());

		BKPNeuralNet* tempMLP = new BKPNeuralNet( 3, layersSizes );
		tempMLP->NetInit();

		// Inserting the RBMs weight values into the MLP:
		unsigned int i=1;
		for( int j=0 ; j<(layersSizes[i])+1 ; j++ )
		{
			for( int k=0 ; k<(layersSizes[i-1])+1 ; k++ )
			{
				double weight = thisRBM_Stack->getWeightValue( i, j, k );
				tempMLP->setWeights( i, j, k, (float)weight );
				tempMLP->setWeights( i+1, k, j, (float)weight );
			}
		}

		// Vetor auxiliar utilizado para manter as amostas a serem treinadas pela MLP:
		float* inSample   = new float[dataSet.columns()];
		float* outSample = new float[dataSet.columns()];


		// Executing the trains:
		for( unsigned int i=0 ; i<dataSet.rows() * _mlpEpochs ; i++ )
		{
			int thisSample = RandInt(0, (dataSet.rows()-1));
			for( unsigned int cont=0 ; cont<dataSet.columns() ; cont++ )
				inSample[cont] = (float)(dataSet.get( thisSample, cont ));
			tempMLP->Train( (int)(dataSet.columns()), inSample, (int)(dataSet.columns()), inSample, _mlpLearningRate, _mlpMomFactor );

			if( (i % 50000) == 0 )
			{
				double PerDimError=0;
				int PerDimErrorCont=0;
				for( unsigned int j=0 ; j<dataSet.rows() ; j+=1 )
				{
					for( unsigned int cont=0 ; cont<dataSet.columns() ; cont++ )
						inSample[cont] = (float)(dataSet.get( j, cont ));
					tempMLP->Use( dataSet.columns(), inSample );
					tempMLP->GetOut( outSample );

					PerDimError+= tempMLP->EuclideanQuadDist( outSample, inSample, dataSet.columns() );
					PerDimErrorCont++;
				}
				std::cout << "epoch: " << (i / (dataSet.rows() )) << " de: " << _mlpEpochs << ". Error: " << (PerDimError / (PerDimErrorCont)) << std::endl;
			}
		}

		// Creating the output MLP:
		int* outputLayersSizes = new int[2];
		outputLayersSizes[0] = (int)(dataSet.columns());
		outputLayersSizes[1] = hiddenLayerSize;

		outputMLP = new BKPNeuralNet( 2, outputLayersSizes );
		outputMLP->NetInit();

		i=1;
		for( int j=0 ; j<(layersSizes[i])+1 ; j++ )
		{
			for( int k=0 ; k<(layersSizes[i-1])+1 ; k++ )
			{
				float weight;
				tempMLP->getWeights( i, j, k, weight );
				outputMLP->setWeights( i, j, k, weight );
			}
		}

		// Armazenando a rede completa para uso posterior:
		_mlpsVectorTmp.push_back( tempMLP );

		delete[] inSample;
	}


	/**
	 * It trains a network.
	 */
	int train( Matrix& dataSet )
	{
		// In this step we create the first MLP:
		BKPNeuralNet* layerMLP = NULL;
		trainOneDeepLayer( dataSet, _layersSizes[1], layerMLP );
		_mlpsVector.push_back( layerMLP );

		// Auxiliary vector, contains entry samples:
		std::vector<float> inSample;
		inSample.resize( dataSet.columns() );

		// Creating each of the other MLPs:
		for( unsigned int cont=2 ; cont<_layersSizes.size() ; cont++ )
		{
			// It first uses the already developed network to create the layer's dataset:
			Matrix dataSetFeatures( dataSet.rows(),  _layersSizes[cont-1] );

			// For each entry feature vector we get the output using the network's deepest layer:
			for( unsigned int i=0 ; i<dataSet.rows() ; i++ )
			{
				for( unsigned int c=0 ; c<dataSet.columns() ; c++ )
					inSample[c] = (float)(dataSet.get( i, c ));

				std::vector<float> output;
				Use( inSample, output );

				for( unsigned int j=0 ; j<dataSetFeatures.columns() ; j++ )
					dataSetFeatures.set( i, j, output[j] );
			}

			// Once we have the dataset, we are able to train the next layer:
			BKPNeuralNet* layerMLP = NULL;
			trainOneDeepLayer( dataSetFeatures, _layersSizes[cont], layerMLP );

			// Finally we insert the new layer into the vector:
			_mlpsVector.push_back( layerMLP );
		}

		return 0;
	}


	/**
	 * It uses the network.
	 */
	int Use( int inSize, float* inSample, int outSize , float* &output )
	{
		std::vector<float> out;
		std::vector<float> in;
		for( int cont=0 ; cont<inSize ; cont++ )
			in.push_back( inSample[cont] );

		// Using each MLP:
		for( unsigned int cont=0 ; cont<_mlpsVector.size() ; cont++ )
		{
			BKPNeuralNet* thisMLP = _mlpsVector[cont];
			thisMLP->Use( in );
			thisMLP->GetOut( out );

			in.clear();
			for( unsigned int cont=0 ; cont<out.size() ; cont++ )
				in.push_back( out[cont] );
		}

		// Retornando a saida:
		for( int cont=0 ; cont<outSize ; cont++ )
			output[cont] = out[cont];

		return 0;
	}


	/**
	 * It uses the network.
	 */
	int Use( std::vector<float>& inSample, std::vector<float>& output )
	{
		std::vector<float> out;
		std::vector<float> in;
		for( unsigned int cont=0 ; cont<inSample.size() ; cont++ )
			in.push_back( inSample[cont] );

		// Using each MLP:
		for( unsigned int cont=0 ; cont<_mlpsVector.size() ; cont++ )
		{
			BKPNeuralNet* thisMLP = _mlpsVector[cont];
			thisMLP->Use( in );
			thisMLP->GetOut( out );

			in.clear();
			for( unsigned int cont=0 ; cont<out.size() ; cont++ )
				in.push_back( out[cont] );
		}

		// Retornando a saida:
		output.clear();
		for( unsigned int cont=0 ; cont<out.size() ; cont++ )
			output.push_back( out[cont] );

		return 0;
	}


	/**
	 * Construtor.
	 */
	DeepLearningNetwork( std::vector<unsigned int>& layersSizes,
			float mlpMomFactor=0.1f , float mlpLearningRate=0.15f, TrainType mlpTrainType=RANDOM_TRAIN,
			unsigned int mlpEpochs = 3, double rbmLearningRate=0.25f, unsigned int rbmEpochs=50, double rbmMaxAcceptableError=0.02f )
	{
		_layersSizes = layersSizes;
		_mlpMomFactor = mlpMomFactor;
		_mlpLearningRate = mlpLearningRate;
		_mlpTrainType = mlpTrainType;
		_mlpEpochs = mlpEpochs;
		_rbmLearningRate = rbmLearningRate;
		_rbmEpochs = rbmEpochs;
		_rbmMaxAcceptableError = rbmMaxAcceptableError;


		if( layersSizes.size() < 2 )
		{
			std::cout << "DeepLearningNetwork constructor. layersSizes.size() < 2. Erro." << std::endl;
			return;
		}
	}


	~DeepLearningNetwork()
	{
		for( unsigned int cont=0 ; cont<_mlpsVector.size() ; cont++ )
		{
			BKPNeuralNet* thisNeuralNet = _mlpsVector[cont];
			delete thisNeuralNet;
		}
	}


	void getLayersSizes( std::vector<unsigned int>& layersSizes )
	{
		layersSizes = _layersSizes;
	}


private:

};

#endif
