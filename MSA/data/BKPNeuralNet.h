#ifndef BKP_NEURAL_NET
#define BKP_NEURAL_NET

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "Matrix.h"


typedef enum
{
	RANDOM_TRAIN,
	ORDERED_TRAIN
} TrainType;


class BKPNeuralNet{

	//protected:

	int _layers;            ///< Number of Net Layers.
	int *_nLayers;          ///< Number of neurons in each Layer.
	float **_neurons;       ///< Output value of each neuron.
	float ***_weights;      ///< Connection Weights of each neuron from Layer n to Layer n+1.
	float ***_gradients;    ///< Gradients of each Weight.
	float ***_gradientsMt;  ///< Temporary Gradients of each Weight (Momentum).
	float **_deltas;        ///< Auxiliary vector to save temporary deltas while training the net.
	int _flagMomentum;      ///< Auxiliary flag to indicate this is not the first train.



	/**
	 * Function used to calculate the activation function value for any input, based on the table.
	 * @param x Input to calculate the corresponding output value of the function.
	 * @return Returns the function output value for the corresponding x input.
	 */
	inline float Act_func (float x) {return 1.0f / (1.0f + (float)exp(-x));}

	/**
	 * To train the neural Net, we'll need the activation function derivative's value of any
	 * point. This function returns this derivative.
	 * @param f Input value. The activation function derivative will be calculated for this value.
	 * @return Returns the value of the activation function derivative.
	 */
	inline float Actderiv (float f) {return f * (1 - f);}

	/**
	 * Example: for a 3-Layer Net, if I want to train the third Layer
	 * I just need to call ActivCalculus( Net, 2 ); -> the indice begins with zero!!!
	 * The first Layer (Layer zero) doesn't have Weights before it, right???
	 */
	int ActivCalculus( int Layer );

	/** 
	 * Function used to find the activation outputs for all the neurons of the Net, calling ActivCalculus for each of the Layers.
	 * @return Returns -2 if the value pointed by Net is NULL, or 0 if ok.
	 */
	void ActivAll();

	/**
	 * Multiplies 2 vectors, element by element, and adds all results.
	 * Ideally, the first vector would be the input vector, and the second would be the Weight vector with the Weights to be
	 * applied to each one of the neuron's inputs.
	 */
	float DotProduct( int n, float *vec1, float *vec2 );

	/**
	 * Function that inserts entry values on the Net.
	 * @param size Length of the entry vector.
	 * @param entry Vector that contains the entries.
	 * @return It returns 0 if ok.
	 */
	int SetEntry( int size, float *entry );

	/**
	 * Function that inserts entry values on the input Layer of the Net.
	 */
	int SetEntry( std::vector<float>& entry );

	/**
	 * Function used to calculate the gradients of the last Layer of neurons.
	 * @param PtNet Net Pointer to the Net in use.
	 * @param targets Target vector (desired outputs).
	 * @param dimTargets Dimension of the target vector.
	 * @return Returns -2 if Net points to NULL, -1 if dimTargets is less than zero, or 0 if ok.
	 */
	void OutGradient( float *targets, int dimTargets );

	/**
	 * Function used to train the hidden Layers's neurons.
	 * @param Net Net being used.
	 * @param Layer Layer to receive the Gradients based on the already calculated Weights.
	 * @return Returns -2 if Net points to NULL, -1 if dimTargets is less than zero, or 0 if ok.
	 */
	void HiddenGradient( int Layer );

	/**
	 * Function used to calculate the gradients of the last Layer of neurons.
	 * @param targets Target vector (desired outputs).
	 * @param dimTargets Dimension of the target vector.
	 * @param MFactor Momentum factor.
	 */
	void OutGradientMtm( float *targets, int dimTargets, float MFactor );

	/**
	 * Function used to train the hidden Layers's neurons.
	 * @param Layer Layer to receive the Gradients based on the already calculated Weights.
	 * @param MFactor Momentum factor.
	 * @return Returns -2 if Net points to NULL, -1 if dimTargets is less than zero, or 0 if ok.
	 */
	int HiddenGradientMtm( int Layer, float MFactor );

	/**
	 * Function that update the Weights on the Net.
	 * @param Net Net being used.
	 * @param rate The Learning rate.
	 * @return Returns -2 if Net points to NULL, -1 if the value of rate is not between 0.0 and 1.0, or 0.
	 */
	int UpdateWeights( float rate );



public:

	/**
	 * Retorna a distancia Euclideana.
	 */
	float EuclideanQuadDist( float *vec1, float* vec2, int size )
	{
		int i;
		float temp = 0;
		for( i=0 ; i<size ; i++ )
			temp+= (float)( (vec1[i]-vec2[i])*(vec1[i]-vec2[i]) );
		return temp;
	}


	/**
	 * Variancia de um vetor qualquer.
	 */
	float Variance( float *vec, int size, float *m_ret )
	{
		// Media
		float media=0;
		for( int i=0 ; i<size ; i++ )
			media+= vec[i];
		media = media/size;
		*m_ret = media;

		// Obtendo a variancia:
		float var=0;
		for( int i=0 ; i<size ; i++ )
			var += ( (vec[i]-media)*(vec[i]-media) );
		return var/size;
	}


	/**
	 * Recebe um vetor de tamanho n e desloca os valores 1 posicao a esquerda.
	 * O primeiro valor (o valor contido na posicao de indice 0 do vetor) sera deletado.
	 */
	bool ShiftLeft( float *Vector, int size, float newValue, int newValuePosition );

	/**
	 * Constructor. It receives a filename of a previous trained BKP Neural Network.
	 * @param filename File to be loaded.
	 */
	BKPNeuralNet( char* filename );

	/**
	 * Constructor.
	 * @param Layers Number of layers desired.
	 * @param nLayers Vector containing the number of neurons on each layer.
	 */
	BKPNeuralNet( int Layers, int *nLayers);

	/**
	 * Function used to create a new neural Net.
	 */
	void CreateNet( int Layers, int *nLayers);

	/**
	 * Destrutor.
	 */
	~BKPNeuralNet();

	/**
	 * Inserts random numbers between 0.0 and 0.1 as Weight values.
	 * @return Returns -2 if Net is pointing to NULL, 0.
	 */
	int NetInit();


	/**
	 * Insert specific values into the weight vectors.
	 * Example: for a 3-Layer Net, if I want to insert values into the neurons on third Layer.
	 * I just need to set the weights from position 2; -> the index begins IN zero!!!
	 * The first Layer (Layer zero) doesn't have Weights before it, right???
	 */
	int setWeights( int outLayer, int neuronId, int prevLayerNeuronId, float value );


	/**
	 * Get specific values from the weight vectors.
	 * Example: for a 3-Layer Net, if I want to insert values into the neurons on third Layer.
	 * I just need to set the weights from position 2; -> the index begins IN zero!!!
	 * The first Layer (Layer zero) doesn't have Weights before it, right???
	 */
	int getWeights( int outLayer, int neuronId, int prevLayerNeuronId, float& value );


	/**
	 * Function that calculates the Net's output error
	 * @param targets Neuron's targeted output.
	 * @param dimTargets Dimension of target vector.
	 * @param mode Medium error, Quadratic medium error.
	 * @return Returns the resulting error.
	 */
	float OutputError( float *targets, int dimTargets );


	/**
	 * Function used to train the net.
	 * @param size The size of entry vector.
	 * @param entry Vector that contains values of entries.
	 * @param size2 The size of out vector.
	 * @param out Vector that contains values of outs.
	 * @param l_rate The learning rate.
	 * @param momentum Momentum factor.
	 * @return returns 1 if ok.
	 */
	int Train( int size, float *entry, int size2, float *out, float l_rate, float momentum = 0 );


	/**
	 * Once the net is ready, we need a function to use it.
	 * @param size Length of the entry vector.
	 * @param entry The entries vector.
	 * @return It returns 0 if ok.
	 */
	int Use(int size, float *entry);


	/**
	 *  Funcao para utilizar a rede depois de treinada.
	 */
	int Use( std::vector<float>& entry );


	/**
	 * Function that gets out values from the Net.
	 * @param out Out vetcor.
	 * @return It returns 0 if ok.
	 */
	int GetOut( float *out );

	/**
	 * Function that gets out values from the Net.
	 */
	int GetOut( std::vector<float>& out );

	/**
	 * Saves a neural net.
	 * @param filename Name of the new file.
	 * @return It returns 0 if ok.
	 */
	int SaveNet( char* filename );


	/**
	 * Obtem o erro medio quadratico de um conjunto de amostras fornecido.
	 * @param inMatrix Matriz M[nSamples][inSize] contendo todas as amostras de entrada.
	 * @param outMatrix Matriz M[nSamples][inSize] contendo todas as amostras de saida respectivas.
	 * @param inSize Dimensao das amostras de entrada (mesma dimensao da camada de entrada da rede).
	 * @param outSize DImensao das amostras de saida (mesma dimensao da camada de saida da rede).
	 * @param nSamples Numero de amostras fornecido.
	 * @param retRMS_Error Valor do erro medio quadratico retornado.
	 * @return It returns 0 if ok.
	 */
	int RMS_error( float**inMatrix, float **outMatrix, int inSize, int outSize, int nSamples, float* retRMS_Error );


	/**
	 * Obtem o erro medio quadratico de um conjunto de amostras fornecido.
	 */
	void RMS_error( Matrix& inMatrix, Matrix& outMatrix, float* retRMS_Error );


	/**
	 *   Funcao construida com o objetivo de facilitar o treinamento de uma rede. Utiliza criterios
	 * de parada  pre-definicos. O objetivo e paralizar o treinamento a partir do momento em que o
	 * erro medio quadratico da rede em relacao as amostras para de  diminuir. Recebe um  parametro
	 * indicando um numero minimo de treinos, a a partir do qual se inicia a verificacao da variacao
	 * do erro medio quadratico. Recebe tambem o numero de treinamentos a ser executado ate que uma
	 * nova medicao do erro seja feita. Caso a variancia (porcentual) das ultimas n medicoes seja
	 * menor ou igual a um determinado valor (entre 0 e 1), paraliza o treinamento.
	 *   A funcao recebe ainda um conjunto de amostras (matriz de entradas/matriz de saidas), numero
	 * de amostras contidas nas matrizes, a dimensao de cada amostra de entrada e de cada amostra de
	 * saida e um flag indicando se as amostras devem ser treinadas aleatoriamente ou em ordem.
	 * @param inMatrix Matriz M[nSamples][inSize] contendo todas as amostras de entrada.
	 * @param outMatrix Matriz M[nSamples][inSize] contendo todas as amostras de saida respectivas.
	 * @param inSize Dimensao das amostras de entrada (mesma dimensao da camada de entrada da rede).
	 * @param outSize Dimensao das amostras de saida (mesma dimensao da camada de saida da rede).
	 * @param nSamples Numero de amostras fornecido.
	 * @param minTrains Numero minimo de treinamentos a ser executado.
	 * @param varVectorSize Numero de medicoes do erro medio quadratico utilizadas na obtencao da variancia.
	 * @param minStdDev Desvio-padrao minimo (porcentuam em relacao a media 0~1) das medicoes de erro
	 * utilizadas como o critedo de parada.
	 * @param numTrains Uma vez que o numero minimo (minTrains) sa foi executado, numTrains é o numero de
	 * treinos executado ate a proxima medicao do erro medio quadratico .
	 * @param type Forma como as amostras serao  apresentadas para a rede (RANDOM_TRAIN/ORDERED_TRAIN).
	 * @param l_rate The learning rate.
	 * @param momentum Momentum factor.
	 * @param retExecutedTrains Numero de treinamentos executados ate o criterio de parada ser atingido.
	 * @return It returns 0 if ok.
	 */
	int AutoTrain( float**inMatrix, float **outMatrix, int inSize, int outSize, int nSamples,
			int minTrains, int varVectorSize, float minStdDev, int numTrains, TrainType type,
			float l_rate, float momentum, int* retExecutedTrains );



};

#endif
// BKP_NEURAL_NET

