/* 
 * File:   NeuralNetwork.h
 * Author: ederperez
 *
 * Created on November 12, 2015, 1:43 PM
 */

#ifndef NEURALNETWORK_H
#define	NEURALNETWORK_H

#include <vector>
#include "Eigen/Dense"
#include "DataTypes.h"

class NeuralNetwork
{
    public:

        /**
         * Constructor.
         * @param layers VectorN containing the number of neurons on each layer.
         */
        NeuralNetwork( const std::vector<int>& layers );
        
        /**
         * Constructor. It receives a filename of a previous trained Neural Network.
         * @param path File to be loaded.
         */
        NeuralNetwork( const std::string& path );
        
        /**
	 * Insert specific values into the weight vectors.
	 * Example: for a 3-Layer Net, if I want to insert values into the neurons on third Layer.
	 * I just need to set the weights from position 2; -> the index begins IN zero!!!
	 * The first Layer (Layer zero) doesn't have Weights before it, right???
	 */
	int setWeights( int outLayer, int neuronId, int prevLayerNeuronId, FLOAT value );
        
        void setWeights( int inputLayer, const MatrixNxM& weights );
        
        /**
	 * Get specific values from the weight vectors.
	 * Example: for a 3-Layer Net, if I want to insert values into the neurons on third Layer.
	 * I just need to set the weights from position 2; -> the index begins IN zero!!!
	 * The first Layer (Layer zero) doesn't have Weights before it, right???
	 */
	int getWeights( int outLayer, int neuronId, int prevLayerNeuronId, FLOAT& value );
        
        MatrixNxM getWeights( int inputLayer );
        
        const std::vector<int>& getLayers()
        {
            return _layers;
        }
        
        /**
	 * Example: for a 3-Layer Net, if I want to train the third Layer
	 * I just need to call activateCalculus( Net, 2 ); -> the indice begins with zero!!!
	 * The first Layer (Layer zero) doesn't have Weights before it, right???
	 */
	int activateCalculus( int layer );
        
        /**
	 * Function used to calculate the activation function value for any input, based on the table.
	 * @param x Input to calculate the corresponding output value of the function.
	 * @return Returns the function output value for the corresponding x input.
	 */
	FLOAT activationFunc( FLOAT x )
        {
            return 1.0f / (1.0f + (FLOAT)exp(-x));
        }
        
        /** 
	 * Function used to find the activation outputs for all the neurons of the Net, calling ActivCalculus for each of the Layers.
	 * @return Returns -2 if the value pointed by Net is NULL, or 0 if ok.
	 */
	void activateAll();
        
        /**
	 * Function that calculates the Net's output error
	 * @param targets Neuron's targeted output.
	 * @return Returns the resulting error.
	 */
	FLOAT outputError( const std::vector<FLOAT>& targets );
        
        /**
	 * Function that inserts entry values on the input Layer of the Net.
	 */
	int setEntry( const std::vector<FLOAT>& entry );
        
        /**
	 * To train the neural Net, we'll need the activation function derivative's value of any
	 * point. This function returns this derivative.
	 * @param f Input value. The activation function derivative will be calculated for this value.
	 * @return Returns the value of the activation function derivative.
	 */
	FLOAT activationDerivative( FLOAT f )
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
	void outGradient( const std::vector<FLOAT>& targets );
        
        /**
	 * Function used to train the hidden Layers's neurons.
	 * @param Net Net being used.
	 * @param Layer Layer to receive the Gradients based on the already calculated Weights.
	 * @return Returns -2 if Net points to NULL, -1 if dimTargets is less than zero, or 0 if ok.
	 */
	void hiddenGradient( int layer );
        
        /**
	 * Function used to calculate the gradients of the last Layer of neurons.
	 * @param targets Target vector (desired outputs).
	 * @param dimTargets Dimension of the target vector.
	 * @param MFactor Momentum factor.
	 */
	void outGradientMtm( const std::vector<FLOAT>& targets, FLOAT MFactor );
        
        /**
	 * Function used to train the hidden Layers's neurons.
	 * @param Layer Layer to receive the Gradients based on the already calculated Weights.
	 * @param MFactor Momentum factor.
	 * @return Returns -2 if Net points to NULL, -1 if dimTargets is less than zero, or 0 if ok.
	 */
	int hiddenGradientMtm( int layer, FLOAT MFactor );
        
        /**
	 * Function that update the Weights on the Net.
	 * @param Net Net being used.
	 * @param rate The Learning rate.
	 * @return Returns -2 if Net points to NULL, -1 if the value of rate is not between 0.0 and 1.0, or 0.
	 */
	int updateWeights( FLOAT rate );
        
        /**
	 * Recebe um vetor de tamanho n e desloca os valores 1 posicao a esquerda.
	 * O primeiro valor (o valor contido na posicao de indice 0 do vetor) sera deletado.
	 */
	bool shiftLeft( VectorN& vector, FLOAT newValue, int newValuePosition );
        
        /**
	 * Function used to train the net.
	 * @param size The size of entry vector.
	 * @param entry VectorN that contains values of entries.
	 * @param size2 The size of out vector.
	 * @param out VectorN that contains values of outs.
	 * @param l_rate The learning rate.
	 * @param momentum Momentum factor.
	 * @return returns 1 if ok.
	 */
	int train( const std::vector<FLOAT>& entry, std::vector<FLOAT>& out, FLOAT l_rate, FLOAT momentum );
        
        /**
	 * Once the net is ready, we need a function to use it.
	 * @param size Length of the entry vector.
	 * @param entry The entries vector.
	 * @return It returns 0 if ok.
	 */
	int use( const std::vector<FLOAT>& entry );
        
        /**
	 * Function that gets out values from the Net.
	 */
	int getOut( std::vector<FLOAT>& out );
        
        /**
	 * Obtem o erro medio quadratico de um conjunto de amostras fornecido.
	 */
	void RMSError( MatrixNxM& inMatrix, MatrixNxM& outMatrix, FLOAT* retRMS_Error );

    private:
        
        /**
	 * Inserts random numbers between 0.0 and 0.1 as weight values.
	 */
        void init();

        /**
         * Function used to create a new neural Net.
         */
        void create( const std::vector<int>& layers );
        
        /**
         * Layer structure containing the number of neurons on each layer.
         */
        std::vector<int> _layers;
        
        /**
         * Output value of each neuron.
         */
        std::vector< VectorN > _neurons;
        
        /**
         * Auxiliary vector to save temporary deltas while training the net.
         */
        std::vector< VectorN > _deltas;
        
        /**
         * Connection Weights of each neuron from Layer n to Layer n+1.
         */
        std::vector< MatrixNxM > _weights;
        
        /**
         * Gradients of each Weight.
         */
        std::vector< MatrixNxM > _gradients;
        
        /**
         * Temporary Gradients of each Weight (Momentum).
         */
        std::vector< MatrixNxM > _gradientsMt;
        
        /**
         * Auxiliary flag to indicate this is not the first train.
         */
        int _flagMomentum;
};


#endif	/* NEURALNETWORK_H */

