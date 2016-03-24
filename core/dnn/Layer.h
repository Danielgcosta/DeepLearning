/* 
 * This is a base class for layer creation in a neural networks.
 * File:   Layer.h
 * Author: ederperez
 */

#ifndef LAYER_H
#define LAYER_H


namespace gpu
{
    template<typename T>
    class CUDAMatrix;
}


namespace dnn
{

struct LayerParameters
{
    LayerParameters( int inputs, int outputs ) : inputs( inputs ), outputs( outputs ) {}

    // Number of input neurons
    int inputs;
    
    // Number of output neurons
    int outputs;
};

template<typename T>
class Layer
{
    public:
        Layer( const LayerParameters& parameters ) : mParameters( parameters ) {}
        virtual ~Layer() {}

        /**
         * Executes the forward pass. The structure of input/output data
         * is given by LayerParameters.
         * 
         * @param deviceInput Input neurons stored in device memory.
         * @param deviceOutput Output neurons stored in device memory.
         * @param batchSize Number of samples.
         */
        virtual void forwardPass( const void* deviceInput, void* deviceOutput, int batchSize = 1 ) = 0;
        
        /**
         * Executes the backward pass. The structure of input and output data
         * is given by LayerParameters.
         * 
         * @param deviceInput Input stored in device memory.
         * @param deviceOutput Output stored in device memory.
         * @param batchSize Number of samples.
         */
        virtual void backwardPass( const void* deviceInput, void* deviceOutput, int batchSize = 1 ) = 0;

        /**
         * Gets the weights matrix.
         */
        virtual const gpu::CUDAMatrix<T>& getWeights() = 0;

        /**
         * Sets new weights.
         */
        virtual bool setWeights( const gpu::CUDAMatrix<T>& weights ) = 0;
        
        virtual int inputs()
        {
            return mParameters.inputs;
        }

        virtual int outputs()
        {
            return mParameters.outputs;
        }

    protected:

        LayerParameters mParameters;
};

}

#endif
