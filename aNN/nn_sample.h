#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "neuronFactory.h"
#include "trainAlgorithm.h"
#include <cstring>
#include <iostream>

class trainAlgorithm;

// Neural network class.
 
class neuralNetwork
{
public:
	neuralNetwork(  const int inputSize,
			const int outputSize,
			const int depth = 0,
			const int hiddenLayerSize = 0,
			const char * nnType = "MultiLayerPerceptron");

	~neuralNetwork();

	bool train( const std::vector<std::vector<float> >& data,
                    const std::vector<std::vector<float > >& target );

	std::vector<int> getNetworkOutput( const std::vector<float>& data );

	void setAlgorithm( trainAlgorithm* inTrainingAlgorithm )
                                        { mTrainingAlgoritm = inTrainingAlgorithm; };

	void setNeuronFactory( neuronFactory * inNeuronFactory )
                                        { mNeuronFactory = inNeuronFactory; };

	void showNetworkState();

	const double getMinMSE() { return mMinMSE; };

	void setMinMSE( const double inMinMse ) 	{ mMinMSE = inMinMse; };

	friend class  backPropagation;

protected:

	std::vector<neuron *> &	getLayer( const int idx )
                            { return mLayers[idx]; };


	size_t	size()	{ return mLayers.size(); };

	std::vector<neuron *>&	getOutputLayer()
                    { return mLayers[mLayers.size()-1]; };

	std::vector<neuron *>&	getInputLayer()
                        { return mLayers[0]; };

	std::vector<neuron *>& 	getBiasLayer()
                        { return mBiasLayer; };

	void weightUpdate();

	void resetCharges();

	void addMSE( const double piece )					
                            { mMeanSquaredError += piece; };

	double getMSE()
                    { return mMeanSquaredError; };
                    
	void resetMSE()
                { mMeanSquaredError = 0; };


	neuronFactory *                     mNeuronFactory;
	trainAlgorithm *                    mTrainingAlgoritm;			
	std::vector<std::vector<neuron *>> mLayers;
	std::vector<neuron *>               mBiasLayer;					
	size_t                              mInputs, mOutputs, mHidden;			
	double                              mMeanSquaredError;				
	double                              mMinMSE;					
};




#endif // NEURALNETWORK_H_