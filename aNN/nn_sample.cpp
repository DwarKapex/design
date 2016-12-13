#include "neuralNetwork.h"

neuralNetwork::neuralNetwork( const int inputSize, const int outputSize, 
                              const int depth, const int hiddenLayerSize, const char * nnType )
{
	if(inputSize > 0 && outputSize > 0){

		mMinMSE	= 0.01f;
		mMeanSquaredError = 0;
		mInputs = inputSize;
		mOutputs = outputSize;
		mHidden = hiddenLayerSize;

		networkFunction * outputNeuronsFunc;
		networkFunction * inputNeuronsFunc;


		std::vector<neuron *> outputLayer;
		std::vector<neuron *> inputLayer;


		if( strcmp( nnType, "MultiLayerPerceptron" ) == 0){
			mNeuronFactory = new perceptronNeuronFactory;
			mTrainingAlgoritm = new backPropagation(this);

			outputNeuronsFunc = new bipolarSigmoid;
			inputNeuronsFunc = new Linear;
		}

		// output layer
		for(int iOutput = 0; iOutput < outputSize; ++iOutput){
			outputLayer.push_back( mNeuronFactory->createOutputNeuron(outputNeuronsFunc) );
		}
		mLayers.push_back(outputLayer);

		// hidden layers 
		for(int i = 0; i < depth; ++i){
			std::vector<neuron *> hiddenLayer;
			for(int j = 0; j < hiddenLayerSize; ++j ){
				neuron* hidden = mNeuronFactory->createHiddenNeuron(mLayers[0], outputNeuronsFunc);
				hiddenLayer.push_back(hidden);
			}
			mBiasLayer.insert(mBiasLayer.begin(), mNeuronFactory->createInputNeuron(mLayers[0], inputNeuronsFunc));
			mLayers.insert(mLayers.begin(),hiddenLayer);
		}

		// input layers

		for(int iInput = 0; iInput < inputSize; ++iInput){
			inputLayer.push_back(mNeuronFactory->createInputNeuron(mLayers[0], inputNeuronsFunc));
		}
		mBiasLayer.insert(mBiasLayer.begin(),mNeuronFactory->createInputNeuron(mLayers[0], inputNeuronsFunc));
		mLayers.insert(mLayers.begin(),inputLayer);

		mTrainingAlgoritm->weightsInitialization();
	}
	else{
		std::cout << "Error: the number of input and output neurons must be more than 0!\n";
	}
}

neuralNetwort::~neuralNetwork()
{
	delete mNeuronFactory;
	delete mTrainingAlgoritm;

	for( size_t iBias = 0; iBias < mBiasLayer.size(); ++iBias ){
		delete mBiasLayer[ iBias ];
	}

	for( size_t iLayer = 0; iLayer < mLayers.size(); ++iLayer){
		for( size_t iNeuron = 0; iNeuron < mLayers[iLayer].size(); ++iNeuron){
			delete mLayers[ iLayer ].at( iNeuron );
		}
	}

}

bool neuralNetwork::train( const std::vector<std::vector<float>>& data, const std::vector<std::vector<float>>& target )
{
	bool goOn = true;
	int iIter = 0;
	while(goOn){
		++iIter;
		for(size_t i = 0; i < data.size(); ++i)
			mTrainingAlgoritm->train( data[i], target[i] );
		
		double MSE = GetMSE();
		if( MSE < mMinMSE){
			std::cout << iIter << ": finish training. Error: " << MSE << std::endl;
			goOn = false;
		}
		resetMSE();
	}
	return goOn;
}

std::vector<float> neuralNetwork::getNetworkOutput( const std::vector<float>& data )
{
	std::vector<float> networkOutput;
	if(data.size() != mInputs){
		std::cout << "Error input size, want : " << mInputs << std::endl;
		return networkOutput;
	}
	else{
		for(size_t iData = 0; iData < getInputLayer().size(); ++iData){
			getInputLayer().at(iData)->Input(data[iData]);
		}

		for(size_t iLayer = 0; iLayer < mLayers.size() - 1; ++iLayer){
			mBiasLayer[iLayer]->Input(1.0);

			for(size_t iData = 0; iData < mLayers.at(iLayer).size(); ++iData){
				mLayers.at(iLayer).at(iData)->perform();
			}
			mBiasLayer[iLayer]->perform();
		}

		std::cout << "Network output :  ";
                double res = 0.0f;
		for(size_t i = 0; i < mOutputs; ++i){
			res = getOutputLayer().at(i)->perform();
			std::cout << res << " ";
		}
		std::cout << std::endl;
		resetCharges();
		return networkOutput;
	}
}

void neuralNetwork::resetCharges()
{
	for(size_t i = 0; i < mLayers.size(); ++i)
		for(size_t iElement = 0; iElement < mLayers.at(i).size(); ++iElement)
			mLayers.at(i).at(iElement)->resetChargesSum();
		
	for(size_t i = 0; i < mLayers.size()-1; ++i)
			mBiasLayer[i]->resetChargesSum();
}

void neuralNetwork::weightUpdate()
{
	for(size_t iLayer = 0; iLayer < mLayers.size(); ++iLayer)
		for(size_t iNeuron = 0; iNeuron < mLayers[iLayer].size(); ++iNeuron)
			mLayers[iLayer].at(iNeuron)->wUpdate();
}

void neuralNetwork::showNetworkState()
{
	for(size_t iLayer = 0; iLayer < mLayers.size(); ++iLayer){
		std::cout << "Layer index: " << iLayer << std::endl;
		for(size_t iNeuron = 0; iNeuron < mLayers[iLayer].size(); ++iNeuron){
			std::cout << "  neuron index: " << iNeuron << std::endl;
			mLayers[iLayer].at(iNeuron)->showNeuronState();
		}
		if(iLayer < mBiasLayer.size()){
			std::cout << "  Bias: " << std::endl;
			mBiasLayer[iLayer]->showNeuronState();
		}
	}
}