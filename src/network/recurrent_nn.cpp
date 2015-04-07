#include <sstream> 
#include "neural_net.h"

feedForwardNN::feedForwardNN (ConfReader *confReader, int minibatchSize) {
	m_nMinibatchSize = minibatchSize;

	m_numLayer = confReader->getInt("num_layer");
	m_numNeuronList = new int[m_numLayer];
	m_layerTypeList = new int[m_numLayer];

	for (int layer=0; layer<m_numLayer; ++layer) {
		std::stringstream ss;
  		ss << layer;
		m_numNeuronList[layer] = confReader->getInt("num_neuron_layer_" + ss.str());
		m_layerTypeList[layer] = confReader->getInt("type_layer_" + ss.str());
	}

	// Initialize layers
	for (int i=0; i<m_numLayer; i++) {
		int numNeuron = m_numNeuronList[i];
		int layerType = m_layerTypeList[i];
		layerBase *layer = initLayer(numNeuron, layerType);

		m_vecLayers.push_back(layer);
	}
	m_softmaxLayer = new softmaxLayer(m_numNeuronList[m_numLayer-1]);

	// Allocate memory to hold forward input to each non-input layer
	for (int connectIdx=0; connectIdx<m_numLayer-1; connectIdx++) {
		int numNeuron = m_numNeuronList[connectIdx+1];
		float * forwardInfo = new float[numNeuron];

		m_vecForwardInfo.push_back(forwardInfo);
	}

	// Allocate memory to hold backprop error to each non-output layer
	for (int connectIdx=0; connectIdx<m_numLayer-1; connectIdx++) {
		int numNeuron = m_numNeuronList[connectIdx];
		float * backpropInfo = new float[numNeuron];

		m_vecBackpropInfo.push_back(backpropInfo);
	}

	// Compute m_nParamSize
	m_nParamSize = 0;
	for (int connectIdx=0; connectIdx<m_numLayer-1; connectIdx++) {
		int fanIn = m_numNeuronList[connectIdx]+1;
		int fanOut = m_numNeuronList[connectIdx+1];

		m_nParamSize += fanIn*fanOut;
	}

	printf("Constructor feedForwardNN finished\n");
}

feedForwardNN::~feedForwardNN() {
	if(!m_numNeuronList) {
		delete [] m_numNeuronList;
	}
	if(!m_layerTypeList) {
		delete [] m_layerTypeList;
	}
	for (std::vector<layerBase *>::iterator it = m_vecLayers.begin(); it != m_vecLayers.end(); ++it) {
		if(!(*it)) {
			delete *it;
		}
	}
	for (std::vector<float *>::iterator it = m_vecBackpropInfo.begin(); it != m_vecBackpropInfo.end(); ++it) {
		if(!(*it)) {
			delete [] *it;
		}
	}
	for (std::vector<float *>::iterator it = m_vecForwardInfo.begin(); it != m_vecForwardInfo.end(); ++it) {
		if(!(*it)) {
			delete [] *it;
		}
	}
}

layerBase *feedForwardNN::initLayer (int numNeuron, int layerType) {
	layerBase *layer;
	switch (layerType) {
		case 0:
			layer = new linearLayer(numNeuron);
			break;
		case 1:
			layer = new sigmoidLayer(numNeuron);
			break;		
		default:
			printf("Error in initLayer.");
			exit(-1);
	}
	return layer;
}

void feedForwardNN::initParams (float *params) {
	// Initialize values for weights
	float *cursor = params;
	for (int connectIdx=0; connectIdx<m_numLayer-1; ++connectIdx) {
		int fanIn = m_numNeuronList[connectIdx]+1;
		int fanOut = m_numNeuronList[connectIdx+1];

		float *weights = cursor;
		initWeights(weights, fanIn, fanOut);

		cursor += fanIn * fanOut;
	}
}

void feedForwardNN::initWeights (float *weights, int fanIn, int fanOut, int type) {
	switch(type) {
		case 0: {
			float multiplier = 4.f * sqrt(6.f / (float)(fanIn + fanOut));
			for (int i=0; i<fanIn*fanOut; i++) {
				weights[i] = multiplier * SYM_UNIFORM_RAND;
			}
			break;
		}
		default: {
			printf("Error in initWeights.");
			exit(-1);
		}
	}
}

void feedForwardNN::feedForward (float *input) {	
	layerBase *inLayer, *outLayer;
	float *forwarInfo;
	float *weights;

	// input layer
	m_vecLayers[0]->activateFunc(input);

	// for each connection
	for (int connectIdx=0; connectIdx<m_numLayer-1; ++connectIdx) {

		// get fanIn and fanOut
		int fanIn = m_numNeuronList[connectIdx]+1;
		int fanOut = m_numNeuronList[connectIdx+1];

		// create ptr to corresponding weights
		weights = m_vecWeights[connectIdx];

		// create ptr to associated layers (in & out)
		inLayer = m_vecLayers[connectIdx];
		outLayer = m_vecLayers[connectIdx+1];

		// create ptr to associated forwardInfo & clean the buffer
		forwarInfo = m_vecForwardInfo[connectIdx];
		memset(forwarInfo, 0x00, sizeof(float)*m_numNeuronList[connectIdx+1]);

		// compute forwarInfo 		
		for (int in=0; in<fanIn-1; in++) {			
			float inAct = inLayer->m_activation[in];
			int wIdx = in * fanOut;
			for (int out=0; out<fanOut; out++) {
				forwarInfo[out] += inAct * weights[wIdx+out];
			}
		}
		// bias term
		int wIdx = (fanIn-1) * fanOut;
		for (int out=0; out<fanOut; out++) {
			forwarInfo[out] += weights[wIdx+out];
		}

		// outLayer compute activation
		outLayer->activateFunc(forwarInfo);
	}

	// softmax classification
	m_softmaxLayer->activateFunc(m_vecLayers[m_numLayer-1]->m_activation);
	// for (int i=0; i<m_numNeuronList[m_numLayer-1]; i++) {
	// 	printf("m_activation[%d]: %f\n", i, m_softmaxLayer->m_activation[i]);
	// }
}

void feedForwardNN::backProp (float *target) {
	layerBase *inLayer, *outLayer;
	float *backpropInfo;
	float *weights, *weightsGrad;

	// softmax layer
	m_softmaxLayer->computeDelta(target);
	// for (int i=0; i<m_numNeuronList[m_numLayer-1]; i++) {
	// 	printf("m_delta[%d]: %f\n", i, m_softmaxLayer->m_delta[i]);
	// 	printf("target[%d]: %f\n", i, target[i]);
	// }

	// computeDelta for output layer (error signal is different for output layer)
	m_vecLayers[m_numLayer-1]->computeDelta(m_softmaxLayer->m_delta);

	// for each connection
	for (int connectIdx=m_numLayer-2; connectIdx>=0; --connectIdx) {
		// printf("connectIdx: %d\n", connectIdx);
		// get fanIn and fanOut
		int fanIn = m_numNeuronList[connectIdx]+1;
		int fanOut = m_numNeuronList[connectIdx+1];

		// create ptr to corresponding weights
		weights = m_vecWeights[connectIdx];

		// create ptr to associated layers (in & out)
		inLayer = m_vecLayers[connectIdx];
		outLayer = m_vecLayers[connectIdx+1];

		// create ptr to corresponding weightsGrad
		weightsGrad = m_vecWeightsGrad[connectIdx];
		
		// create ptr to associated backpropInfo & clean the buffer
		backpropInfo = m_vecBackpropInfo[connectIdx];
		memset(backpropInfo, 0x00, sizeof(float)*m_numNeuronList[connectIdx]);

		// compute weightsGrad
		// printf("compute weightsGrad\n");
		for (int in=0; in<fanIn-1; in++) {
			float inAct = inLayer->m_activation[in];
			int wIdx = in * fanOut;
			for (int out=0; out<fanOut; out++) {
				weightsGrad[wIdx+out] += inAct * outLayer->m_delta[out];
			}
		}
		// weightsGrad for bias term
		// printf("weightsGrad for bias term\n");
		int wIdx = (fanIn-1) * fanOut;
		for (int out=0; out<fanOut; out++) {
			weightsGrad[wIdx+out] += outLayer->m_delta[out];
		}
		
		// compute backpropInfo
		// printf("compute backpropInfo\n");
		for (int in=0; in<fanIn-1; in++) {
			int wIdx = in * fanOut;
			for (int out=0; out<fanOut; out++) {
				backpropInfo[in] += weights[wIdx+out] * outLayer->m_delta[out];
			}
		}

		// non-input inLayer computeDelta
		if (connectIdx > 0) {
			inLayer->computeDelta(backpropInfo);
		}
	}

}

float feedForwardNN::computeGrad (float *grad, float *params, float *data, float *label) {
	// bind weights
	// printf("Bind weights\n");
	float *paramsCursor = params;
	for (int connectIdx=0; connectIdx<m_numLayer-1; ++connectIdx) {
		int fanIn = m_numNeuronList[connectIdx]+1;
		int fanOut = m_numNeuronList[connectIdx+1];

		float *weights = paramsCursor;
		m_vecWeights.push_back(weights);

		paramsCursor += fanIn * fanOut;
	}

	// clean the grad buffer and bind it to 
	// printf("Bind weights grad\n");
	memset(grad, 0x00, sizeof(float)*m_nParamSize);
	float *gradCursor = grad;
	for (int connectIdx=0; connectIdx<m_numLayer-1; ++connectIdx) {
		int fanIn = m_numNeuronList[connectIdx]+1;
		int fanOut = m_numNeuronList[connectIdx+1];

		float *weightsGrad = gradCursor;		
		m_vecWeightsGrad.push_back(weightsGrad);

		gradCursor += fanIn * fanOut;
	}

	// compute grad by several forward pass and backward pass
	int dataDim = m_numNeuronList[0];
	int labelDim = m_numNeuronList[m_numLayer-1];
	float *dataCursor = data;
	float *oneOnlabel = new float[dataDim];
	int labelInt;
	float error = 0.f;
	float correctCount = 0.f;

	for (int dataIdx=0; dataIdx<m_nMinibatchSize; dataIdx++) {
		// produce one-on label representation
		memset(oneOnlabel, 0x00, sizeof(float)*labelDim);
		labelInt = (int)label[dataIdx];
		if (labelInt < 0) {
			labelInt = 0;
		}
		oneOnlabel[labelInt] = 1.f;
		
		// feedforward and backpropagation
		feedForward(dataCursor);
		backProp(oneOnlabel);

		// compute some statistics
		float maxP = 0.f;
		int maxIndex = -1;
		for (int i=0; i<labelDim; i++) {
			error += - oneOnlabel[i] * log(m_softmaxLayer->m_activation[i]);
			if (m_softmaxLayer->m_activation[i] > maxP) {
				maxP = m_softmaxLayer->m_activation[i];
				maxIndex = i;
			}			
		}
		if (maxIndex == labelInt) {
			correctCount ++;
		}

		// move the dataCursor
		dataCursor += dataDim;		
	}
	printf("Error: %f\n", error / m_nMinibatchSize);
	printf("Correct rate: %f\n", correctCount / m_nMinibatchSize);

	for (int dim=0; dim<m_nParamSize; ++dim) {
		grad[dim] /= m_nMinibatchSize;
	}

	delete [] oneOnlabel;
	return error;
}