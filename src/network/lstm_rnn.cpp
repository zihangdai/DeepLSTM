#include <sstream> 
#include "lstm_rnn.h"

using namespace std;

// #define DEBUG_LSTM_RNN

LSTM_RNN::LSTM_RNN(ConfReader *confReader) {
	/* read conf and allocate memory */
	m_numLayer = confReader->getInt("num_layer");
	m_numNeuronList = new int[m_numLayer];
	m_layerTypeList = new string[m_numLayer];
	m_connTypeList = new string[m_numLayer-1];

	// type and number of neurons of each layer
	for (int layerIdx=0; layerIdx<m_numLayer; ++layerIdx) {
		std::stringstream ss;
  		ss << layerIdx;
		m_numNeuronList[layerIdx] = confReader->getInt("num_neuron_layer_" + ss.str());
		m_layerTypeList[layerIdx] = confReader->getString("type_layer_" + ss.str());
	}	
	
	// type of each conectection
	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		std::stringstream ss;
  		ss << connIdx;
		m_connTypeList[connIdx] = confReader->getString("type_connection_" + ss.str());
	}
	
	m_nParamSize = 0;
	m_maxSeqLen = confReader->getInt("max_sequence_length");

	/* initialize layers */
	for (int layerIdx=0; layerIdx<m_numLayer; layerIdx++) {
		RecurrentLayer *layer = initLayer(layerIdx);
		m_nParamSize += layer->m_nParamSize;
		m_vecLayers.push_back(layer);
	}

	/* initialize connections */
	for (int connIdx=0; connIdx<m_numLayer-1; connIdx++) {
		RecConnection *conn = initConnection(connIdx);
		m_nParamSize += conn->m_nParamSize;
		m_vecConnections.push_back(conn);
	}
	#ifdef DEBUG_LSTM_RNN
	printf("LSTM_RNN Constructor finished\n");
	#endif
}

LSTM_RNN::~LSTM_RNN() {
	if (!m_numNeuronList) {delete [] m_numNeuronList;}
	if (!m_layerTypeList) {delete [] m_layerTypeList;}
	if (!m_connTypeList) {delete [] m_connTypeList;}
	
	for (int layerIdx=0; layerIdx<m_numLayer; ++layerIdx) {
		if (!m_vecLayers[layerIdx]) {delete [] m_vecLayers[layerIdx];}
	}

	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		if (!m_vecConnections[connIdx]) {delete [] m_vecConnections[connIdx];}
	}
}

RecConnection *LSTM_RNN::initConnection(int connIdx) {
	string connType = m_connTypeList[connIdx];
	#ifdef DEBUG_LSTM_RNN
	printf("connType[%d]:%s\n",connIdx,connType.c_str());
	#endif
	RecurrentLayer *preLayer = m_vecLayers[connIdx];
	RecurrentLayer *postLayer = m_vecLayers[connIdx+1];
	RecConnection *conn;
	if (connType == "full_connection") {
		conn = new FullConnection(preLayer, postLayer);
	} else if (connType == "lstm_connection") {
		conn = new LSTMConnection(preLayer, postLayer);
	} else {
		printf("Error in initConnection.\n");
		exit(-1);
	}
	return conn;
}

RecurrentLayer *LSTM_RNN::initLayer(int layerIdx) {
	string layerType = m_layerTypeList[layerIdx];
	#ifdef DEBUG_LSTM_RNN
	printf("layerType[%d]:%s\n",layerIdx,layerType.c_str());
	#endif
	int numNeuron = m_numNeuronList[layerIdx];
	RecurrentLayer *layer;
	if (layerType == "input_layer") {
		layer = new InputLayer(numNeuron, m_maxSeqLen);
	} else if (layerType == "lstm_layer") {
		int inputSize = m_numNeuronList[layerIdx-1];
		layer = new LSTMLayer(numNeuron, m_maxSeqLen, inputSize);		
	} else if (layerType == "softmax_layer") {
		m_errorType = "cross_entropy_error";
		layer = new SoftmaxLayer(numNeuron, m_maxSeqLen);
	} else if (layerType == "mse_layer") {
		m_errorType = "mean_squared_error";
		layer = new MSELayer(numNeuron, m_maxSeqLen);
	} else {
		printf("Error in initLayer.\n");
		exit(-1);
	}
	return layer;
}

void LSTM_RNN::initParams(float *params) {
	#ifdef DEBUG_LSTM_RNN
	printf("LSTM_RNN init parameters.\n");
	#endif
	float *cursor = params;
	// layer part
	for (int layerIdx=0; layerIdx<m_numLayer; ++layerIdx) {
		m_vecLayers[layerIdx]->initParams(cursor);
		cursor += m_vecLayers[layerIdx]->m_nParamSize;
	}
	// connection part
	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		m_vecConnections[connIdx]->initParams(cursor);
		cursor += m_vecConnections[connIdx]->m_nParamSize;
	}
}

void LSTM_RNN::feedForward(float *sampleData, int inputSeqLen) {
	float *dataCursor = sampleData;
	/* bind input sequence to the input layer */
	RecurrentLayer *curLayer = m_vecLayers[0];
	int curNumNeuron = curLayer->m_numNeuron;
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		memcpy(curLayer->m_outputActs[seqIdx], dataCursor, sizeof(float)*curNumNeuron);
		dataCursor += curNumNeuron;
	}
	/* feed forward through connections and layers */
	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		m_vecConnections[connIdx]->feedForward(inputSeqLen);
		m_vecLayers[connIdx+1]->feedForward(inputSeqLen);
	}
}

void LSTM_RNN::feedBackward(float *sampleLabel, int inputSeqLen) {
	float *labelCursor = sampleLabel;
	/* bind target label to output layer */
	RecurrentLayer *curLayer = m_vecLayers[m_numLayer-1];
	int curNumNeuron = curLayer->m_numNeuron;
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		// bind target label to m_outputErrs of output layer
		memcpy(curLayer->m_outputErrs[seqIdx], labelCursor, sizeof(float)*curNumNeuron);		
		labelCursor += curNumNeuron;
	}

	/* feed backward through connections and layers */
	curLayer->feedBackward(inputSeqLen);
	for (int connIdx=m_numLayer-2; connIdx>=0; --connIdx) {
		m_vecConnections[connIdx]->feedBackward(inputSeqLen);
		m_vecLayers[connIdx]->feedBackward(inputSeqLen);
	}	
}

float LSTM_RNN::computeError(float *sampleLabel, int inputSeqLen) {
	float sampleError = 0.f;
	float *labelCursor = sampleLabel;
	RecurrentLayer *curLayer = m_vecLayers[m_numLayer-1];
	int curNumNeuron = curLayer->m_numNeuron;
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		for (int i=0; i<curNumNeuron; ++i) {
			if (m_errorType == "cross_entropy_error") {
				sampleError += labelCursor[i] * log(curLayer->m_outputActs[seqIdx][i]);
			} else if (m_errorType == "mean_squared_error") {
				float diff = labelCursor[i] - curLayer->m_outputActs[seqIdx][i];
				sampleError += diff * diff;
			}
		}
		labelCursor += curNumNeuron;
	}
	return sampleError;
}

float LSTM_RNN::computeGrad(float *grad, float *params, float *data, float *label, int minibatchSize) {
	float error = 0.f;
	
	memset(grad, 0x00, sizeof(float)*m_nParamSize);
	bindWeights(params, grad);
	
	/*** feed forward and feed backward ***/
	for (int dataIdx=0; dataIdx<minibatchSize; ++dataIdx) {
		// TODO
		int inputSeqLen = m_maxSeqLen;
		float *sampleData = data + dataIdx * m_vecLayers[0]->m_numNeuron;
		float *sampleLabel = label + dataIdx * m_vecLayers[m_numLayer-1]->m_numNeuron;
		
		// feedforward
		feedForward(sampleData, inputSeqLen);

		// compute error
		error += computeError(sampleLabel, inputSeqLen);		

		// feedbackword
		feedBackward(sampleLabel, inputSeqLen);

		/* reset internal states of LSTM layers */
		resetStates(inputSeqLen); // this is subject to change
		
	}

	// normalization by number of input sequences and clip gradients to [-1, 1]
	float normFactor = 1.f / (float) minibatchSize;
	for (int dim=0; dim<m_nParamSize; ++dim) {
		grad[dim] *= normFactor;
		if (grad[dim] < -1.f) {
			grad[dim] = -1.f;
		} else if (grad[dim] > 1.f) {
			grad[dim] = 1.f;
		}
	}
	error *= normFactor;

	return error;
}

void LSTM_RNN::resetStates(int inputSeqLen) {
	for (int layerIdx=0; layerIdx<m_numLayer; ++layerIdx) {
		m_vecLayers[layerIdx]->resetStates(inputSeqLen);
	}
}

void LSTM_RNN::bindWeights(float *params, float *grad) {
	// define cursors
	float *paramsCursor = params;
	float *gradCursor = grad;
	// layer part
	for (int layerIdx=0; layerIdx<m_numLayer; ++layerIdx) {
		m_vecLayers[layerIdx]->bindWeights(paramsCursor, gradCursor);
		paramsCursor += m_vecLayers[layerIdx]->m_nParamSize;
		gradCursor += m_vecLayers[layerIdx]->m_nParamSize;
	}
	// connection part
	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		m_vecConnections[connIdx]->bindWeights(paramsCursor, gradCursor);
		paramsCursor += m_vecConnections[connIdx]->m_nParamSize;
		gradCursor += m_vecConnections[connIdx]->m_nParamSize;
	}
}