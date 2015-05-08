#include <sstream> 
#include "rnn_lstm.h"

using namespace std;

#define DEBUG_RNNLSTM 0

RNNLSTM::RNNLSTM(boost::property_tree::ptree *confReader, string section) {
	// read basic conf 
	m_numLayer = confReader->get<int>(section + "num_layer");
	m_maxSeqLen = confReader->get<int>(section + "max_sequence_length");
	
	m_inputSize = confReader->get<int>(section + "input_size");
	m_outputSize = confReader->get<int>(section + "output_size");

	// allocate memory
	m_numNeuronList = new int[m_numLayer];
	m_layerTypeList = new string[m_numLayer];
	m_connTypeList = new string[m_numLayer-1];

	// read type and number of neurons of each layer from conf
	for (int layerIdx=0; layerIdx<m_numLayer; ++layerIdx) {
		stringstream ss;
  		ss << layerIdx;
		m_numNeuronList[layerIdx] = confReader->get<int>(section + "num_neuron_layer_" + ss.str());
		m_layerTypeList[layerIdx] = confReader->get<string>(section + "type_layer_" + ss.str());
	}
	
	// read type of each conectection from conf
	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		stringstream ss;
  		ss << connIdx;
		m_connTypeList[connIdx] = confReader->get<string>(section + "type_connection_" + ss.str());
	}
	
	// m_paramSize based on the m_paramSize of layers and connections
	m_paramSize = 0;

	// initialize layers
	for (int layerIdx=0; layerIdx<m_numLayer; layerIdx++) {
		RecurrentLayer *layer = initLayer(layerIdx);
		m_paramSize += layer->m_paramSize;
		m_vecLayers.push_back(layer);
	}	

	// initialize connections
	for (int connIdx=0; connIdx<m_numLayer-1; connIdx++) {
		RecurrentConnection *conn = initConnection(connIdx);
		m_paramSize += conn->m_paramSize;
		m_vecConnections.push_back(conn);
	}

	DLOG_IF(INFO, DEBUG_RNNLSTM) << "RNNLSTM deconstructor." << endl;
}

RNNLSTM::~RNNLSTM() {
	if (m_numNeuronList != NULL) {delete [] m_numNeuronList;}
	if (m_layerTypeList != NULL) {delete [] m_layerTypeList;}
	if (m_connTypeList != NULL) {delete [] m_connTypeList;}
	
	for (int layerIdx=0; layerIdx<m_numLayer; ++layerIdx) {
		if (m_vecLayers[layerIdx] != NULL) {delete m_vecLayers[layerIdx];}
	}

	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		if (m_vecConnections[connIdx] != NULL) {delete m_vecConnections[connIdx];}
	}
}

RecurrentConnection *RNNLSTM::initConnection(int connIdx) {
	string connType = m_connTypeList[connIdx];
	DLOG_IF(INFO, DEBUG_RNNLSTM) << "connType[" << connIdx << "]:"<< connType << endl;
	
	RecurrentLayer *preLayer = m_vecLayers[connIdx];
	RecurrentLayer *postLayer = m_vecLayers[connIdx+1];
	RecurrentConnection *conn;
	if (connType == "full_connection") {
		conn = new RNNFullConnection(preLayer, postLayer);
	} else if (connType == "lstm_connection") {
		conn = new RNNLSTMConnection(preLayer, postLayer);
	} else {
		LOG(ERROR) << "Error in initConnection." << endl;
		exit(-1);
	}
	return conn;
}

RecurrentLayer *RNNLSTM::initLayer(int layerIdx) {
	string layerType = m_layerTypeList[layerIdx];
	int numNeuron = m_numNeuronList[layerIdx];
	DLOG_IF(INFO, DEBUG_RNNLSTM) << "layerType[" << layerIdx << "]:"<< layerType << ", numNeuron:" << numNeuron << endl;
	RecurrentLayer *layer;
	if (layerType == "input_layer") {
		layer = new RNNInputLayer(numNeuron, m_maxSeqLen);
	} else if (layerType == "lstm_layer") {
		int inputSize;
		if (layerIdx == 0) {
			inputSize = m_inputSize;
		} else {
			inputSize = m_numNeuronList[layerIdx-1];
		}
		layer = new RNNLSTMLayer(numNeuron, m_maxSeqLen, inputSize);
	} else if (layerType == "softmax_layer") {
		m_taskType = "classification";
		layer = new RNNSoftmaxLayer(numNeuron, m_maxSeqLen);
	} else if (layerType == "mse_layer") {
		m_taskType = "regression";
		layer = new RNNMSELayer(numNeuron, m_maxSeqLen);
	} else {
		LOG(ERROR) << "Error in initLayer." << endl;
		exit(-1);
	}
	return layer;
}

void RNNLSTM::initParams(float *params) {
	float *cursor = params;
	// layer part
	for (int layerIdx=0; layerIdx<m_numLayer; ++layerIdx) {
		m_vecLayers[layerIdx]->initParams(cursor);
		cursor += m_vecLayers[layerIdx]->m_paramSize;
	}
	// connection part
	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		m_vecConnections[connIdx]->initParams(cursor);
		cursor += m_vecConnections[connIdx]->m_paramSize;
	}
}

void RNNLSTM::feedForward(int inputSeqLen) {
	/* feed forward through connections and layers */
	m_vecLayers[0]->feedForward(inputSeqLen);
	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		m_vecConnections[connIdx]->feedForward(inputSeqLen);
		m_vecLayers[connIdx+1]->feedForward(inputSeqLen);
	}
}

void RNNLSTM::feedBackward(int inputSeqLen) {
	/* feed backward through connections and layers */
	m_vecLayers[m_numLayer-1]->feedBackward(inputSeqLen);
	for (int connIdx=m_numLayer-2; connIdx>=0; --connIdx) {
		m_vecConnections[connIdx]->feedBackward(inputSeqLen);
		m_vecLayers[connIdx]->feedBackward(inputSeqLen);
	}
}

void RNNLSTM::forwardStep(int seqIdx) {
	/* forwardStep through connections and layers */
	m_vecLayers[0]->forwardStep(seqIdx);
	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		m_vecConnections[connIdx]->forwardStep(seqIdx);
		m_vecLayers[connIdx+1]->forwardStep(seqIdx);
	}
}

void RNNLSTM::backwardStep(int seqIdx) {
	/* backwardStep through connections and layers */
	m_vecLayers[m_numLayer-1]->backwardStep(seqIdx);
	for (int connIdx=m_numLayer-2; connIdx>=0; --connIdx) {
		m_vecConnections[connIdx]->backwardStep(seqIdx);
		m_vecLayers[connIdx]->backwardStep(seqIdx);
	}
}

void RNNLSTM::setInput(float *input, int begIdx, int endIdx, int stride) {
	float *inputCursor = input;
	// bind input sequence to m_inputActs of the input layer
	RecurrentLayer *RNNInputLayer = m_vecLayers[0];
	for (int seqIdx=begIdx; seqIdx<=endIdx; seqIdx+=stride) {
		memcpy(RNNInputLayer->m_inputActs[seqIdx], inputCursor, sizeof(float)*m_inputSize);
		inputCursor += m_inputSize;
	}
}

void RNNLSTM::setTarget(float *target, int begIdx, int endIdx, int stride) {
	float *targetCursor = target;
	// bind target sequence to m_outputErrs of the output layer
	RecurrentLayer *outputLayer = m_vecLayers[m_numLayer-1];
	for (int seqIdx=begIdx; seqIdx<=endIdx; seqIdx+=stride) {
		memcpy(outputLayer->m_outputErrs[seqIdx], targetCursor, sizeof(float)*m_outputSize);
		targetCursor += m_outputSize;
	}
}

float RNNLSTM::computeError(int inputSeqLen) {
	float sampleError = 0.f;	
	RecurrentLayer *outputLayer = m_vecLayers[m_numLayer-1];
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		for (int i=0; i<m_outputSize; ++i) {
			if (m_taskType == "classification") {
				sampleError += outputLayer->m_outputErrs[seqIdx][i] * log(outputLayer->m_outputActs[seqIdx][i]);
			} else if (m_taskType == "regression") {
				float diff = outputLayer->m_outputErrs[seqIdx][i] - outputLayer->m_outputActs[seqIdx][i];
				sampleError += diff * diff;
			}
		}
	}
	return sampleError;
}

void RNNLSTM::getPredict(float *samplePredict, int inputSeqLen) {
	float *predictCursor = samplePredict;
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		memcpy(predictCursor, m_vecLayers[m_numLayer-1]->m_outputActs[seqIdx], sizeof(float) * m_outputSize);
		predictCursor += m_outputSize;
	}
}

float RNNLSTM::computeGrad(float *grad, float *params, float *data, float *target, int minibatchSize) {
	float error = 0.f;
	
	memset(grad, 0x00, sizeof(float)*m_paramSize);
	bindWeights(params);
	bindGrads(grad);
	
	float *sampleData = data;
	float *sampleTarget = target;

	/*** feed forward and feed backward ***/
	for (int dataIdx=0; dataIdx<minibatchSize; ++dataIdx) {
		// TODO
		int inputSeqLen = m_maxSeqLen;
		
		/* reset internal states of LSTM layers */
		resetStates(inputSeqLen); // this is subject to change
		
		/* feedforward */
		// set input
		for (int i=1; i<=inputSeqLen; ++i) {
			setInput(sampleData, i, i);
		}
		// feedForward through connections and layers
		feedForward(inputSeqLen);

		/* compute error */
		error += computeError(inputSeqLen);

		/* feedbackword */
		// set target
		for (int i=1; i<=inputSeqLen; ++i) {
			setTarget(sampleTarget, i, i);
		}
		// feedback through connections and layers
		feedBackward(inputSeqLen);

		sampleData += m_inputSize;
		sampleTarget += m_outputSize;
	}

	// normalization by number of input sequences and clip gradients to [-1, 1]
	float normFactor = 1.f / (float) minibatchSize;
	#pragma omp parallel for
	for (int dim=0; dim<m_paramSize; ++dim) {
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

void RNNLSTM::resetStates(int inputSeqLen) {
	for (int layerIdx=0; layerIdx<m_numLayer; ++layerIdx) {
		m_vecLayers[layerIdx]->resetStates(inputSeqLen);
	}
}

// Sequential Part
void RNNLSTM::bindWeights(float *params) {
	// define cursors
	float *paramsCursor = params;
	// layer part
	for (int layerIdx=0; layerIdx<m_numLayer; ++layerIdx) {
		m_vecLayers[layerIdx]->bindWeights(paramsCursor);
		paramsCursor += m_vecLayers[layerIdx]->m_paramSize;		
	}
	// connection part
	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		m_vecConnections[connIdx]->bindWeights(paramsCursor);
		paramsCursor += m_vecConnections[connIdx]->m_paramSize;
	}
}

// Sequential Part
void RNNLSTM::bindGrads(float *grad) {
	// define cursors
	float *gradCursor = grad;
	// layer part
	for (int layerIdx=0; layerIdx<m_numLayer; ++layerIdx) {
		m_vecLayers[layerIdx]->bindGrads(gradCursor);
		gradCursor += m_vecLayers[layerIdx]->m_paramSize;
	}
	// connection part
	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		m_vecConnections[connIdx]->bindGrads(gradCursor);
		gradCursor += m_vecConnections[connIdx]->m_paramSize;
	}
}