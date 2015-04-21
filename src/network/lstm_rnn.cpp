#include <sstream> 
#include "lstm_rnn.h"

using namespace std;

#define DEBUG_LSTM_RNN 0

LSTM_RNN::LSTM_RNN(boost::property_tree::ptree *confReader, string section) {
	/* read conf and allocate memory */	
	m_numLayer = confReader->get<int>(section + "num_layer");
	m_numNeuronList = new int[m_numLayer];
	m_layerTypeList = new string[m_numLayer];
	m_connTypeList = new string[m_numLayer-1];
	DLOG_IF(INFO, DEBUG_LSTM_RNN) << "Finish reading conf and allocating memory." << endl;

	// type and number of neurons of each layer
	for (int layerIdx=0; layerIdx<m_numLayer; ++layerIdx) {
		stringstream ss;
  		ss << layerIdx;
		m_numNeuronList[layerIdx] = confReader->get<int>(section + "num_neuron_layer_" + ss.str());
		m_layerTypeList[layerIdx] = confReader->get<string>(section + "type_layer_" + ss.str());
	}
	DLOG_IF(INFO, DEBUG_LSTM_RNN) << "Finish reading type and number of neurons of each layer." << endl;
	
	// type of each conectection
	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		stringstream ss;
  		ss << connIdx;
		m_connTypeList[connIdx] = confReader->get<string>(section + "type_connection_" + ss.str());
	}
	DLOG_IF(INFO, DEBUG_LSTM_RNN) << "Finish reading type of each conectection." << endl;	
	
	m_nParamSize = 0;
	m_maxSeqLen = confReader->get<int>(section + "max_sequence_length");
	DLOG_IF(INFO, DEBUG_LSTM_RNN) << "Finish reading max_sequence_length." << endl;	

	/* initialize layers */
	for (int layerIdx=0; layerIdx<m_numLayer; layerIdx++) {
		RecurrentLayer *layer = initLayer(layerIdx);
		m_nParamSize += layer->m_nParamSize;
		m_vecLayers.push_back(layer);
	}
	m_dataSize = m_vecLayers[0]->m_inputSize;
	m_targetSize = m_vecLayers[m_numLayer-1]->m_numNeuron;
	DLOG_IF(INFO, DEBUG_LSTM_RNN) << "inish initializine layers." << endl;	

	/* initialize connections */
	for (int connIdx=0; connIdx<m_numLayer-1; connIdx++) {
		RecConnection *conn = initConnection(connIdx);
		m_nParamSize += conn->m_nParamSize;
		m_vecConnections.push_back(conn);
	}
	DLOG_IF(INFO, DEBUG_LSTM_RNN) << "LSTM_RNN Constructor finished." << endl;
}

LSTM_RNN::~LSTM_RNN() {
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
	int numNeuron = m_numNeuronList[layerIdx];
	#ifdef DEBUG_LSTM_RNN
	printf("layerType[%d]:%s, numNeuron:%d\n",layerIdx,layerType.c_str(),numNeuron);
	#endif
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
		#ifdef DEBUG_LSTM_RNN
		printf("m_vecLayers[%d] init parameters.\n", layerIdx);
		#endif
		m_vecLayers[layerIdx]->initParams(cursor);
		cursor += m_vecLayers[layerIdx]->m_nParamSize;
	}
	// connection part
	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		#ifdef DEBUG_LSTM_RNN
		printf("m_vecConnections[%d] init parameters.\n", connIdx);
		#endif
		m_vecConnections[connIdx]->initParams(cursor);
		cursor += m_vecConnections[connIdx]->m_nParamSize;
	}
}

void LSTM_RNN::feedForward(int inputSeqLen) {	
	/* feed forward through connections and layers */
	m_vecLayers[0]->feedForward(inputSeqLen);
	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		m_vecConnections[connIdx]->feedForward(inputSeqLen);
		m_vecLayers[connIdx+1]->feedForward(inputSeqLen);
	}
}

void LSTM_RNN::feedBackward(int inputSeqLen) {
	/* feed backward through connections and layers */
	m_vecLayers[m_numLayer-1]->feedBackward(inputSeqLen);
	for (int connIdx=m_numLayer-2; connIdx>=0; --connIdx) {
		m_vecConnections[connIdx]->feedBackward(inputSeqLen);
		m_vecLayers[connIdx]->feedBackward(inputSeqLen);
	}
}

float LSTM_RNN::computeError(float *sampleTarget, int inputSeqLen) {
	float sampleError = 0.f;
	float *targetCursor = sampleTarget;
	RecurrentLayer *curLayer = m_vecLayers[m_numLayer-1];
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		for (int i=0; i<m_targetSize; ++i) {
			if (m_errorType == "cross_entropy_error") {
				sampleError += targetCursor[i] * log(curLayer->m_outputActs[seqIdx][i]);
			} else if (m_errorType == "mean_squared_error") {
				float diff = targetCursor[i] - curLayer->m_outputActs[seqIdx][i];
				sampleError += diff * diff;
			}
		}
		targetCursor += m_targetSize;
	}
	return sampleError;
}

float LSTM_RNN::computeGrad(float *grad, float *params, float *data, float *target, int minibatchSize) {
	float error = 0.f;
	
	memset(grad, 0x00, sizeof(float)*m_nParamSize);
	bindWeights(params, grad);
	
	float *sampleData = data;
	float *sampleTarget = target;

	/*** feed forward and feed backward ***/
	for (int dataIdx=0; dataIdx<minibatchSize; ++dataIdx) {
		// TODO
		int inputSeqLen = m_maxSeqLen;
		
		/* reset internal states of LSTM layers */
		resetStates(inputSeqLen); // this is subject to change
		
		/* feedforward */
		float *dataCursor = sampleData;
		// bind input sequence to m_inputActs of the input layer 
		RecurrentLayer *inputLayer = m_vecLayers[0];
		for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
			memcpy(inputLayer->m_inputActs[seqIdx], dataCursor, sizeof(float)*m_dataSize);
			dataCursor += m_dataSize;
		}
		// feedForward through connections and layers
		feedForward(inputSeqLen);

		/* compute error */
		error += computeError(sampleTarget, inputSeqLen);

		/* feedbackword */
		float *targetCursor = sampleTarget;
		// bind target sequence to m_outputErrs of the output layer
		RecurrentLayer *outputLayer = m_vecLayers[m_numLayer-1];
		for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
			memcpy(outputLayer->m_outputErrs[seqIdx], targetCursor, sizeof(float)*m_targetSize);
			targetCursor += m_targetSize;
		}
		// feedback through connections and layers
		feedBackward(inputSeqLen);		

		sampleData += m_dataSize * inputSeqLen;
		sampleTarget += m_targetSize * inputSeqLen;
	}

	// normalization by number of input sequences and clip gradients to [-1, 1]
	// float normFactor = 1.f / (float) minibatchSize;
	// for (int dim=0; dim<m_nParamSize; ++dim) {
	// 	grad[dim] *= normFactor;
	// 	if (grad[dim] < -1.f) {
	// 		grad[dim] = -1.f;
	// 	} else if (grad[dim] > 1.f) {
	// 		grad[dim] = 1.f;
	// 	}
	// }
	// error *= normFactor;

	return error;
}

void LSTM_RNN::resetStates(int inputSeqLen) {	
	for (int layerIdx=0; layerIdx<m_numLayer; ++layerIdx) {
		m_vecLayers[layerIdx]->resetStates(inputSeqLen);
	}
}

// Sequential Part
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