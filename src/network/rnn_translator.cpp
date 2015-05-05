#include <cmath>
#include <iostream>
#include "rnn_translator.h"

// #define DEBUG_RNN_TRANSLATOR

using namespace std;

RNNTranslator::RNNTranslator(boost::property_tree::ptree *confReader, string section) {
	// init encoder and decoder
	m_reverseEncoder = confReader->get<int>(section + "reverse_encoder");

	m_encoder = new RNNLSTM(confReader, "Encoder.");
	m_decoder = new RNNLSTM(confReader, "Decoder.");
	
	// compute paramSize
	m_paramSize = 0;
	m_paramSize += m_encoder->m_paramSize;
	m_paramSize += m_decoder->m_paramSize;
}

RNNTranslator::~RNNTranslator() {
	if (m_encoder != NULL) {
		delete m_encoder;
	}
	if (m_decoder != NULL) {
		delete m_decoder;
	}
}

void RNNTranslator::initParams (float *params) {
	float *paramsCursor = params;
	
	m_encoder->initParams(paramsCursor);
	
	paramsCursor += m_encoder->m_paramSize;

	m_decoder->initParams(paramsCursor);
}

void RNNTranslator::bindWeights(float *params) {
	float *paramsCursor = params;	
	m_encoder->bindWeights(paramsCursor);
	
	paramsCursor += m_encoder->m_paramSize;
	m_decoder->bindWeights(paramsCursor);
}

void RNNTranslator::bindGrads(float *grad) {
	float *gradCursor = grad;	
	m_encoder->bindGrads(gradCursor);
	
	gradCursor += m_encoder->m_paramSize;
	m_decoder->bindGrads(gradCursor);
}

float RNNTranslator::translate (float *params, float *input, float *predict, int batchSize) {
	float error = 0.f;
	bindWeights(params);

	float *sampleInput = input;
	float *samplePredict = predict;

	RecurrentLayer *enInputLayer  = m_encoder->m_vecLayers[0];
	RecurrentLayer *enOutputLayer = m_encoder->m_vecLayers[m_encoder->m_numLayer-1];

	RecurrentLayer *deInputLayer  = m_decoder->m_vecLayers[0];
	RecurrentLayer *deOutputLayer = m_decoder->m_vecLayers[m_decoder->m_numLayer-1];

	/*** feed forward and feed backward ***/
	for (int sampleIdx=0; sampleIdx<batchSize; ++sampleIdx) {
		// TODO
		int encoderSeqLen = m_encoder->m_maxSeqLen;
		int decoderSeqLen = m_decoder->m_maxSeqLen;

		/* reset internal states of LSTM layers */
		m_encoder->resetStates(encoderSeqLen);
		m_decoder->resetStates(decoderSeqLen);
		
		/****************************************************************
		*                      Feed Forward Phase                       *
		****************************************************************/
		// ********* encoder *********
		
		// step 1: bind input sequence to m_inputActs of the input layer of the encoder
		if (m_reverseEncoder) {
			m_encoder->setInput(sampleInput, encoderSeqLen, 1, -1);
		} else {
			m_encoder->setInput(sampleInput, 1, encoderSeqLen);
		}
		
		// step 2: encoder feedforward
		m_encoder->feedForward(encoderSeqLen);

		// ********* decoder *********
		// step 1: set the internal states of the decoder at t' = 0 to the internal states of encoder at t = T
		#pragma omp parallel for
		for (int layerIdx=0; layerIdx<m_encoder->m_numLayer; layerIdx++) {
			RNNLSTMLayer *enLayer = dynamic_cast<RNNLSTMLayer*>(m_encoder->m_vecLayers[layerIdx]);
			RNNLSTMLayer *deLayer = dynamic_cast<RNNLSTMLayer*>(m_decoder->m_vecLayers[layerIdx]);
			memcpy(deLayer->m_states[0], enLayer->m_states[encoderSeqLen], sizeof(float) * deLayer->m_numNeuron);
			memcpy(deLayer->m_outputActs[0], enLayer->m_outputActs[encoderSeqLen], sizeof(float) * deLayer->m_numNeuron);
		}
		
		// step 2: predict
		// for seqIdx = 1
		m_decoder->forwardStep(1);
		// for seqIdx = 2 to T'		
		for (int seqIdx=2; seqIdx<=decoderSeqLen; ++seqIdx) {
			// set input for t = seqIdx
			if (m_decoder->m_taskType == "classification") {
				int predictClass = argmax(deOutputLayer->m_outputActs[seqIdx-1], m_decoder->m_outputSize);
				m_decoder->setInput(sampleInput+m_decoder->m_inputSize*predictClass, seqIdx, seqIdx);
			} else if (m_decoder->m_taskType == "regression") {
				m_decoder->setInput(deOutputLayer->m_outputActs[seqIdx-1], seqIdx, seqIdx);
			}
			// forward step for t = seqIdx
			m_decoder->forwardStep(seqIdx);
		}				
		
		/****************************************************************
		*                      Get Predicted Values                     *
		****************************************************************/
		m_decoder->getPredict(samplePredict, decoderSeqLen);
				
		sampleInput   += encoderSeqLen * m_encoder->m_inputSize;
		samplePredict += decoderSeqLen * m_decoder->m_outputSize;
	}

	return error;
}

float RNNTranslator::computeGrad (float *grad, float *params, float *input, float *target, int minibatchSize) {
	
	float error = 0.f;
	memset(grad, 0x00, sizeof(float)*m_paramSize);
	
	bindWeights(params);
	bindGrads(grad);

	float *sampleInput = input;
	float *sampleTarget = target;

	RecurrentLayer *enInputLayer  = m_encoder->m_vecLayers[0];
	RecurrentLayer *enOutputLayer = m_encoder->m_vecLayers[m_encoder->m_numLayer-1];

	RecurrentLayer *deInputLayer  = m_decoder->m_vecLayers[0];
	RecurrentLayer *deOutputLayer = m_decoder->m_vecLayers[m_decoder->m_numLayer-1];

	/*** feed forward and feed backward ***/
	for (int sampleIdx=0; sampleIdx<minibatchSize; ++sampleIdx) {
		// TODO
		int encoderSeqLen = m_encoder->m_maxSeqLen;
		int decoderSeqLen = m_decoder->m_maxSeqLen;

		/* reset internal states of LSTM layers */
		m_encoder->resetStates(encoderSeqLen);
		m_decoder->resetStates(decoderSeqLen);
		
		/****************************************************************
		*                      Feed Forward Phase                       *
		****************************************************************/
		// ********* encoder *********
		
		// step 1: bind input sequence to m_inputActs of the input layer of the encoder
		if (m_reverseEncoder) {
			m_encoder->setInput(sampleInput, encoderSeqLen, 1, -1);
		} else {
			m_encoder->setInput(sampleInput, 1, encoderSeqLen);
		}
		
		// step 2: encoder feedforward
		m_encoder->feedForward(encoderSeqLen);

		// ********* decoder *********
		// step 1: set the internal states of the decoder at t' = 0 to the internal states of encoder at t = T
		#pragma omp parallel for
		for (int layerIdx=0; layerIdx<m_encoder->m_numLayer; layerIdx++) {
			RNNLSTMLayer *enLayer = dynamic_cast<RNNLSTMLayer*>(m_encoder->m_vecLayers[layerIdx]);
			RNNLSTMLayer *deLayer = dynamic_cast<RNNLSTMLayer*>(m_decoder->m_vecLayers[layerIdx]);
			memcpy(deLayer->m_states[0], enLayer->m_states[encoderSeqLen], sizeof(float) * deLayer->m_numNeuron);
			memcpy(deLayer->m_outputActs[0], enLayer->m_outputActs[encoderSeqLen], sizeof(float) * deLayer->m_numNeuron);
		}
		
		// step 2: set the input for the decoder
		// for seqIdx = 1, the input will be empty, which is similar to the END-OF-SENTENCE in Ilya's work
		// for seqIdx = 2 to T', bind input sequence to m_inputActs of the input layer of the decoder
		// note at time step t, the input vector should be the target vector at time step t-1
		if (m_decoder->m_taskType == "classification") {
			#pragma omp parallel for
			for (int seqIdx=2; seqIdx<=decoderSeqLen; ++seqIdx) {
				float *targetCursor = sampleTarget + (seqIdx-2) * m_decoder->m_outputSize;
				int targetClass = argmax (targetCursor, m_decoder->m_outputSize);
				m_decoder->setInput(sampleInput+m_decoder->m_inputSize*targetClass, seqIdx, seqIdx);
			}
		} else if (m_decoder->m_taskType == "regression") {
			m_decoder->setInput(sampleTarget, 2, decoderSeqLen);
		}
		
		// step 3: decoder feedforward
		m_decoder->feedForward(decoderSeqLen);
		
		/****************************************************************
		*                      Compute Error Phase                      *
		****************************************************************/
		error += m_decoder->computeError(sampleTarget, decoderSeqLen);
		
		/****************************************************************
		*                      Feed Backword Phase                      *
		****************************************************************/
		// ********* decoder *********
		
		// step 1: bind target sequence to m_outputErrs of the output layer of the decoder
		m_decoder->setTarget(sampleTarget, 1, decoderSeqLen);
		
		// step 2: decoder feed backward
		m_decoder->feedBackward(decoderSeqLen);

		// ********* encoder *********
		// step 1: set the internal deltas of the encoder at t = T+1 to the internal deltas of encoder at t' = 1
		#pragma omp parallel for
		for (int layerIdx=0; layerIdx<m_encoder->m_numLayer; layerIdx++) {
			RNNLSTMLayer *enLayer = dynamic_cast<RNNLSTMLayer*>(m_encoder->m_vecLayers[layerIdx]);
			RNNLSTMLayer *deLayer = dynamic_cast<RNNLSTMLayer*>(m_decoder->m_vecLayers[layerIdx]);
			memcpy(enLayer->m_cellStateErrs[encoderSeqLen+1], deLayer->m_cellStateErrs[1], sizeof(float) * deLayer->m_numNeuron);
			memcpy(enLayer->m_inGateDelta[encoderSeqLen+1], deLayer->m_inGateDelta[1], sizeof(float) * deLayer->m_numNeuron);
			memcpy(enLayer->m_forgetGateDelta[encoderSeqLen+1], deLayer->m_forgetGateDelta[1], sizeof(float) * deLayer->m_numNeuron);
			memcpy(enLayer->m_outGateDelta[encoderSeqLen+1], deLayer->m_outGateDelta[1], sizeof(float) * deLayer->m_numNeuron);
			memcpy(enLayer->m_preGateStateDelta[encoderSeqLen+1], deLayer->m_preGateStateDelta[1], sizeof(float) * deLayer->m_numNeuron);
			memcpy(enLayer->m_forgetGateActs[encoderSeqLen+1], deLayer->m_forgetGateActs[1], sizeof(float) * deLayer->m_numNeuron);
		}
		
		// step 2: encoder feed backward based on deltas at t = T+1
		m_encoder->feedBackward(encoderSeqLen);

		/****************************************************************
		*                   move cursor to new position                 *
		****************************************************************/
		sampleInput += encoderSeqLen * m_encoder->m_inputSize;
		sampleTarget += decoderSeqLen * m_decoder->m_outputSize;
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