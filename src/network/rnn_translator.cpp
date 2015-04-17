#include "rnn_translator.h"

RNNTranslator::RNNTranslator(ConfReader *confReader) {		
	// init encoder and decoder
	string encoderConfSec = confReader->getString("encoder_conf");
	string decoderConfSec = confReader->getString("decoder_conf");

	confReader *encoderConf = new confReader("config.conf", encoderConfSec);
	confReader *decoderConf = new confReader("config.conf", decoderConfSec);

	m_encoder = new lstm(encoderConf);	
	m_decoder = new lstm(decoderConf);
	
	// compute paramSize
	m_nParamSize = 0;
	m_nParamSize += m_encoder->m_nParamSize;
	m_nParamSize += m_decoder->m_nParamSize;
	m_nParamSize += m_encoder->m_targetSize * m_decoder->m_dataSize;
}

RNNTranslator::~RNNTranslator() {
	if (!m_encoder) {
		delete m_encoder;
	}
	if (!m_decoder) {
		delete m_decoder;
	}	
}

void RNNTranslator::initParams (float *params) {
	float *paramsCursor = params;
	
	m_encoder->initParams(paramsCursor);
	paramsCursor += m_encoder->m_nParamSize;

	m_decoder->initParams(paramsCursor);
	paramsCursor += m_decoder->m_nParamSize;

	float multiplier = 0.08;
	for (int i=0; i<m_encoder->m_targetSize * m_decoder->m_dataSize; i++) {
		paramsCursor[i] = multiplier * SYM_UNIFORM_RAND;
	}
}

void RNNTranslator::bindWeights(float *params, float *grad) {
	float *paramsCursor = params;
	float *gradCursor = grad;
	
	m_encoder->bindWeights(paramsCursor, gradCursor);
	paramsCursor += m_encoder->m_nParamSize;
	gradCursor += m_encoder->m_nParamSize;

	m_decoder->bindWeights(paramsCursor, gradCursor);
	paramsCursor += m_decoder->m_nParamSize;
	gradCursor += m_decoder->m_nParamSize;

	m_encodingW = paramsCursor;
	m_gradEncodingW = gradCursor;
}

float RNNTranslator::computeGrad (float *grad, float *params, float *data, float *label, int minibatchSize) {

}

