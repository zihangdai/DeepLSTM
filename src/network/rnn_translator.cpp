#include "rnn_translator.h"

RNNTranslator::RNNTranslator(ConfReader *confReader) {
	// init encoder and decoder
	string encoderConfSec = confReader->getString("encoder_conf");
	string decoderConfSec = confReader->getString("decoder_conf");

	confReader *encoderConf = new confReader("config.conf", encoderConfSec);
	confReader *decoderConf = new confReader("config.conf", decoderConfSec);

	m_encoder = new lstm(encoderConf);
	m_decoder = new lstm(decoderConf);

	// init m_encodingW and m_decodingW
}

RNNTranslator::~RNNTranslator() {

}