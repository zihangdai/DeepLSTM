#include <math.h>
#include <string.h>
#include "sgd.h"

rmsprop::rmsprop (boost::property_tree::ptree *confReader, string section, int paramSize) : sgdBase(confReader, section, paramSize){	
	m_decayFactor = confReader->get<float>(section + "rmsprop_decay_factor");
	m_useMomentum  = confReader->get<int>(section + "use_momentum");

	m_meanSquareGrad  = new float [m_paramSize];

	memset(m_meanSquareGrad, 0x00, sizeof(float) * m_paramSize);
}

rmsprop::~rmsprop () {
	if (m_meanSquareGrad != NULL) {
		delete [] m_meanSquareGrad;
	}
}

void rmsprop::updateParams (float *params, float *grad, int rank) {
	float delta;
	for (int i=0; i<m_paramSize; i++) {
		// accumulate mean squared grad
		m_meanSquareGrad[i] = m_decayFactor * m_meanSquareGrad[i] + (1 - m_decayFactor) * grad[i] * grad[i];
		// compute delta
		params[i] -= grad[i] / sqrt(m_meanSquareGrad[i]);
	}
}