#include "sgd.h"

sgdBase::sgdBase(ConfReader *confReader, int paramSize) {
	m_nParamSize = paramSize;
	m_stepCount = 0;
	m_useMomentum = confReader->getInt("use_momentum");
	if (m_useMomentum) {
		m_momentumFactor = confReader->getFloat("momentum_factor");		
	} else {
		m_momentumFactor = 0.f;
	}
	m_velocity = new float [m_nParamSize];
	memset(m_velocity, 0x00, sizeof(float)*m_nParamSize);
}

sgdBase::~sgdBase() {
	if(m_velocity != NULL) {
		delete [] m_velocity;
	}
}

sgdBasic::sgdBasic (ConfReader *confReader, int paramSize) : sgdBase(confReader, paramSize) {
	m_learningRate = confReader->getFloat("learning rate");
}

sgdBasic::~sgdBasic () {
	// nothing to do for basic sgd
}

void sgdBasic::updateParams (float *params, float *grad, int rank) {
	for (int i=0; i<m_nParamSize; i++) {
		params[i] -= m_learningRate * grad[i];
	}
}