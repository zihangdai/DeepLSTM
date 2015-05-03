#include "sgd.h"

sgdBase::sgdBase(boost::property_tree::ptree *confReader, string section, int paramSize) {
	m_paramSize = paramSize;
	m_stepCount = 0;
	m_useMomentum = confReader->get<int>(section + "use_momentum");
	if (m_useMomentum) {
		m_momentumFactor = confReader->get<float>(section + "momentum_factor");		
	} else {
		m_momentumFactor = 0.f;
	}
	m_velocity = new float [m_paramSize];
	memset(m_velocity, 0x00, sizeof(float)*m_paramSize);
}

sgdBase::~sgdBase() {
	if(m_velocity != NULL) {
		delete [] m_velocity;
	}
}

sgdBasic::sgdBasic (boost::property_tree::ptree *confReader, string section, int paramSize) : sgdBase(confReader, section, paramSize) {
	m_learningRate = confReader->get<float>(section + "learning rate");
}

sgdBasic::~sgdBasic () {
	// nothing to do for basic sgd
}

void sgdBasic::updateParams (float *params, float *grad, int rank) {
	for (int i=0; i<m_paramSize; i++) {
		params[i] -= m_learningRate * grad[i];
	}
}