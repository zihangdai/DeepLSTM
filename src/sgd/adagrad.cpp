#include <math.h>
#include <string.h>
#include <omp.h>
#include "sgd.h"

using namespace std;

#define DEBUG_ADAGRAD 1

adagrad::adagrad (boost::property_tree::ptree *confReader, string section, int paramSize) : sgdBase(confReader, section, paramSize) {
	m_learningRate = confReader->get<float>(section + "learning_rate");

	m_histSquareGrad = new float [m_nParamSize];
	for (int i=0; i<m_nParamSize; i++) {
		m_histSquareGrad[i] = 0.1f;
	}

	m_residual = m_nParamSize % SIMD_WIDTH;
	m_stopSIMD = m_nParamSize - m_residual;

}

adagrad::~adagrad () {
	if (m_histSquareGrad != NULL) {
		delete [] m_histSquareGrad;
	}
}

void adagrad::updateParams (float *params, float *grad, int rank) {
	m_stepCount += 1;

	if (!SIMD) {
		#pragma omp parallel for
		for (int i=0; i<m_nParamSize; i++) {
			m_histSquareGrad[i] += grad[i] * grad[i];
			m_velocity[i] = m_momentumFactor * m_velocity[i] - m_learningRate * grad[i] / sqrt(m_histSquareGrad[i]);
			params[i] += m_velocity[i];
		}
	} else {
		#ifdef linux
		__m256 vecLearnRate = _mm256_set1_ps(m_learningRate);
		__m256 vecMomentum  = _mm256_set1_ps(m_momentumFactor);
		
		#pragma omp parallel for
		for (int i=0; i<m_stopSIMD; i+=SIMD_WIDTH) {
			__m256 vec_grad, vec_vel, vec_param, vec_hist;
			vec_grad = _mm256_loadu_ps(grad + i);
			vec_param = _mm256_loadu_ps(params + i);
			vec_vel = _mm256_loadu_ps(m_velocity + i);
			vec_hist = _mm256_loadu_ps(m_histSquareGrad + i);

			// m_histSquareGrad[i] += grad[i] * grad[i];
			vec_hist = _mm256_add_ps(vec_hist, _mm256_mul_ps(vec_grad, vec_grad));
			// m_velocity[i] = m_momentumFactor * m_velocity[i] - m_learningRate * grad[i] / sqrt(m_histSquareGrad[i]);
			vec_vel = _mm256_sub_ps(_mm256_mul_ps(vecMomentum, vec_vel), _mm256_div_ps(_mm256_mul_ps(vecLearnRate, vec_grad), _mm256_sqrt_ps(vec_hist)));
			// params[i] += m_velocity[i];
			vec_param = _mm256_add_ps(vec_param, vec_vel);

			_mm256_storeu_ps(params + i, vec_param);
			_mm256_storeu_ps(m_velocity + i, vec_vel);
			_mm256_storeu_ps(m_histSquareGrad + i, vec_hist);
		}

		for (int i=m_stopSIMD; i<m_nParamSize; i++) {
			m_histSquareGrad[i] += grad[i] * grad[i];
			m_velocity[i] = m_momentumFactor * m_velocity[i] - m_learningRate * grad[i] / sqrt(m_histSquareGrad[i]);
			params[i] += m_velocity[i];
		}
		#endif
	}
}