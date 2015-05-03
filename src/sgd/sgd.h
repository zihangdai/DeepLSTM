#ifndef __SGD_H__
#define __SGD_H__

#include <map>

#include "matrix.h"
#include "common.h"

using namespace std;

class sgdBase
{
public:
    sgdBase(boost::property_tree::ptree *confReader, string section, int paramSize);
    virtual ~sgdBase();

    /* data */

    /* method */
    void virtual updateParams (float *params, float *grad, int rank=0) {};

protected:
    /* data */
    int m_useMomentum;
    int m_paramSize;    
    float m_learningRate;
    float m_momentumFactor;

    float *m_velocity;

    int m_stepCount;

    /* method */
    //TODO void truncate (float);
    void printInfo (float *buffer) {
        float sum = 0.f;
        for (int i=0; i<m_paramSize; ++i) {
            sum += buffer[i];
        }
        printf("sum: %f\n", sum);
    }
};

/****************************************************************
* BASIC SGD
****************************************************************/
class sgdBasic: public sgdBase
{
public:
    sgdBasic(boost::property_tree::ptree *confReader, string section, int paramSize);
    ~sgdBasic();

    /* data */

    /* method */
    void updateParams (float *params, float *grad, int rank=0);
};

/****************************************************************
* ADAGRAD
****************************************************************/
class adagrad: public sgdBase
{
public:
    adagrad(boost::property_tree::ptree *confReader, string section, int paramSize);
    ~adagrad();

    /* data */
    int m_residual;
    int m_stopSIMD;

    /* method */
    void updateParams (float *params, float *grad, int rank=0);

private:
    /* data */
    float *m_histSquareGrad;
};

/****************************************************************
* ADADELTA
****************************************************************/
class adadelta: public sgdBase
{
public:
    adadelta(boost::property_tree::ptree *confReader, string section, int paramSize);
    ~adadelta();

    /* data */

    /* method */
    void updateParams (float *params, float *grad, int rank=0);

private:
    /* data */
    float m_decayFactor;
    float m_stableConst;

    float *m_ESquareGrad;
    float *m_ESquareDelta;    
};

/****************************************************************
* RMSPROP
****************************************************************/
class rmsprop: public sgdBase
{
public:
    rmsprop(boost::property_tree::ptree *confReader, string section, int paramSize);
    ~rmsprop();

    /* data */

    /* method */
    void updateParams (float *params, float *grad, int rank=0);

private:
    /* data */
    float m_decayFactor;

    float *m_meanSquareGrad;    
};

#endif