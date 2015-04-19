#ifndef __SGD_H__
#define __SGD_H__

#include <stdio.h>
#include <map>
#include "confreader.h"

class sgdBase
{
public:
    sgdBase() {};
    virtual ~sgdBase() {};

    /* data */

    /* method */
    void virtual updateParams (float *params, float *grad, int rank) {};

protected:
    /* data */
    int m_useMomentum;
    int m_nParamSize;
    float m_learningRate;
    int m_stepCount;

    /* method */
    //TODO void truncate (float);
    void printInfo (float *buffer) {
        float sum = 0.f;
        for (int i=0; i<m_nParamSize; ++i) {
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
    sgdBasic(ConfReader *confReader, int paramSize);
    ~sgdBasic();

    /* data */

    /* method */
    void updateParams (float *params, float *grad, int rank);
};

/****************************************************************
* ADAGRAD
****************************************************************/
class adagrad: public sgdBase
{
public:
    adagrad(ConfReader *confReader, int paramSize);
    ~adagrad();

    /* data */

    /* method */
    void updateParams (float *params, float *grad, int rank);

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
    adadelta(ConfReader *confReader, int paramSize);
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
    rmsprop(ConfReader *confReader, int paramSize);
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