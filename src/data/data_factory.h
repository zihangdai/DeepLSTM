#ifndef __DATA_FACTORY_H__
#define __DATA_FACTORY_H__

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

class DataFactory
{
    public:
		DataFactory() {};
		virtual ~DataFactory() {};
		
		virtual int getNumberOfData() {return 0;};		
		virtual int getDataSize() {return 0;};
		virtual int getLabelSize() {return 0;};

		virtual void printOutData() {};
		virtual void getDataBatch(float*, float*, int*, int) {};
    
    protected:
		int m_numSample;
		virtual float getDataByIndex(int,int) {return 0.0;};
};

#endif
