#ifndef __CONFREADER_H__
#define __CONFREADER_H__

#include <string>
#include "configfile.h"

class ConfReader
{
public:
	ConfReader(std::string, std::string);
	~ConfReader();

	int getInt(std::string);
	std::string getString(std::string);
	float getFloat(std::string);

	/* data */
	ConfigFile *m_configFile;
	std::string m_confKey;
};

#endif
