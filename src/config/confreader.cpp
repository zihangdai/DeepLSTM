#include <stdio.h>
#include "confreader.h"

ConfReader::ConfReader (std::string filename, std::string confKey) {
	printf("ConfReader Constructor: %s, %s\n", filename.c_str(), confKey.c_str());
	m_configFile = new ConfigFile(filename);
	m_confKey = confKey;
	printf("ConfReader Constructor finished.\n");
}

ConfReader::~ConfReader () {
	if (!m_configFile) {
		delete m_configFile;
	}
}

int ConfReader::getInt (std::string key) {
	int val = m_configFile->Value(m_confKey, key);
	return val;
}

float ConfReader::getFloat(std::string key)
{
   	float val = m_configFile->Value(m_confKey, key);
	return val;
}

std::string ConfReader::getString(std::string key)
{
   std::string val = m_configFile->Value(m_confKey, key);
	return val;
}
