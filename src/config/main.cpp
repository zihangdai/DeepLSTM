#include <stdio.h>
#include "confreader.h"

int main () {
	ConfReader *confReader = new ConfReader("config.conf", "Master");
	int i = confReader->getInt("number of Slave");
	std::string j = confReader->getString("foo");
	printf("%d,%s\n",i,j.c_str());
	return 0;
}