#ifndef __MASTER_H__
#define __MASTER_H__

#include <mpi.h>

#include "common.h"
#include "sgd.h"

#define ROOT 0

#define WORKTAG 1
#define STOPTAG 2

using namespace std;

void masterFunc ();

sgdBase * initSgdSolver (boost::property_tree::ptree *confReader, string section, int paramSize);

#endif