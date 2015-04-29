#ifndef __COMMON_HEADER_H__
#define __COMMON_HEADER_H__

/****************************************************************
* NOTE: This file is created to collect all commonly used headers
* and macro definitions
****************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <assert.h>

#include <immintrin.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include <omp.h>

#include <glog/logging.h>

#include "cycle_timer.h"

#define SIMD_WIDTH 8
#define SIMD 1
#define BLOCK_SIZE 64

#define SYM_UNIFORM_RAND (2 * ((float) rand() / (RAND_MAX)) - 1)   // rand float in [-1, 1]

#endif