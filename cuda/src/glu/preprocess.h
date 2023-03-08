#ifndef __PREPROCESS__
#define __PREPROCESS__

#include "nicslu/nics_config.h"
#include "nicslu/nicslu.h"
#include "type.h"

extern "C"
{
	int preprocess(SNicsLU* nicslu, uint__t n, uint__t nnz, double* ax, unsigned int* ai, unsigned int* ap);
}

#endif
