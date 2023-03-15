#ifndef WCC_H
#define WCC_H
#include "scc.h"
#include "bitset.h"
#include "cutil_subset.h"
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

bool find_wcc(int m, const int *d_row_offsets, const int *d_column_indices, unsigned *d_colors, unsigned char *d_status, int *scc_root, unsigned min_color);

#endif
