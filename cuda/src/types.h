#pragma once

#include <thrust/device_vector.h>

using index_t = int;
using d_idxvec = thrust::device_vector<index_t>;

using real_t = float;
using d_datvec = thrust::device_vector<real_t>;
