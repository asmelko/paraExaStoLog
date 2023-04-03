#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using index_t = int;
using real_t = float;
using d_idxvec = thrust::device_vector<index_t>;
using d_datvec = thrust::device_vector<real_t>;

using h_idxvec = thrust::host_vector<index_t>;
using h_datvec = thrust::host_vector<real_t>;
