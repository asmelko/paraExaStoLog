#pragma once

#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename T>
void print(const char* msg, const thrust::device_vector<T>& v)
{
	thrust::host_vector<T> h = v;

	std::cout << msg;
	for (auto t : h)
		std::cout << t << " ";
	std::cout << std::endl;
}
