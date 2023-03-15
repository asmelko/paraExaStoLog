#pragma once

#include <iostream>
#include <string_view>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename T>
void print(std::string_view msg, const thrust::device_vector<T>& v, size_t count = 0)
{
	thrust::host_vector<T> h = v;

	std::cout << msg;
	auto end = count == 0 ? h.end() : h.begin() + std::min(count, h.size());
	for (auto it = h.begin(); it < end; it++)
		std::cout << *it << " ";
	std::cout << std::endl;
}
