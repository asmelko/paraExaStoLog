#pragma once

#include <iostream>
#include <string>

#include "sparse_utils.h"

template <typename T>
void print(const char* msg, const thrust::device_vector<T>& v, size_t count = 0)
{
	thrust::host_vector<T> h = v;

	std::cout << msg;
	auto end = count == 0 ? h.end() : h.begin() + std::min(count, h.size());
	for (auto it = h.begin(); it < end; it++)
		std::cout << *it << " ";
	std::cout << std::endl;
}

template <typename T1, typename T2>
void print(const char* msg, T1& indptr, T1& indices, T2& data, size_t count = 0)
{
	print(std::string(msg) + " indptr  ", indptr, count);
	print(std::string(msg) + " indices ", indices, count);
	print(std::string(msg) + " data    ", data, count);
}

template <cs_kind k, typename... Args>
void print(const char* msg, sparse_cs_matrix<k, Args...>& m, size_t count = 0)
{
	print(std::string(msg) + " indptr  ", m.indptr, count);
	print(std::string(msg) + " indices ", m.indices, count);
	print(std::string(msg) + " data    ", m.data, count);
}
