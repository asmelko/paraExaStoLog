#pragma once

#include <iostream>

#include "sga/timer.h"
#include "types.h"

constexpr bool diags_enabled = true;

template <typename last_t>
void diag_print_to_line(last_t last)
{
	if constexpr (diags_enabled)
	{
		std::cout << last;
	}
}

template <typename first_t, typename... args_t>
void diag_print_to_line(first_t first, args_t... args)
{
	if constexpr (diags_enabled)
	{
		std::cout << first;
		diag_print_to_line(args...);
	}
}

template <typename... args_t>
void diag_print(args_t... args)
{
	if constexpr (diags_enabled)
	{
		diag_print_to_line(args...);
		std::cout << std::endl;
	}
}

void print_big_scc_info(size_t begin, h_idxvec sizes);
void print_terminal_info(const h_idxvec& terminals_offsets);
