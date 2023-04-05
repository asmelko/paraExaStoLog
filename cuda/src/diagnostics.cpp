#include "diagnostics.h"

#include <map>

void print_big_scc_info(size_t begin, h_idxvec sizes)
{
	std::map<size_t, size_t> terminals_histo;

	for (size_t terminal_scc_idx = begin; terminal_scc_idx < sizes.size(); terminal_scc_idx++)
	{
		size_t size =
			terminal_scc_idx == begin ? sizes[terminal_scc_idx] : sizes[terminal_scc_idx] - sizes[terminal_scc_idx - 1];

		if (terminals_histo.find(size) == terminals_histo.end())
			terminals_histo[size] = 1;
		else
			terminals_histo[size]++;
	}

	size_t max_print = 10;
	std::string diag;
	for (auto it = terminals_histo.rbegin(); it != terminals_histo.rend(); it++)
	{
		if (--max_print == 0)
			break;

		diag += "(" + std::to_string(it->first) + ", " + std::to_string(it->second) + ") ";
	}

	if (terminals_histo.size())
		diag_print("LU: the biggest sccs (size, #occurences): ", diag);
}

void print_terminal_info(const h_idxvec& terminals_offsets)
{
	if constexpr (!diags_enabled)
		return;

	std::map<size_t, size_t> terminals_histo;

	for (size_t terminal_scc_idx = 0; terminal_scc_idx < terminals_offsets.size() - 1; terminal_scc_idx++)
	{
		size_t size = terminals_offsets[terminal_scc_idx + 1] - terminals_offsets[terminal_scc_idx];

		if (terminals_histo.find(size) == terminals_histo.end())
			terminals_histo[size] = 1;
		else
			terminals_histo[size]++;
	}

	size_t max_print = 10;
	std::string diag;
	for (auto it = terminals_histo.rbegin(); it != terminals_histo.rend(); it++)
	{
		if (--max_print == 0)
			break;

		diag += "(" + std::to_string(it->first) + ", " + std::to_string(it->second) + ") ";
	}

	diag_print("Terminals count: ", terminals_offsets.size() - 1);
	diag_print("The biggest terminals (size, #occurences): ", diag);
}
