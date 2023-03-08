#include "common.h"

int rename_colors(int m, int* h_scc);
void update_transitive_closure(int num_scc, bool* h_trans_closure);
void merge_scc(int m, int num_scc, int* h_scc, bool* h_trans_closure);
void initialize_transitive_closure(int* row_offsets, int* column_indices, int num_scc, int* scc_root, bool* trans_closure);
void read_updates(char* file_path, int m, int num_scc, int* out_row_offsets, int* out_column_indices, int* scc_root, bool* trans_closure);