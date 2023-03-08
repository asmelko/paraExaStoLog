#include "kernel.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#define GPU_ERR_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   	if (code != cudaSuccess) 
   	{
	  	fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	  	if (abort) exit(code);
   	}
}

using namespace cooperative_groups;

__global__ void update_colors(int m, int* scc, int* colors){
   	int tid = blockIdx.x * blockDim.x + threadIdx.x;
   
   	int scc_id = -1;
   	if(tid < m){
	  	scc_id = scc[tid];
	  	colors[scc_id] = 1;
   	}

   	grid_group grid = this_grid();
   	grid.sync();

   	int tmp;
   	// prefix sum
   	for(int off = 1; off < m; off *= 2){
	  	if(tid >= off && tid < m){
		 	tmp = colors[tid - off];
	  	}
	  	grid.sync();
	  
	  	if(tid >= off && tid < m){
			colors[tid] += tmp;
	 	}
	  	grid.sync();
   	}

   	if(tid < m){
	  	scc[tid] = colors[scc_id] - 1;
   	}
}

__global__ void update_tr_clsr_kernel(int k, int num_scc, bool* trans_closure){
   	int tid = blockIdx.x * blockDim.x + threadIdx.x;

   	if(tid < num_scc * num_scc){
	  	int i = tid / num_scc, j = tid % num_scc;
	 	trans_closure[tid] |= (trans_closure[i*num_scc + k] && trans_closure[k*num_scc + j]);
 	}
}

__global__ void merge_scc_kernel(int m, int num_scc, int* scc, int* colors, bool* trans_closure){
   	int tid = blockIdx.x * blockDim.x + threadIdx.x;
   
   	if(tid < num_scc)
	  	colors[tid] = tid;

   	grid_group grid = this_grid();
   	grid.sync();

	for(int i = 0; i < num_scc; i++){
		if(tid < num_scc && tid > i){
			if(trans_closure[i*num_scc + tid] && trans_closure[tid*num_scc + i]){
				if(colors[tid] < colors[i]){
			   		colors[i] = colors[tid];
				}
				else{
			   		colors[tid] = colors[i];
				}
			}
		}

		grid.sync();
	}

	for(int i = 0; i <= m / num_scc; i++){
		int index = i * num_scc + tid;

		if(tid < num_scc && index < m)
			scc[index] = colors[scc[index]];
	}
}

int rename_colors(int m, int* h_scc){
	int *d_colors, *d_scc, *num_scc;
	GPU_ERR_CHK(cudaMalloc(&d_colors, m * sizeof(int)));
	GPU_ERR_CHK(cudaMalloc(&d_scc, m * sizeof(int)));
	num_scc = (int *)malloc(sizeof(int));
	GPU_ERR_CHK(cudaMemset(d_colors, 0, m * sizeof(int)))
	GPU_ERR_CHK(cudaMemcpy(d_scc, h_scc, m * sizeof(int), cudaMemcpyHostToDevice));

	int nthreads = MAXBLOCKSIZE;
	int nblocks = (m - 1) / nthreads + 1;
	void *kernelArgs[] = {
	    (void*)&m,
	    (void*)&d_scc,
	    (void*)&d_colors,
	};
	dim3 dimBlock(nthreads, 1, 1);
	dim3 dimGrid(nblocks, 1, 1);

	//update_colors<<<nblocks, nthreads>>>(m, d_scc, d_colors);
	GPU_ERR_CHK(cudaLaunchCooperativeKernel((void*)update_colors, dimGrid, dimBlock, kernelArgs, 0, NULL));
	GPU_ERR_CHK(cudaMemcpy(h_scc, d_scc, m * sizeof(int), cudaMemcpyDeviceToHost));
	GPU_ERR_CHK(cudaMemcpy(num_scc, &(d_scc[m-1]), sizeof(int), cudaMemcpyDeviceToHost));

	GPU_ERR_CHK(cudaFree(d_colors));
	GPU_ERR_CHK(cudaFree(d_scc));

	int ret = (1+*num_scc);
	free(num_scc);
	return ret;
}

void update_transitive_closure(int num_scc, bool* h_trans_closure){
	int nthreads = MAXBLOCKSIZE;
	int nblocks = (num_scc*num_scc - 1) / nthreads + 1;

	bool *d_trans_closure;
	GPU_ERR_CHK(cudaMalloc(&d_trans_closure, num_scc * num_scc * sizeof(bool)));
	GPU_ERR_CHK(cudaMemcpy(d_trans_closure, h_trans_closure, num_scc * num_scc * sizeof(bool), cudaMemcpyHostToDevice));

	for(int k = 0; k < num_scc; k++)
	   update_tr_clsr_kernel<<<nblocks, nthreads>>>(k, num_scc, d_trans_closure);

	GPU_ERR_CHK(cudaMemcpy(h_trans_closure, d_trans_closure, num_scc * num_scc * sizeof(bool), cudaMemcpyDeviceToHost));
	GPU_ERR_CHK(cudaFree(d_trans_closure));
}

void merge_scc(int m, int num_scc, int* h_scc, bool* h_trans_closure){
	int nthreads = MAXBLOCKSIZE;
	int nblocks = (num_scc - 1) / nthreads + 1;
   
	int *d_scc, *d_colors;
	bool *d_trans_closure;
	GPU_ERR_CHK(cudaMalloc(&d_scc, m * sizeof(int)));
	GPU_ERR_CHK(cudaMalloc(&d_trans_closure, num_scc * num_scc * sizeof(bool)));
	GPU_ERR_CHK(cudaMalloc(&d_colors, num_scc * sizeof(int)));
	GPU_ERR_CHK(cudaMemcpy(d_scc, h_scc, m * sizeof(int), cudaMemcpyHostToDevice));
	GPU_ERR_CHK(cudaMemcpy(d_trans_closure, h_trans_closure, num_scc * num_scc * sizeof(bool), cudaMemcpyHostToDevice));

	void *kernelArgs[] = {
	    (void*)&m,
	    (void*)&num_scc,
	    (void*)&d_scc,
	    (void*)&d_colors,
	    (void*)&d_trans_closure,
	};
	dim3 dimBlock(nthreads, 1, 1);
	dim3 dimGrid(nblocks, 1, 1);

	GPU_ERR_CHK(cudaLaunchCooperativeKernel((void*)merge_scc_kernel, dimGrid, dimBlock, kernelArgs, 0, NULL));
	GPU_ERR_CHK(cudaMemcpy(h_scc, d_scc, m * sizeof(int), cudaMemcpyDeviceToHost));

	GPU_ERR_CHK(cudaFree(d_scc));
	GPU_ERR_CHK(cudaFree(d_trans_closure));
	GPU_ERR_CHK(cudaFree(d_colors));
}

void initialize_transitive_closure(int* row_offsets, int* column_indices, int num_scc, int* scc_root, bool* trans_closure){

   //int dev = 0;
   //cudaDeviceProp deviceProp;
   //cudaGetDeviceProperties(&deviceProp, dev);
   //printf("Maximum Active Blocks (assuming one block per SM)= %d\n", deviceProp.multiProcessorCount);
   //printf("Maximum size of graph = %d nodes\n", deviceProp.multiProcessorCount*MAXBLOCKSIZE);

   for(int i = 1; i <= num_scc; i++){
	  int row_start = row_offsets[i-1], row_end = row_offsets[i];
	  int u = scc_root[i-1], v;
	  
	  for(int j = row_start; j < row_end; j++){
		 v = scc_root[column_indices[j]];
		 trans_closure[u*num_scc + v] = true;
	  }
   }
}

void read_updates(char* file_path, int m, int num_scc, int* out_row_offsets, int* out_column_indices, int* scc_root, bool* trans_closure){
	FILE* file = fopen(file_path, "r");
	  
	if(!file){  
	   printf("Update file can't be read\n"); 
	   exit(-1); 
	} 

	int x, y;
	//std::vector<int> tail, head;

	while(fscanf(file, "%d", &x) == 1 && fscanf(file, "%d", &y) == 1)  
	{
	   x--; y--;
	   if(x >= m || y >= m){
		  printf("Node %d or %d in update file doesn't exist\n", x, y); 
		  exit(-1); 
	   }

	   //tail.push_back(x);
	   //head.push_back(y);
	   x = scc_root[x];
	   y = scc_root[y];

	   trans_closure[x*num_scc + y] = true;
	}

	// deallocate the memmory held by vectors
	// tail = std::vector<int>();
	// head = std::vector<int>();

	// TODO: edge update must be added to original graph for edge deletion case
}