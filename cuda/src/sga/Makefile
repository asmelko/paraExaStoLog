include common.mk
CXXFLAGS+=-std=c++11
INCLUDES+=-I $(CUB_DIR)
SRC=main.cc
OBJS=base.o bitset.o main.o verifier.o omp_base.o topo.o wcc.o two_phase.o kernel_sparse.o

all: $(OBJS) scc_two_phase

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PARFLAG) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

cc_omp: omp_base.o main.o verifier.o kernel_sparse.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o verifier.o kernel_sparse.o -o $@ -lgomp
	mv $@ $(BIN)

scc_base: main.o base.o verifier.o kernel_sparse.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o base.o verifier.o kernel_sparse.o -o $@ $(LIBS)
	mv $@ $(BIN)

scc_two_phase: main.o bitset.o two_phase.o verifier.o wcc.o kernel_sparse.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o bitset.o two_phase.o wcc.o verifier.o kernel_sparse.o -o $@ $(LIBS)
	mv $@ $(BIN)

cc_lb:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
