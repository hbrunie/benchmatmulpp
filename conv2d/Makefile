
#Flags
CCGCC = g++ -DUSE_GNU
CCLLVM = clang++-12 -DUSE_LLVM
CCICPC = icpc -DMKL_ILP64  -qmkl=parallel -qopt-report=3 -DUSE_ICPC
CCPOLLY = clang++-12 -mllvm -polly
CCPOLLY = clang++-12 -mllvm -polly  -Rpass -fsave-optimization-record -DUSE_POLLY
FLAGS =-O3 -g -Wall -DMORE_CONV -march=native -mtune=native
FLAGS =-O3 -g -Wall -march=native -mtune=native -DMORE_CONV
#CFLAGS += $(FLAGS)  -O3 -std=c++11 -DNO_RECORD -I /opt/intel/oneapi/dnnl/latest/cpu_iomp/include/
#CFLAGS += $(FLAGS)  -mllvm -polly -std=c++11 -DNO_RECORD -I /opt/intel/oneapi/dnnl/latest/cpu_iomp/include/
#LDFLAGS += $(FLAGS)

FC_SRC=main_unrolled_function.cc
C_SRC= main_unrolled_macro.cc

.phony: all clean
BIN=\
	convDNN_llvm_polly_ufunc\
	convDNN_llvm_polly_ufunc_cc\
	#convDNN_intel_ufunc_cc  \
	#convDNN_llvm_ufunc_cc  \
	#convDNN_gnu_ufunc_cc \
	#convDNN_intel_ufunc  \
	#convDNN_llvm_ufunc  \
	#convDNN_gnu_ufunc \
	#convDNN_intel_unrolled_cc  \
	#convDNN_llvm_unrolled_cc  \
	#convDNN_gnu_unrolled_cc \
	#convDNN_intel_unrolled  \
	#convDNN_llvm_unrolled  \
	#convDNN_gnu_unrolled \
	#convDNN_llvm_polly_ufunc_cc \
	#convDNN_llvm_polly_ufunc \

all: $(BIN)

convDNN_intel_ufunc_cc:
	$(CCICPC) $(FC_SRC) $(FLAGS) -o $@  -DUSE_COLDCACHE

convDNN_gnu_ufunc_cc:
	$(CCGCC) $(FC_SRC) $(FLAGS) -o $@  -DUSE_COLDCACHE

convDNN_llvm_ufunc_cc:
	$(CCLLVM) $(FC_SRC) $(FLAGS) -o $@  -DUSE_COLDCACHE

convDNN_llvm_polly_ufunc_cc:
	$(CCPOLLY) $(FC_SRC) $(FLAGS) -o $@  -DUSE_COLDCACHE

convDNN_intel_unrolled_cc:
	$(CCICPC) $(C_SRC) $(FLAGS) -o $@  -DUSE_COLDCACHE

convDNN_gnu_unrolled_cc:
	$(CCGCC) $(C_SRC) $(FLAGS) -o $@  -DUSE_COLDCACHE

convDNN_llvm_unrolled_cc:
	$(CCLLVM) $(C_SRC) $(FLAGS) -o $@  -DUSE_COLDCACHE

convDNN_llvm_polly_unrolled_cc:
	$(CCPOLLY) $(C_SRC) $(FLAGS) -o $@  -DUSE_COLDCACHE

convDNN_intel_ufunc:
	$(CCICPC) $(FC_SRC) $(FLAGS) -o $@

convDNN_gnu_ufunc:
	$(CCGCC) $(FC_SRC) $(FLAGS) -o $@

convDNN_llvm_ufunc:
	$(CCLLVM) $(FC_SRC) $(FLAGS) -o $@

convDNN_llvm_polly_ufunc:
	$(CCPOLLY) $(FC_SRC) $(FLAGS) -o $@

convDNN_intel_unrolled:
	$(CCICPC) $(C_SRC) $(FLAGS) -o $@

convDNN_gnu_unrolled:
	$(CCGCC) $(C_SRC) $(FLAGS) -o $@

convDNN_llvm_unrolled:
	$(CCLLVM) $(C_SRC) $(FLAGS) -o $@

convDNN_llvm_polly_unrolled:
	$(CCPOLLY) $(C_SRC) $(FLAGS) -o $@

clean:
	rm -f $(BIN)

cleanall: clean
	rm -f *.opt.yaml *.optrpt



