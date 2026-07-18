NVCC     := /usr/local/cuda-13.1/bin/nvcc
ARCH     := -gencode arch=compute_120a,code=sm_120a
STD      := -std=c++17
OPT      := -O3
PTXFLAGS := -Xptxas=-v

BIN := fa_probe_bank

$(BIN): fa_probe_bank.cu
	$(NVCC) $(OPT) $(ARCH) $(STD) $(PTXFLAGS) fa_probe_bank.cu -o $(BIN) 2>&1 | tee probe_build.log

sass: $(BIN)
	/usr/local/cuda-13.1/bin/cuobjdump --dump-sass $(BIN) > probe_sass_full.txt 2>&1
	@echo "SASS dumped to probe_sass_full.txt"

clean:
	rm -f $(BIN) probe_build.log probe_sass_full.txt
