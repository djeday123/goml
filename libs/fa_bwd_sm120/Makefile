## libfa_bwd_sm120 — FlashAttention backward chain wrappers для sm_120a.
## A-LLM-1 G2 (2026-07-25). Отдельный .so от libfa_sm120.so (боевой fwd не трогаем).
##
## make lib       → libfa_bwd_sm120.so + .a
## make clean

CUDA      ?= /usr/local/cuda-13.1
NVCC      := $(CUDA)/bin/nvcc
ARCH      := sm_120a

V02       := ../../release_v0.2.0/src
INCLUDE   := -Iinclude -I$(V02)

NVCC_FLAGS := -O3 -gencode arch=compute_120a,code=$(ARCH) -std=c++17 \
              -Xcompiler "-fPIC -fvisibility=hidden" \
              -Xptxas=-v

BUILD     := build

# v0.2.0 kernel sources — все нужны для linkage merged_v1 + dk_new + dq_new + d_precompute.
# merged_v1 depends on ds_gen + dv_mma_p1 (fused).
# d_precompute lives in fa_bwd_dk.cu (same namespace as legacy launch_dk).
KERNEL_SRCS := $(V02)/fa_bwd_dk.cu \
               $(V02)/fa_bwd_dk_new.cu \
               $(V02)/fa_bwd_dq_new.cu \
               $(V02)/fa_bwd_ds_gen.cu \
               $(V02)/fa_bwd_dv_mma_p1.cu \
               $(V02)/fa_bwd_merged_v1.cu

WRAPPER_SRC := src/wrapper.cu

KERNEL_OBJS := $(patsubst $(V02)/%.cu,$(BUILD)/v02_%.o,$(KERNEL_SRCS))
WRAPPER_OBJ := $(BUILD)/wrapper.o

LIB_SO    := libfa_bwd_sm120.so
LIB_A     := libfa_bwd_sm120.a

.PHONY: all lib clean

all: lib

lib: $(LIB_SO) $(LIB_A)

$(LIB_SO): $(WRAPPER_OBJ) $(KERNEL_OBJS)
	$(NVCC) -shared -o $@ $^ $(NVCC_FLAGS) -lcudart
	@echo ""
	@echo "== Built $(LIB_SO) =="
	@nm -D --defined-only $(LIB_SO) | grep -E "gt_fa_bwd_" || true

$(LIB_A): $(WRAPPER_OBJ) $(KERNEL_OBJS)
	ar rcs $@ $(WRAPPER_OBJ) $(KERNEL_OBJS)

$(BUILD)/wrapper.o: $(WRAPPER_SRC)
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE) -c $< -o $@

$(BUILD)/v02_%.o: $(V02)/%.cu
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE) -c $< -o $@

clean:
	rm -rf $(BUILD) $(LIB_SO) $(LIB_A)
