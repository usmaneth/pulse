NVCC ?= /usr/local/cuda/bin/nvcc
CXX ?= g++

NVCCFLAGS = -O3 -std=c++17 -arch=sm_121 -Xcompiler -fPIC -Iinclude
LDFLAGS = -lcudart -lpthread

BIN_DIR = bin
BUILD_DIR = build

TARGET = $(BIN_DIR)/pulse
SO_TARGET = $(BIN_DIR)/libpulse_engine.so

SRCS_CU = src/cuda/speculative_engine.cu
SRCS_CPP = src/main.cpp

OBJS_CU = $(patsubst src/cuda/%.cu, $(BUILD_DIR)/%.o, $(SRCS_CU))
OBJS_CPP = $(patsubst src/%.cpp, $(BUILD_DIR)/%.o, $(SRCS_CPP))

all: $(TARGET) $(SO_TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: src/cuda/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET): $(BIN_DIR) $(OBJS_CU) $(OBJS_CPP)
	$(NVCC) $(NVCCFLAGS) $(OBJS_CU) $(OBJS_CPP) $(LDFLAGS) -o $@

$(SO_TARGET): $(BIN_DIR) $(OBJS_CU)
	$(NVCC) $(NVCCFLAGS) -shared $(OBJS_CU) $(LDFLAGS) -o $@

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)/pulse $(BIN_DIR)/libpulse_engine.so

.PHONY: all clean

LLAMA_DIR ?= /home/usman/Bonsai-demo/llama.cpp
LLAMA_LIB_DIR ?= $(LLAMA_DIR)/build-cuda/bin
ENGINE_LINK = -L$(LLAMA_LIB_DIR) -Wl,-rpath,$(LLAMA_LIB_DIR) -lggml-base

$(BIN_DIR)/pulse-engine: src/engine/engine.cu src/engine/pq2_q8.cuh src/engine/sequential.cuh src/engine/sequence_contract.h src/engine/normalization.h src/engine/model.h src/engine/gguf.h | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -L$(LLAMA_LIB_DIR) -Xlinker -rpath -Xlinker $(LLAMA_LIB_DIR) -lggml-base -o $@

$(BIN_DIR)/pulse-dumpref: src/engine/dump_ref.cpp src/engine/reference_layout.h src/engine/sequence_contract.h | $(BIN_DIR)
	$(CXX) -O2 -std=c++17 -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include $< $(ENGINE_LINK) -lllama -lggml -o $@

$(BIN_DIR)/test-reference-layout: tests/engine/reference_layout.cpp src/engine/reference_layout.h | $(BIN_DIR)
	$(CXX) -O2 -std=c++17 -Isrc/engine $< -o $@

$(BIN_DIR)/test-sequence-contract: tests/engine/sequence_contract.cpp src/engine/sequence_contract.h src/engine/normalization.h | $(BIN_DIR)
	$(CXX) -O2 -std=c++17 -Isrc/engine $< -o $@

native-cpu-test: $(BIN_DIR)/test-reference-layout $(BIN_DIR)/test-sequence-contract
	./$(BIN_DIR)/test-reference-layout
	./$(BIN_DIR)/test-sequence-contract
	python3 tests/engine/test_compare_sequence.py
	python3 tests/engine/test_compare_native_dumps.py

.PHONY: native-cpu-test

$(BIN_DIR)/test-pq2-q8: tests/engine/pq2_q8.cu src/engine/pq2_q8.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(LLAMA_DIR)/ggml/include -Isrc/engine -I$(LLAMA_DIR)/ggml/src -I$(LLAMA_DIR)/ggml/src/ggml-cuda $< -L$(LLAMA_LIB_DIR) -Xlinker -rpath -Xlinker $(LLAMA_LIB_DIR) -lggml-cuda -lggml-base -o $@

$(BIN_DIR)/test-gdn-prefill: tests/engine/gdn_prefill.cu src/engine/gdn_prefill.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -Isrc/engine -I$(LLAMA_DIR)/ggml/include $< -L$(LLAMA_LIB_DIR) -Xlinker -rpath -Xlinker $(LLAMA_LIB_DIR) -lggml-cuda -lggml-base -o $@
