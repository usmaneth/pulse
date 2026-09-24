LLAMA_DIR ?= ../../../llama.cpp
LLAMA_LIB_DIR ?= $(LLAMA_DIR)/build-cuda/bin
OUTPUT ?= ./artifacts/rope-probe
CXX ?= g++

$(OUTPUT): rope_probe.cpp
	$(CXX) -O2 -std=c++17 -I$(LLAMA_DIR)/ggml/include $< -L$(LLAMA_LIB_DIR) -Wl,-rpath,$(LLAMA_LIB_DIR) -lggml-cpu -lggml-base -o $@
