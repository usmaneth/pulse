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
