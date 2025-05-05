CC = gcc

# comment this out if you don't use cuda
CUDA_DIR = /home/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda


CFLAGS = -Wall -O2
LD_LIBS = -lm
#CFLAGS += -DDEBUG
CFLAGS += -DOMP -fopenmp


ifdef CUDA_DIR
	CFLAGS += -DCUDA
	LD_LIBS += -L$(CUDA_DIR)/lib64 -lcudart
	NVCC = $(CUDA_DIR)/bin/nvcc
endif


# for debugging segmentation fault
# CFLAGS += -fsanitize=address -g  # useful for segmentation fault

# Parse all the files
SRC = $(wildcard src/*.c) $(wildcard lib/*.c)
SRC_GPU = $(wildcard src/*.cu)
OBJ = $(patsubst %.c,build/%.o,$(notdir $(SRC))) 
ifdef CUDA_DIR
	OBJ += $(patsubst %.cu,build/%.o,$(notdir $(SRC_GPU)))
endif
DEP = $(OBJ:%.o=%.d)
DIR = bin/ build/

all: checkdirs bin/treeeeee

checkdirs: $(DIR)

$(DIR):
	mkdir $@

bin/treeeeee: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LD_LIBS)

-include $(DEP)

build/%.o: src/%.c
	$(CC) $(CFLAGS) -c -MMD -o $@ $<

ifdef CUDA_DIR
build/%.o: src/%.cu

		$(NVCC) -O2 -Xcompiler -fPIC -shared -c -MMD -o $@ $<
endif

build/%.o: lib/%.c
	$(CC) -O2 -c -MMD -o $@ $<

clean:
	rm build/*

