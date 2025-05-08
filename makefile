CC = clang
CFLAGS = -Wall -O2 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
LDFLAGS = -L/opt/homebrew/opt/libomp/lib
LD_LIBS = -lm -lomp -framework OpenCL
# LD_LIBS = -lm -lomp

# CC = gcc
# CFLAGS = -Wall -O2 
# LD_LIBS = -lm 
# CFLAGS += -DDEBUG
# CFLAGS += -DMPI

# for debugging segmentation fault
# CFLAGS += -fsanitize=address -g  # useful for segmentation fault

# Parse all the files
SRC = $(wildcard src/*.c) $(wildcard lib/*.c)
OBJ = $(patsubst %.c,build/%.o,$(notdir $(SRC)))
DEP = $(OBJ:%.o=%.d)
DIR = bin/ build/

all: checkdirs bin/treeeeee bin/treeeeee_opencl

checkdirs: $(DIR)

$(DIR):
	mkdir -p $@

bin/treeeeee: $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJ) $(LD_LIBS)

-include $(DEP)

# bin/treeeeee_opencl: $(OBJ)
# 	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJ) $(LD_LIBS)

build/main.o: src/main.c
	$(CC) $(CFLAGS) -DUSE_OPENCL -c -MMD -o $@ $<

build/%.o: src/%.c
	$(CC) $(CFLAGS) -c -MMD -o $@ $<

build/%.o: lib/%.c
	$(CC) -O2 -c -MMD -o $@ $<

ifdef CUDA_DIR
build/%.o: src/%.cu

		$(NVCC) -O2 -Xcompiler -fPIC -shared -c -MMD -o $@ $<
endif

build/%.o: lib/%.c
	$(CC) -O2 -c -MMD -o $@ $<

clean:
	rm build/*