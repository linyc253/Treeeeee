CC = clang
CFLAGS = -Wall -O2 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
LDFLAGS = -L/opt/homebrew/opt/libomp/lib
LD_LIBS = -lm -lomp


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

all: checkdirs bin/treeeeee

checkdirs: $(DIR)

$(DIR):
	mkdir $@

bin/treeeeee: $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJ) $(LD_LIBS)

-include $(DEP)

build/%.o: src/%.c
	$(CC) $(CFLAGS) -c -MMD -o $@ $<

build/%.o: lib/%.c
	$(CC) -O2 -c -MMD -o $@ $<

clean:
	rm build/*

