CC = gcc
CFLAGS = -Wall -O2 -fopenmp
LD_LIBS = -lm
CFLAGS += -DDEBUG
CFLAGS += -DOMP
export OMP_NUM_THREADS = 4

# for debugging segmentation fault
CFLAGS += -fsanitize=address -g  # useful for segmentation fault

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
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LD_LIBS)

-include $(DEP)

build/%.o: src/%.c
	$(CC) $(CFLAGS) -c -MMD -o $@ $<

build/%.o: lib/%.c
	$(CC) -O2 -c -MMD -o $@ $<

clean:
	rm build/*

