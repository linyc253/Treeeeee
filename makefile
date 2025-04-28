CC = g++
CFLAGS = -Wall -g -O2 
LD_LIBS = -lm 
# CFLAGS += -DDEBUG
# CFLAGS += -DMPI

# for debugging segmentation fault
# CFLAGS += -fsanitize=address -g  # useful for segmentation fault

# Parse all the files
SRC = $(wildcard src/*.c)
OBJ = $(patsubst %.c,build/%.o,$(notdir $(SRC)))
DEP = $(OBJ:%.o=%.d)
DIR = bin/ build/

all: checkdirs bin/treeeeee

checkdirs: $(DIR)

$(DIR):
	mkdir $@

bin/treeeeee: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ)

-include $(DEP)

build/%.o: src/%.c
	$(CC) $(CFLAGS) -c -MMD -o $@ $< $(LD_LIBS)

clean:
	rm build/*

