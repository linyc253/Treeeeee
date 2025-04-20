CC = gcc
CFLAGS = -O2
#CFLAGS += -Ddebug
#CFLAGS += -Dmpi

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
	$(CC) $(CFLAGS) -c -MMD -o $@ $<

clean:
	rm build/*
