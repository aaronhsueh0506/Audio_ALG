# HPF (High-Pass Filter) Makefile

CC ?= gcc
CFLAGS = -Wall -Wextra -O2 -std=c99 -I./include
CFLAGS += $(EXTRA_CFLAGS)

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

SRCS = $(SRC_DIR)/hpf.c
OBJS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))

LIB_TARGET = $(BIN_DIR)/libhpf.a

.PHONY: all lib clean dirs

all: lib

dirs:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)

lib: dirs $(OBJS)
	ar rcs $(LIB_TARGET) $(OBJS)
	@echo "Static library built: $(LIB_TARGET)"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Clean complete"
