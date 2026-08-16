CC ?= cc
AR ?= ar
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=off
SIMD ?= 1

ifneq ($(words $(SIMD)),1)
$(error SIMD must be exactly 0 or 1)
endif
ifneq ($(filter-out 0 1,$(SIMD)),)
$(error SIMD must be exactly 0 or 1)
endif

# -Werror opt-in for this Makefile's OWN TUs (default off; CI can flip it on
# with `make WERROR=1`), same control name and shape as pipelines/Makefile's.
# Every object below is first-party C pre/post code compiled by rules in this
# file, so no vendor-code warning isolation is needed here.
#
# `override` is required, not stylistic: CFLAGS above is a `?=` default, so a
# caller passing CFLAGS= on the command line makes CFLAGS a command-line
# variable, and a plain `CFLAGS +=` here would then be discarded -- WERROR=1
# would silently compile without -Werror (verified).
WERROR ?= 0
ifeq ($(WERROR),1)
override CFLAGS += -Werror
endif

CPPFLAGS := -I. -IAlign_ULCNet -IDeepVQE_S -IDeepFilterNet_AENR \
	-I../AINR/DeepFilterNet2 -I../../audio_common/include
ifeq ($(SIMD),0)
CPPFLAGS += -DSIMD_KERNELS_FORCE_SCALAR
endif

BUILD_DIR := build
LIB := $(BUILD_DIR)/libaiaec_prepost.a
OBJECTS := $(BUILD_DIR)/ulcnet_process.o \
	$(BUILD_DIR)/ulcnet_model_io.o \
	$(BUILD_DIR)/ulcnet_accelerator_adapter.o \
	$(BUILD_DIR)/aiaec_process.o \
	$(BUILD_DIR)/deepvqe_process.o \
	$(BUILD_DIR)/dfn_aenr_process.o \
	$(BUILD_DIR)/dfn2_process.o

.PHONY: all lib clean

all: lib

lib: $(LIB)

$(LIB): $(OBJECTS)
	$(AR) rcs "$@" $(OBJECTS)

$(BUILD_DIR):
	mkdir -p "$@"

$(BUILD_DIR)/ulcnet_process.o: Align_ULCNet/ulcnet_process.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c "$<" -o "$@"

$(BUILD_DIR)/ulcnet_model_io.o: Align_ULCNet/ulcnet_model_io.c \
	Align_ULCNet/ulcnet_model_io.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c "$<" -o "$@"

$(BUILD_DIR)/ulcnet_accelerator_adapter.o: \
	Align_ULCNet/ulcnet_accelerator_adapter.c \
	Align_ULCNet/ulcnet_accelerator_adapter.h \
	Align_ULCNet/ulcnet_model_io.h Align_ULCNet/ulcnet_process.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c "$<" -o "$@"

$(BUILD_DIR)/aiaec_process.o: aiaec_process.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c "$<" -o "$@"

$(BUILD_DIR)/deepvqe_process.o: DeepVQE_S/deepvqe_process.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c "$<" -o "$@"

$(BUILD_DIR)/dfn_aenr_process.o: DeepFilterNet_AENR/dfn_aenr_process.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c "$<" -o "$@"

$(BUILD_DIR)/dfn2_process.o: ../AINR/DeepFilterNet2/dfn2_process.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c "$<" -o "$@"

clean:
	rm -rf "$(BUILD_DIR)"
