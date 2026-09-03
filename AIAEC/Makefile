CC ?= cc
AR ?= ar
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=off
SIMD ?= 1
override SIMD := $(strip $(SIMD))

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

CPPFLAGS := -I. -IAlign_ULCNet -IDeepVQE_S -I../../audio_common/include
ifeq ($(SIMD),0)
CPPFLAGS += -DSIMD_KERNELS_FORCE_SCALAR
endif

# --- Configuration-keyed outputs --------------------------------------------
# Every knob that changes what an object contains is folded into a signature,
# and the objects, their dependency files and the archive live under
# build/simd<N>-<sig>/. Without this, `make SIMD=0 lib` after a SIMD=1 build
# reports "Nothing to be done" and hands the NEON archive to a caller that
# asked for the scalar one. pipelines/Makefile keys its outputs the same way.
# `make print-lib-path` prints the archive for the current configuration, and
# each directory records its payload so a signature collision is refused
# rather than reinterpreted.
BUILD_DIR := build
CFG_SIG_PAYLOAD := CC=$(CC) AR=$(AR) CFLAGS=$(CFLAGS) CPPFLAGS=$(CPPFLAGS) SIMD=$(SIMD) WERROR=$(WERROR)
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)
OBJ_DIR := $(BUILD_DIR)/simd$(SIMD)-$(CFG_SIG)
LIB := $(OBJ_DIR)/libaiaec_prepost.a
# ulcnet_prepost.o and deepvqe_prepost.o are the pre/post classes, each a
# THIRD TU on purpose: the two files a class composes (ulcnet_process.c +
# ulcnet_model_io.c; aiaec_process.c + deepvqe_process.c) stay independently
# linkable for their own parity tests. The accelerator adapter drives the
# Align-ULCNet class.
OBJECTS := $(OBJ_DIR)/ulcnet_process.o \
	$(OBJ_DIR)/ulcnet_model_io.o \
	$(OBJ_DIR)/ulcnet_prepost.o \
	$(OBJ_DIR)/ulcnet_accelerator_adapter.o \
	$(OBJ_DIR)/aiaec_process.o \
	$(OBJ_DIR)/deepvqe_process.o \
	$(OBJ_DIR)/deepvqe_prepost.o

PYTHON ?= python3

.PHONY: all lib print-lib-path test clean _cfg_guard

all: lib

lib: $(LIB)

print-lib-path:
	@echo "$(abspath $(LIB))"

# rm first: `ar rcs` updates in place and never prunes, so an object dropped
# from OBJECTS would otherwise linger in the archive under the same key.
$(LIB): $(OBJECTS)
	rm -f "$@" && $(AR) rcs "$@" $(OBJECTS)

_cfg_guard:
	@mkdir -p "$(OBJ_DIR)"
	@m="$(OBJ_DIR)/config.manifest"; \
	if [ -f "$$m" ] && [ "$$(cat "$$m")" != "$(CFG_SIG_PAYLOAD)" ]; then \
	  echo "FATAL: configuration signature collision in $$m" >&2; \
	  echo "  stored : $$(cat "$$m")" >&2; \
	  echo "  current: $(CFG_SIG_PAYLOAD)" >&2; \
	  exit 1; \
	fi; \
	printf '%s\n' "$(CFG_SIG_PAYLOAD)" > "$$m"

# One recipe for every object: the sources live in three directories, so
# VPATH resolves the %.c, and -MMD -MP records each object's real header
# dependencies in $(OBJ_DIR)/*.d instead of hand-maintained prerequisite
# lists.
VPATH := Align_ULCNet:DeepVQE_S:.

$(OBJ_DIR)/%.o: %.c | _cfg_guard
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c "$<" -o "$@"

-include $(OBJECTS:.o=.d)

# The command README.md documents. Building the library first is part of the
# gate: it is the only place the objects are compiled with this file's flags
# (the pytest C drivers compile their own translation units).
test: lib
	$(PYTHON) -m pytest tests

clean:
	rm -rf "$(BUILD_DIR)"
