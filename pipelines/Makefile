# Audio_ALG/pipelines/Makefile
#
# AEC(linear) -> echo-aware NR -> RES pipeline — two versions of the SAME
# algorithm (see aec_nr_pipeline.c's header for the DSP chain):
#
#   aec_nr_pipeline         (Version A: malloc)         — heap-allocated modules
#   aec_nr_pipeline_static  (Version B: static memory)  — one caller pool, no
#                                                          malloc after init
#
# Both link against the submodule libs (lib/aec, lib/nr) + the shared
# audio_common FFT/fast_math layer (../../audio_common relative to this dir).
#
# Usage:
#   make                       # builds libs (BACKEND=kiss) + both binaries
#   make BACKEND=ne10          # NE10 backend -- F12: no manual clean needed
#                              # any more, a CFG_STAMP in each of this file's
#                              # and the submodules' Makefiles now busts stale
#                              # objects automatically on a BACKEND/EXTRA_CFLAGS
#                              # change (make clean-all still works, just isn't
#                              # required)
#   make EXTRA_CFLAGS="..."    # see the EXTRA_CFLAGS block below
#   make WERROR=1              # opt-in: -Werror for this file's own two TUs
#   make clean                 # remove the two pipeline binaries (+ .o's)
#   make clean-all             # also clean lib/aec, lib/nr, audio_common build
#                              # dirs (clean-libs is a compatibility alias)

AEC_DIR = ../lib/aec/c_impl
NR_DIR  = ../lib/nr/c_impl

# audio_common sits two levels up from pipelines/ (SE/Audio_ALG/pipelines -> SE/audio_common).
# lib/aec/c_impl and lib/nr/c_impl each resolve it themselves from their own
# location (../../../../audio_common from c_impl/, verified to exist as a
# submodule checkout); AC_DIR here is for OUR OWN CFLAGS/LDFLAGS and is passed
# through explicitly to the sub-makes so all three always agree on the path.
AC_DIR ?= ../../audio_common
BACKEND ?= kiss

CC  ?= gcc
CXX ?= c++

# EXTRA_CFLAGS — pass-through to the AEC/NR lib sub-builds AND this pipeline's
# own compile (both submodule Makefiles already do `CFLAGS += $(EXTRA_CFLAGS)`,
# so one EXTRA_CFLAGS on this top-level `make` reaches every .o for the build).
# The fast matched-filter arithmetic and delay-estimator duty-cycling are
# built in unconditionally (no flag) — see lib/aec/c_impl/src/delay_aec3.c.
#
# F12 fix: switching EXTRA_CFLAGS (or BACKEND, or WERROR) on an already-built
# tree used to need `make clean-libs` by hand first -- object files weren't
# flag-tagged, so a stale non-flagged .o (or a stale flagged one, if you're
# turning a flag back off) would silently persist otherwise. The CFG_STAMP
# mechanism further down now detects the change and wipes the stale objects
# automatically (both here and, independently, inside lib/aec's and lib/nr's
# own Makefiles); `make clean-libs`/`clean-all` remains available but is no
# longer required for correctness.
EXTRA_CFLAGS ?=

CFLAGS = -Wall -Wextra -O2 -std=gnu99 \
         -I$(AEC_DIR)/include -I$(AEC_DIR)/example \
         -I$(NR_DIR)/include \
         -I$(AC_DIR)/include
CFLAGS += $(EXTRA_CFLAGS)

# -Werror opt-in for this Makefile's OWN two TUs (default off; CI can flip it
# on with `make WERROR=1`). Vendor code (audio_common's NE10/KISS) is built by
# audio_common's own Makefile (see `libs:` below), not by rules in this file,
# so no separate warning isolation is needed on this side.
WERROR ?= 0
ifeq ($(WERROR),1)
  CFLAGS += -Werror
endif

# The NE10 archive carries one C++ TU (NE10's generic-radix FFT kernel); a
# plain-C pipeline binary linking against it needs the C++ runtime driver.
LINK = $(CC)
LDFLAGS_EXTRA =
ifeq ($(BACKEND),ne10)
  LINK = $(CXX)
  LDFLAGS_EXTRA = -lc++
endif

# Link order matters: pipeline.o, libaec.a, libmmse_lsa.a, libaudio_common.a, -lm
LIBS = $(AEC_DIR)/bin/libaec.a \
       $(NR_DIR)/bin/libmmse_lsa.a \
       $(AC_DIR)/bin/$(BACKEND)/libaudio_common.a
LDFLAGS = $(LIBS) -lm $(LDFLAGS_EXTRA)

# --- Build-configuration stamp (F12 build hygiene) --------------------------
# This Makefile keeps fixed .o names (aec_nr_pipeline.o,
# aec_nr_pipeline_static.o) directly in this directory -- there is no obj/
# subdir to wipe wholesale (this dir also holds the Makefile, scripts, docs,
# etc.). Until now, switching BACKEND or EXTRA_CFLAGS on an already-built
# tree needed `make clean-libs` by hand first (documented above) or a
# stale, differently-flagged .o would silently persist. CFG_SIG hashes the
# exact compiler invocation that produces both of this Makefile's own
# objects.
#
# This check runs at PARSE time (via $(shell ...), not as a build-rule
# recipe): an order-only-prerequisite version (a stamp FILE that a rule
# creates, referenced via `foo.o: foo.c | $(CFG_STAMP)`) is racy -- verified
# empirically. When several objects share one order-only prerequisite whose
# recipe deletes them, GNU Make can finalize "is this object out of date?"
# for one of them using the prerequisite tree it already resolved, before
# the shared prerequisite's recipe actually runs -- so that object silently
# keeps its (about to be deleted) stale .o and is never recompiled. Doing
# the compare-and-wipe unconditionally while the Makefile is being read
# sidesteps the race entirely.
PIPE_OBJS := aec_nr_pipeline.o aec_nr_pipeline_static.o

CFG_SIG        := $(shell printf '%s' "$(CC) $(CFLAGS) $(EXTRA_CFLAGS) BACKEND=$(BACKEND) WERROR=$(WERROR)" | shasum -a 256 | cut -c1-12)
CFG_STAMP_FILE := .cfg_stamp
CFG_LAST_SIG   := $(shell test -f $(CFG_STAMP_FILE) && cat $(CFG_STAMP_FILE))
ifneq ($(CFG_SIG),$(CFG_LAST_SIG))
$(info Audio_ALG/pipelines: build config changed -- removing stale .o's for a full rebuild)
$(shell rm -f $(PIPE_OBJS))
$(shell echo $(CFG_SIG) > $(CFG_STAMP_FILE))
endif
# (Note: this only re-tags THIS Makefile's own two objects -- the
# submodule/shared libs pulled in by `libs:` below are independently
# stamped by their own Makefiles, which now key off the same
# BACKEND/EXTRA_CFLAGS/WERROR axes.)

.PHONY: all clean clean-libs clean-all libs

all: libs aec_nr_pipeline aec_nr_pipeline_static

# Build the shared archive + both submodule libs for the selected BACKEND.
# AC_DIR is passed through explicitly so lib/aec and lib/nr resolve the SAME
# audio_common checkout this Makefile uses (their own AC_DIR fallback also
# resolves correctly at the submodule depth: ../../../../audio_common from
# c_impl/, but pass it anyway to be explicit/robust).
libs:
	$(MAKE) -C $(AC_DIR) BACKEND=$(BACKEND) WERROR=$(WERROR) lib
	$(MAKE) -C $(AEC_DIR) lib BACKEND=$(BACKEND) AC_DIR=$(abspath $(AC_DIR)) EXTRA_CFLAGS="$(EXTRA_CFLAGS)" WERROR=$(WERROR)
	$(MAKE) -C $(NR_DIR) lib BACKEND=$(BACKEND) AC_DIR=$(abspath $(AC_DIR)) EXTRA_CFLAGS="$(EXTRA_CFLAGS)" WERROR=$(WERROR)

# Compile with $(CC) (always a plain-C driver, -std=gnu99 valid) then link with
# $(LINK) (c++ for BACKEND=ne10 — the NE10 archive's one C++ TU needs it). A
# C++ driver invoked directly on a .c source rejects -std=gnu99 and treats the
# file as C++, so the compile and link steps must stay separate here (mirrors
# how lib/aec and lib/nr's own Makefiles split .o-compile from $(LINK)).
aec_nr_pipeline.o: aec_nr_pipeline.c
	$(CC) $(CFLAGS) -c -o $@ $<

aec_nr_pipeline_static.o: aec_nr_pipeline_static.c
	$(CC) $(CFLAGS) -c -o $@ $<

aec_nr_pipeline: aec_nr_pipeline.o | libs
	$(LINK) -o $@ $< $(LDFLAGS)

aec_nr_pipeline_static: aec_nr_pipeline_static.o | libs
	$(LINK) -o $@ $< $(LDFLAGS)

clean:
	rm -f aec_nr_pipeline aec_nr_pipeline_static aec_nr_pipeline.o aec_nr_pipeline_static.o .cfg_stamp

# clean-all: full nuke -- this dir's own artifacts plus all three
# submodule/shared build trees (lib/aec, lib/nr, audio_common). clean-libs is
# kept as a backward-compatible alias (same effect) since existing
# tooling/docs call it by that name.
clean-all: clean
	$(MAKE) -C $(AEC_DIR) clean
	$(MAKE) -C $(NR_DIR) clean
	$(MAKE) -C $(AC_DIR) clean

clean-libs: clean-all
