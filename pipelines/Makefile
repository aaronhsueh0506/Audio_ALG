# Audio_ALG/pipelines/Makefile
#
# AEC(linear) -> echo-aware NR -> RES pipeline — two versions of the SAME
# algorithm (see aec_nr_pipeline.c's header for the DSP chain):
#
#   aec_nr_pipeline         (Version A: malloc)         — heap-allocated modules
#   aec_nr_pipeline_static  (Version B: static memory)  — one caller pool, no
#                                                          malloc after init
#
# Both are now thin CLI shells (arg parsing + WAV I/O + diagnostics) over
# audio_pipeline.h/.c (review F20) — the pool-sizing/carving/per-hop-
# processing logic itself, built into libaudio_pipeline.a, linkable on its
# own by a board's memory manager. See pipelines/README.md's "Board
# Integration" section.
#
# All three link against the submodule libs (lib/aec, lib/nr) + the shared
# audio_common FFT/fast_math layer (../../audio_common relative to this dir).
#
# Usage:
#   make                       # builds libs (BACKEND=kiss) + both binaries
#                              # (+ libaudio_pipeline.a as a side effect)
#   make BACKEND=ne10          # NE10 backend -- objects are hash-keyed per
#                              # (BACKEND, EXTRA_CFLAGS, WERROR) in each of
#                              # this file's and the submodules' Makefiles,
#                              # so switching any of them always lands in a
#                              # fresh object directory automatically; no
#                              # manual clean needed, ever (make clean-all
#                              # still works, just isn't required)
#   make EXTRA_CFLAGS="..."    # see the EXTRA_CFLAGS block below
#   make WERROR=1              # opt-in: -Werror for this file's own TUs
#   make test                  # build + run test_audio_pipeline (audio_pipeline.h
#                              # acceptance tests -- see that file's header)
#   make clean                 # remove the pipeline binaries/objects/archive
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
# Switching EXTRA_CFLAGS (or BACKEND, or WERROR) never needs a manual
# `make clean-libs` first: this Makefile's own objects are hash-keyed by
# config (see the OBJ_DIR/CFG_SIG comment further down), and lib/aec's and
# lib/nr's own Makefiles independently key their objects off the same
# BACKEND/EXTRA_CFLAGS/WERROR axes -- a stale, differently-flagged .o simply
# can't exist under the new path a config produces. `make clean-libs`/
# `clean-all` remain available but are no longer required for correctness.
EXTRA_CFLAGS ?=

# -ffp-contract=off: same FMA-contraction policy as the AEC library TUs.
# Deliberate BYTE-CHANGE when it landed (clang had been contracting a
# handful of mul+add pairs in the pipeline glue, e.g. the |E|^2 hoist) —
# the pipeline render baselines were regenerated at that commit. Keeping
# the flag uniform means future SIMD/EMA-style code added to pipeline TUs
# inherits the repo-wide bit-exactness discipline instead of silently
# depending on the compiler's contraction mood (review F12).
CFLAGS = -Wall -Wextra -O2 -std=gnu99 -ffp-contract=off \
         -I$(AEC_DIR)/include -I$(AEC_DIR)/example \
         -I$(NR_DIR)/include \
         -I$(AC_DIR)/include
# AUDIO_PIPELINE_BACKEND_STR: compile-time FFT backend identity baked into
# audio_pipeline.c's AudioPipelineMemReq.backend field (F20) -- matches
# whichever libaudio_common.a variant this build links (BACKEND= above), so
# the descriptor never has to be told separately which backend produced it.
CFLAGS += -DAUDIO_PIPELINE_BACKEND_STR=\"$(BACKEND)\"
CFLAGS += $(EXTRA_CFLAGS)

# -Werror opt-in for this Makefile's OWN TUs (default off; CI can flip it
# on with `make WERROR=1`). Vendor code (audio_common's NE10/KISS) is built by
# audio_common's own Makefile (see `libs:` below), not by rules in this file,
# so no separate warning isolation is needed on this side.
WERROR ?= 0
ifeq ($(WERROR),1)
  CFLAGS += -Werror
endif

# NO_STDIO=1 (re-review R09): compiles audio_pipeline.o with
# -DAUDIO_PIPELINE_NO_STDIO, which turns every AP_LOG_ERR() diagnostic in
# audio_pipeline.c into a no-op and drops its <stdio.h> include entirely --
# see that file's "No-stdio build gate" section. This is a board-firmware
# concern (a stdio-less image, or one that forbids the stdio symbol set
# outright), so it is scoped to LIB_CFLAGS (used ONLY by the
# libaudio_pipeline.a object below) rather than the shared CFLAGS: the two
# CLIs (aec_nr_pipeline / aec_nr_pipeline_static) are host tools that always
# do their own WAV-I/O stdio regardless, and test_audio_pipeline.c prints
# its own PASS/FAIL via stdio unconditionally -- neither references
# AUDIO_PIPELINE_NO_STDIO, so there is nothing to gain (and a real loss of
# CLI diagnostics) from adding the define to their compiles.
NO_STDIO ?= 0
LIB_CFLAGS = $(CFLAGS)
ifeq ($(NO_STDIO),1)
  LIB_CFLAGS += -DAUDIO_PIPELINE_NO_STDIO
endif

# The NE10 archive carries one C++ TU (NE10's generic-radix FFT kernel); a
# plain-C pipeline binary linking against it needs the C++ runtime driver.
LINK = $(CC)
LDFLAGS_EXTRA =
ifeq ($(BACKEND),ne10)
  LINK = $(CXX)
  LDFLAGS_EXTRA = -lc++
endif

# Link order matters: pipeline.o, libaudio_pipeline.a, libaec.a, libmmse_lsa.a,
# libaudio_common.a, -lm. libaudio_pipeline.a (review F20) holds ONLY the
# pipeline TU (audio_pipeline.o) -- it is not a bundled/fat archive, so it
# still needs libaec.a/libmmse_lsa.a/libaudio_common.a linked after it, same
# as when audio_pipeline.c's logic was file-local code inside each CLI's own
# .o.
LIBS = ./libaudio_pipeline.a \
       $(AEC_DIR)/bin/libaec.a \
       $(NR_DIR)/bin/libmmse_lsa.a \
       $(AC_DIR)/bin/$(BACKEND)/libaudio_common.a
LDFLAGS = $(LIBS) -lm $(LDFLAGS_EXTRA)

# --- Hash-keyed object directory (build hygiene) ----------------------------
# This Makefile used to keep fixed .o names (aec_nr_pipeline.o,
# aec_nr_pipeline_static.o, audio_pipeline.o, test_audio_pipeline.o) directly
# in this directory (which also holds the Makefile, scripts, docs, etc.), so
# switching BACKEND or EXTRA_CFLAGS on an already-built tree needed
# `make clean-libs` by hand first or a stale, differently-flagged .o would
# silently persist. CFG_SIG hashes the exact compiler invocation that
# produces every one of this Makefile's own objects, and OBJ_DIR (below)
# embeds it directly: every distinct (BACKEND, CFLAGS, EXTRA_CFLAGS, WERROR)
# combination gets its own directory, automatically -- no manual clean ever
# required, and two differently-configured builds can run concurrently in
# this same tree without stomping each other's objects. OBJ_ROOT is the
# whole obj/ tree (every config) -- used by `clean` to nuke everything, not
# just the current config's subdirectory. The final binaries/archive
# (aec_nr_pipeline, aec_nr_pipeline_static, libaudio_pipeline.a,
# test_audio_pipeline) stay flat at the top of this directory, same as
# before: last-write-wins wholes, not config-keyed.
#
# CFG_SIG MUST be defined here, before any rule below mentions $(OBJ_DIR) (or
# a path under it, e.g. `libaudio_pipeline.a: $(OBJ_DIR)/audio_pipeline.o`):
# GNU Make expands a rule's target/prerequisite list IMMEDIATELY when that
# line is parsed, not deferred like a recipe body, so a forward reference
# would silently resolve using CFG_SIG's value at THAT point in the file
# (empty, if this line hadn't run yet) -- verified empirically with a
# minimal reproduction.
#
# There is no parse-time comparison and nothing is ever deleted to "detect"
# a config change (the previous scheme did a $(shell rm -f ...)/$(shell
# echo ...) unconditionally while the Makefile was being read, which
# mutated the filesystem even under `make -n` and raced two concurrent
# builds with different flags against each other's shared, un-keyed .o
# names). Under this scheme `make -n` touches nothing on disk (CFG_SIG is a
# pure string hash) and two differently-configured builds (e.g.
# `make BACKEND=kiss -j4 & make BACKEND=ne10 -j4 & wait`) can run
# concurrently in this same tree: each lands in its own obj/<backend>-<sig>/
# directory and never touches the other's objects. OBJ_DIR's own creation
# happens lazily via the order-only prerequisite on the compile rules below
# ($(OBJ_DIR)/%.o: ... | $(OBJ_DIR)), not at Makefile-parse time.
#
# cksum (POSIX -- present on macOS/BSD/GNU/busybox, unlike shasum which is a
# GNU coreutils extension and may be absent from slim embedded build images)
# computes the hash. On stdin it prints "<checksum> <byte-count>" with no
# filename field (verified on macOS's BSD cksum), so `cut -d' ' -f1` isolates
# just the checksum.
# NO_STDIO is folded in here (even though it only ever changes LIB_CFLAGS,
# i.e. only audio_pipeline.o's own compile) so that flipping it still lands
# in a fresh OBJ_DIR like every other axis this Makefile keys on -- without
# this, a NO_STDIO=1 build and a NO_STDIO=0 (default) build of the OTHER
# three objects (both CLIs' .o + test_audio_pipeline.o, which never see
# LIB_CFLAGS) would otherwise share one directory, which is harmless for
# THOSE objects (their recipe never varies with NO_STDIO) but breaks the
# "OBJ_DIR uniquely names a config" invariant `make audit-no-stdio` (below)
# relies on to find audio_pipeline.o's own NO_STDIO=1 build deterministically.
CFG_SIG := $(shell printf '%s' "$(CC) $(CFLAGS) $(EXTRA_CFLAGS) BACKEND=$(BACKEND) WERROR=$(WERROR) NO_STDIO=$(NO_STDIO)" | cksum | cut -d' ' -f1)

OBJ_ROOT = obj
OBJ_DIR  = $(OBJ_ROOT)/$(BACKEND)-$(CFG_SIG)
# (Note: OBJ_DIR only affects THIS Makefile's own objects -- the
# submodule/shared libs pulled in by `libs:` below are independently
# hash-keyed by their own Makefiles, which now key off the same
# BACKEND/EXTRA_CFLAGS/WERROR axes.)

.PHONY: all clean clean-libs clean-all libs test audit-no-stdio print-obj-dir

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

# $(OBJ_DIR) creation is an order-only prerequisite on the compile rules
# below (not a $(shell mkdir) at parse time, and not a plain prerequisite
# that could race a parallel compile under `make -j`) -- every rule that
# writes into it depends on it existing first.
$(OBJ_DIR):
	@mkdir -p $@

# Compile with $(CC) (always a plain-C driver, -std=gnu99 valid) then link with
# $(LINK) (c++ for BACKEND=ne10 — the NE10 archive's one C++ TU needs it). A
# C++ driver invoked directly on a .c source rejects -std=gnu99 and treats the
# file as C++, so the compile and link steps must stay separate here (mirrors
# how lib/aec and lib/nr's own Makefiles split .o-compile from $(LINK)).
$(OBJ_DIR)/aec_nr_pipeline.o: aec_nr_pipeline.c audio_pipeline.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/aec_nr_pipeline_static.o: aec_nr_pipeline_static.c audio_pipeline.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# audio_pipeline.o / libaudio_pipeline.a (review F20): the pool-sizing/
# carving/per-hop-processing library TU, linkable independently of either
# CLI -- see audio_pipeline.h. The archive holds ONLY this one object (not a
# fat archive bundling libaec/libmmse_lsa/libaudio_common); those stay
# separate libraries a consumer links alongside it, same as both CLIs do.
# Compiled with $(LIB_CFLAGS) (not the shared $(CFLAGS)) so NO_STDIO=1 only
# ever affects THIS object -- see the NO_STDIO comment near CFLAGS above.
$(OBJ_DIR)/audio_pipeline.o: audio_pipeline.c audio_pipeline.h pipeline_dims.h | $(OBJ_DIR)
	$(CC) $(LIB_CFLAGS) -c -o $@ $<

libaudio_pipeline.a: $(OBJ_DIR)/audio_pipeline.o
	ar rcs $@ $^

aec_nr_pipeline: $(OBJ_DIR)/aec_nr_pipeline.o libaudio_pipeline.a | libs
	$(LINK) -o $@ $< $(LDFLAGS)

aec_nr_pipeline_static: $(OBJ_DIR)/aec_nr_pipeline_static.o libaudio_pipeline.a | libs
	$(LINK) -o $@ $< $(LDFLAGS)

# test_audio_pipeline: standalone acceptance test for audio_pipeline.h itself
# (create-vs-init byte equality incl. a dirty/poisoned pool, destroy
# idempotence, misaligned/undersized pool rejection, sample-rate whitelist
# rejection) -- see test_audio_pipeline.c's own header for detail.
$(OBJ_DIR)/test_audio_pipeline.o: test_audio_pipeline.c audio_pipeline.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

test_audio_pipeline: $(OBJ_DIR)/test_audio_pipeline.o libaudio_pipeline.a | libs
	$(LINK) -o $@ $< $(LDFLAGS)

test: test_audio_pipeline
	./test_audio_pipeline

# print-obj-dir: prints THIS invocation's $(OBJ_DIR) and nothing else --
# a plumbing helper `audit-no-stdio` below uses (via a `$(MAKE) -s ...
# print-obj-dir` sub-call) to learn the hash-keyed directory a NO_STDIO=1
# build lands in, without duplicating CFG_SIG's cksum computation in shell.
print-obj-dir:
	@echo $(OBJ_DIR)

# audit-no-stdio (re-review R09): builds libaudio_pipeline.a with NO_STDIO=1
# (current BACKEND=$(BACKEND)) and asserts, via `nm` over audio_pipeline.o
# ITSELF (not the CLIs, not test_audio_pipeline.o -- neither is built by
# this target, and neither is gated by AUDIO_PIPELINE_NO_STDIO to begin
# with), that none of fprintf/printf/puts/fputs/stderr/__stderrp made it
# into the object. Pattern/style mirrors
# ../../audio_common/scripts/audit_alloc_symbols.sh (leading-underscore-
# tolerant regex, last-whitespace-field match against nm's NAME column, so
# it works the same under macOS/BSD nm and GNU nm). Unlike that script this
# IS a pass/fail gate (prints PASS/FAIL, non-zero exit on FAIL) -- it is the
# audit_alloc_symbols.sh idea applied to prove a NEGATIVE build property
# (these symbols are ABSENT) rather than report positive allocator-call
# provenance.
audit-no-stdio:
	@echo "audit-no-stdio: building libaudio_pipeline.a (BACKEND=$(BACKEND), NO_STDIO=1)"
	@$(MAKE) --no-print-directory BACKEND=$(BACKEND) NO_STDIO=1 libaudio_pipeline.a
	@objdir=$$($(MAKE) -s --no-print-directory BACKEND=$(BACKEND) NO_STDIO=1 print-obj-dir); \
	obj="$$objdir/audio_pipeline.o"; \
	if [ ! -f "$$obj" ]; then \
	  echo "FAIL: $$obj not found after building libaudio_pipeline.a with NO_STDIO=1"; \
	  exit 1; \
	fi; \
	echo "audit-no-stdio: nm -A $$obj | grep -E '^_?(fprintf|printf|puts|fputs|stderr|__stderrp)$$' (NAME column)"; \
	hits=$$(nm -A "$$obj" | awk '{ name = $$NF; if (name ~ /^_?(fprintf|printf|puts|fputs|stderr|__stderrp)$$/) print }'); \
	if [ -n "$$hits" ]; then \
	  echo "FAIL: forbidden stdio symbol(s) found in $$obj (NO_STDIO=1 build):"; \
	  echo "$$hits"; \
	  exit 1; \
	fi; \
	echo "PASS: no fprintf/printf/puts/fputs/stderr/__stderrp symbols in $$obj (NO_STDIO=1)"

clean:
	rm -rf $(OBJ_ROOT)
	rm -f aec_nr_pipeline aec_nr_pipeline_static libaudio_pipeline.a test_audio_pipeline

# clean-all: full nuke -- this dir's own artifacts plus all three
# submodule/shared build trees (lib/aec, lib/nr, audio_common). clean-libs is
# kept as a backward-compatible alias (same effect) since existing
# tooling/docs call it by that name.
clean-all: clean
	$(MAKE) -C $(AEC_DIR) clean
	$(MAKE) -C $(NR_DIR) clean
	$(MAKE) -C $(AC_DIR) clean

clean-libs: clean-all
