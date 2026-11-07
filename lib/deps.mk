# Audio_ALG/lib/deps.mk
#
# Single source of truth for where this repo's three external producers -- AEC, NR and audiolib --
# are read from, and at which commit each is pinned (lib/aec.pin, lib/nr.pin, lib/audiolib.pin).
# Included by every Makefile that needs AEC_DIR / NR_DIR / AUDIOLIB_DIR / COMMON_DIR
# (pipelines/Makefile, pipelines/4ch_aec_bf_nr_res/Makefile, AINR/Makefile,
# AINR/RNNoise-ERB/Makefile, AIAEC/Makefile) instead of each hardcoding its own relative path.
# Every value here is `?=`, so a caller that already set one on its command line or in its
# environment keeps that value untouched; this file only supplies the default.
#
# Default resolution, as of the dependency-path-convergence pass: the sibling dev checkouts this
# repo is normally developed alongside -- <Audio_ALG root>/../AEC, ../NR, ../audiolib (the
# "Recommended sole dev working directory" column of the dependency table: SE/AEC, SE/NR,
# SE/audiolib, each using its own c_impl/ for AEC and NR). The `lib/aec` and `lib/nr` git
# submodules (see .gitmodules) still exist, still carry a gitlink and still build standalone; they
# are simply no longer where a default build looks first. Point AEC_DIR/NR_DIR back at them
# explicitly (`make AEC_DIR=$(ALG_ROOT)/lib/aec/c_impl ...`) for a build that must use the
# submodule checkout instead.
#
# This file does not fetch or validate anything by itself outside its own two targets below:
#   make -f lib/deps.mk bootstrap-deps    # clone whichever of the three sibling repos is missing,
#                                          # at its pinned commit (AEC_URL/NR_URL/AUDIOLIB_URL,
#                                          # each overridable -- e.g. to a local path, for an
#                                          # offline/scratch bootstrap)
#   make -f lib/deps.mk check-deps-pins   # report (not enforce -- pipelines/Makefile's check-pins
#                                          # is what fails a build) each of AEC_DIR/NR_DIR/
#                                          # AUDIOLIB_DIR against its pin, and, while they still
#                                          # exist, the lib/aec and lib/nr gitlinks against the same
#                                          # pins (the step-5 "gitlink migration" readiness gate)
#   make -f lib/deps.mk print-deps-sh     # emit the four resolved directories as
#                                          # `eval`-able `KEY=value` shell assignments, so a shell
#                                          # script resolves them the same way a Makefile does
#                                          # instead of reimplementing the resolution itself

# This file's own absolute directory, computed before the three .pin files are included: an
# `include` changes $(MAKEFILE_LIST)'s last word to the included file, so capturing
# $(lastword $(MAKEFILE_LIST)) has to happen first, with `:=` (not `=`) so it is captured once,
# here, rather than re-evaluated lazily after the last word has moved on.
DEPS_MK_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
AUDIO_ALG_ROOT := $(patsubst %/,%,$(abspath $(DEPS_MK_DIR)/..))

include $(DEPS_MK_DIR)/aec.pin
include $(DEPS_MK_DIR)/nr.pin
include $(DEPS_MK_DIR)/audiolib.pin

AEC_REPO_DIR  ?= $(abspath $(AUDIO_ALG_ROOT)/../AEC)
NR_REPO_DIR   ?= $(abspath $(AUDIO_ALG_ROOT)/../NR)
AUDIOLIB_DIR  ?= $(abspath $(AUDIO_ALG_ROOT)/../audiolib)

AEC_DIR     ?= $(AEC_REPO_DIR)/c_impl
NR_DIR      ?= $(NR_REPO_DIR)/c_impl
COMMON_DIR  ?= $(AUDIOLIB_DIR)/common

.PHONY: print-deps-sh bootstrap-deps check-deps-pins

print-deps-sh:
	@printf 'AEC_DIR=%s\nNR_DIR=%s\nAUDIOLIB_DIR=%s\nCOMMON_DIR=%s\n' '$(AEC_DIR)' '$(NR_DIR)' '$(AUDIOLIB_DIR)' '$(COMMON_DIR)'

# Clones each of AEC_REPO_DIR / NR_REPO_DIR / AUDIOLIB_DIR from its pinned URL at its pinned
# commit, but only when that directory does not already exist -- an existing checkout at a
# different commit is left alone and reported (as a mismatch) by check-deps-pins, per step 4's
# "if an existing checkout is at a different version, report expected vs actual and leave the
# developer's working tree alone". AEC_URL/NR_URL/AUDIOLIB_URL may be overridden (e.g. to a local
# path or an already-cloned repo) to bootstrap without reaching the network.
bootstrap-deps:
	@if [ -d '$(AEC_REPO_DIR)' ]; then \
	  echo "bootstrap-deps: AEC_REPO_DIR=$(AEC_REPO_DIR) already exists, left alone"; \
	else \
	  echo "bootstrap-deps: cloning AEC $(AEC_COMMIT) from $(AEC_URL) -> $(AEC_REPO_DIR)"; \
	  git clone --quiet '$(AEC_URL)' '$(AEC_REPO_DIR)' && \
	  git -C '$(AEC_REPO_DIR)' checkout --quiet '$(AEC_COMMIT)'; \
	fi
	@if [ -d '$(NR_REPO_DIR)' ]; then \
	  echo "bootstrap-deps: NR_REPO_DIR=$(NR_REPO_DIR) already exists, left alone"; \
	else \
	  echo "bootstrap-deps: cloning NR $(NR_COMMIT) from $(NR_URL) -> $(NR_REPO_DIR)"; \
	  git clone --quiet '$(NR_URL)' '$(NR_REPO_DIR)' && \
	  git -C '$(NR_REPO_DIR)' checkout --quiet '$(NR_COMMIT)'; \
	fi
	@if [ -d '$(AUDIOLIB_DIR)' ]; then \
	  echo "bootstrap-deps: AUDIOLIB_DIR=$(AUDIOLIB_DIR) already exists, left alone"; \
	else \
	  echo "bootstrap-deps: cloning audiolib $(AUDIOLIB_COMMIT) from $(AUDIOLIB_URL) -> $(AUDIOLIB_DIR)"; \
	  git clone --quiet '$(AUDIOLIB_URL)' '$(AUDIOLIB_DIR)' && \
	  git -C '$(AUDIOLIB_DIR)' checkout --quiet '$(AUDIOLIB_COMMIT)'; \
	fi

# Report (never fail the build -- pipelines/Makefile's check-pins target does that, against
# whichever of these a particular build actually resolved AEC_DIR/NR_DIR/AUDIOLIB_DIR to, honoring
# any override) each pinned dependency against the repo it names, AND, while the lib/aec and
# lib/nr submodules still exist, their gitlinks against the same pins -- this second half is the
# step-5 readiness gate: step 5 (migrating the two gitlinks onto these pins, and retiring the
# now-redundant one of the two) is not proposed until every row below agrees.
check-deps-pins:
	@ok=1; \
	have_aec="$$(git -C '$(AEC_REPO_DIR)' rev-parse HEAD 2>/dev/null || echo unknown)"; \
	if [ "$$have_aec" = "$(AEC_COMMIT)" ]; then echo "check-deps-pins: AEC ($(AEC_REPO_DIR)) $$have_aec matches lib/aec.pin"; \
	else echo "check-deps-pins: FAIL AEC ($(AEC_REPO_DIR)) is $$have_aec, lib/aec.pin expects $(AEC_COMMIT)" >&2; ok=0; fi; \
	have_nr="$$(git -C '$(NR_REPO_DIR)' rev-parse HEAD 2>/dev/null || echo unknown)"; \
	if [ "$$have_nr" = "$(NR_COMMIT)" ]; then echo "check-deps-pins: NR ($(NR_REPO_DIR)) $$have_nr matches lib/nr.pin"; \
	else echo "check-deps-pins: FAIL NR ($(NR_REPO_DIR)) is $$have_nr, lib/nr.pin expects $(NR_COMMIT)" >&2; ok=0; fi; \
	have_ac="$$(git -C '$(AUDIOLIB_DIR)' rev-parse HEAD 2>/dev/null || echo unknown)"; \
	if [ "$$have_ac" = "$(AUDIOLIB_COMMIT)" ]; then echo "check-deps-pins: audiolib ($(AUDIOLIB_DIR)) $$have_ac matches lib/audiolib.pin"; \
	else echo "check-deps-pins: FAIL audiolib ($(AUDIOLIB_DIR)) is $$have_ac, lib/audiolib.pin expects $(AUDIOLIB_COMMIT)" >&2; ok=0; fi; \
	if [ -e '$(AUDIO_ALG_ROOT)/lib/aec/.git' ]; then \
	  gl_rec="$$(git -C '$(AUDIO_ALG_ROOT)' ls-files -s -- lib/aec 2>/dev/null | awk '$$3 == 0 {print $$2}')"; \
	  gl_have="$$(git -C '$(AUDIO_ALG_ROOT)/lib/aec' rev-parse HEAD 2>/dev/null || echo unknown)"; \
	  if [ "$$gl_rec" = "$$gl_have" ] && [ "$$gl_have" = "$(AEC_COMMIT)" ]; then \
	    echo "check-deps-pins: lib/aec gitlink $$gl_have matches lib/aec.pin (step 5 ready on this axis)"; \
	  else \
	    echo "check-deps-pins: NOTE lib/aec gitlink is $$gl_have (index records $$gl_rec), lib/aec.pin expects $(AEC_COMMIT) -- step 5 not ready"; \
	  fi; \
	else \
	  echo "check-deps-pins: lib/aec submodule not initialized (not needed for the default sibling build)"; \
	fi; \
	if [ -e '$(AUDIO_ALG_ROOT)/lib/nr/.git' ]; then \
	  gl_rec="$$(git -C '$(AUDIO_ALG_ROOT)' ls-files -s -- lib/nr 2>/dev/null | awk '$$3 == 0 {print $$2}')"; \
	  gl_have="$$(git -C '$(AUDIO_ALG_ROOT)/lib/nr' rev-parse HEAD 2>/dev/null || echo unknown)"; \
	  if [ "$$gl_rec" = "$$gl_have" ] && [ "$$gl_have" = "$(NR_COMMIT)" ]; then \
	    echo "check-deps-pins: lib/nr gitlink $$gl_have matches lib/nr.pin (step 5 ready on this axis)"; \
	  else \
	    echo "check-deps-pins: NOTE lib/nr gitlink is $$gl_have (index records $$gl_rec), lib/nr.pin expects $(NR_COMMIT) -- step 5 not ready"; \
	  fi; \
	else \
	  echo "check-deps-pins: lib/nr submodule not initialized (not needed for the default sibling build)"; \
	fi; \
	[ "$$ok" = 1 ] || exit 1

# Every includer puts `include lib/deps.mk` before its own first real target, so without this,
# GNU Make's "first target in the makefile becomes .DEFAULT_GOAL" rule would silently make
# print-deps-sh the default instead of the includer's own (all / _cfg_guard / ...). Clearing
# .DEFAULT_GOAL makes Make pick it up again from the next target the includer defines.
.DEFAULT_GOAL :=
