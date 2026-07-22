# RNNoise-ERB table drift-guard wiring (round-4 review P2-4, hardened by
# round-5 review P2: the original flat build/ had no compiler/flags identity,
# so `make CC=false CFLAGS=... test-tables` silently re-ran a STALE binary
# from an earlier config -- reported PASS without ever invoking CC=false --
# and a command-line CFLAGS= could replace -ffp-contract=off outright).
#
#   make test-tables    build + run BOTH layers and fail on any FAIL:
#                         1. canonical (default mode): byte-exact memcmp of
#                            rnnoise_tables_gen.h against an independent
#                            recompute -- the pinned-toolchain contract.
#                         2. portable (-DRNN_TABLES_PORTABLE): mathematical-
#                            contract checks + measured-then-margined ULP
#                            bounds (256 ULP recompute-vs-table garbage
#                            detector; 4096 ULP Hann mirror-pair symmetry --
#                            two DIFFERENT bounds for two different checks,
#                            see the test source).
#
# Same discipline as the four big Makefiles (lightweight version):
#   - CFLAGS/FP_POLICY cannot be overridden on the command line (origin
#     gate); EXTRA_CFLAGS is the supported hook, folded in BEFORE the
#     repo-pinned -ffp-contract=off (appended last, conflict-rejected).
#   - Binaries live in build/<cfg-sig>/ keyed by CC+CFLAGS+EXTRA_CFLAGS, with
#     a config.manifest collision guard -- a config change always lands in a
#     fresh directory and must genuinely compile there; a stale binary from
#     another config can never satisfy this target again.

CC ?= cc
CFLAGS = -O2 -Wall -Wextra
CFLAGS += $(EXTRA_CFLAGS)

# --- Command-line override rejection (round-5; same design as the four main
# Makefiles' round-4 P1-1 gate -- see audio_common's Makefile). A
# command-line CFLAGS=/FP_POLICY= would silently drop the FP policy below.
$(foreach v,CFLAGS FP_POLICY,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently drop this Makefile's own flag appends (FP policy); use EXTRA_CFLAGS instead)))

# FP-policy conflict detection + last-position append (round-3 B04 pattern).
ifneq ($(findstring -Ofast,$(CFLAGS)),)
$(error FP policy conflict: EXTRA_CFLAGS contains -Ofast; this repo pins -ffp-contract=off)
endif
ifneq ($(findstring -ffast-math,$(CFLAGS)),)
$(error FP policy conflict: EXTRA_CFLAGS contains -ffast-math; this repo pins -ffp-contract=off)
endif
ifneq ($(findstring -ffp-contract=,$(CFLAGS)),)
$(error FP policy conflict: EXTRA_CFLAGS contains -ffp-contract=; this repo pins -ffp-contract=off)
endif
FP_POLICY := -ffp-contract=off
CFLAGS    += $(FP_POLICY)

# Config-keyed build dir + collision guard (lightweight _cfg_guard).
CFG_SIG_PAYLOAD := CC=$(CC) CFLAGS=$(CFLAGS) EXTRA_CFLAGS=$(EXTRA_CFLAGS)
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)
BUILD_DIR = build/$(CFG_SIG)

.PHONY: test test-tables test-features test-feature-contract test-feature-python test-loss-python clean _cfg_guard
_cfg_guard:
	@mkdir -p $(BUILD_DIR)
	@m="$(BUILD_DIR)/config.manifest"; \
	if [ -f "$$m" ]; then \
	  old="$$(cat "$$m")"; \
	  if [ "$$old" != "$(CFG_SIG_PAYLOAD)" ]; then \
	    echo "FATAL: CFG_SIG collision or coverage gap in $$m" >&2; \
	    echo "  stored : $$old" >&2; \
	    echo "  current: $(CFG_SIG_PAYLOAD)" >&2; \
	    exit 1; \
	  fi; \
	else \
	  tmp="$$m.tmp.$$$$"; printf '%s\n' "$(CFG_SIG_PAYLOAD)" > "$$tmp"; mv -f "$$tmp" "$$m"; \
	fi

test-tables: $(BUILD_DIR)/test_rnnoise_tables $(BUILD_DIR)/test_rnnoise_tables_portable
	@echo "== rnnoise tables drift-guard: layer 1 (canonical, byte-exact) =="
	./$(BUILD_DIR)/test_rnnoise_tables
	@echo "== rnnoise tables drift-guard: layer 2 (portable, math contracts) =="
	./$(BUILD_DIR)/test_rnnoise_tables_portable

test: test-tables test-feature-contract test-features

test-feature-contract:
	@echo "== rnnoise config/Python/C feature contract =="
	python3 test_feature_contract.py

test-features: $(BUILD_DIR)/test_rnnoise_features
	@echo "== rnnoise feature-state contract =="
	./$(BUILD_DIR)/test_rnnoise_features

# Requires the training Python environment (torch + numpy).  Kept separate from
# `make test` so the dependency-free embedded C checks remain runnable on hosts
# that intentionally do not install PyTorch.
test-feature-python:
	python3 test_feature_norm.py
	python3 test_python_c_features.py

test-loss-python:
	python3 test_loss.py

$(BUILD_DIR)/test_rnnoise_tables: test_rnnoise_tables.c rnnoise_tables_gen.h process.h | _cfg_guard
	$(CC) $(CFLAGS) -o $@ test_rnnoise_tables.c -lm

$(BUILD_DIR)/test_rnnoise_tables_portable: test_rnnoise_tables.c rnnoise_tables_gen.h process.h | _cfg_guard
	$(CC) $(CFLAGS) -DRNN_TABLES_PORTABLE -o $@ test_rnnoise_tables.c -lm

$(BUILD_DIR)/test_rnnoise_features: test_rnnoise_features.c process.c process.h rnnoise_tables_gen.h | _cfg_guard
	$(CC) $(CFLAGS) -o $@ test_rnnoise_features.c process.c -lm

clean:
	rm -rf build
