# RNNoise-ERB table drift-guard wiring (round-4 review P2-4: the two-layer
# B09 drift contract in test_rnnoise_tables.c used to be a manual compile
# command documented only in that file's header -- now it is a make target a
# CI job or a human can invoke as one word).
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
# -ffp-contract=off matches the repo-wide FP policy (round-3 review B04) and
# the command documented in test_rnnoise_tables.c's own header.

CC ?= cc
CFLAGS ?= -O2 -ffp-contract=off -Wall -Wextra

BUILD_DIR = build

.PHONY: test-tables clean

test-tables: $(BUILD_DIR)/test_rnnoise_tables $(BUILD_DIR)/test_rnnoise_tables_portable
	@echo "== rnnoise tables drift-guard: layer 1 (canonical, byte-exact) =="
	./$(BUILD_DIR)/test_rnnoise_tables
	@echo "== rnnoise tables drift-guard: layer 2 (portable, math contracts) =="
	./$(BUILD_DIR)/test_rnnoise_tables_portable

$(BUILD_DIR)/test_rnnoise_tables: test_rnnoise_tables.c rnnoise_tables_gen.h process.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ test_rnnoise_tables.c -lm

$(BUILD_DIR)/test_rnnoise_tables_portable: test_rnnoise_tables.c rnnoise_tables_gen.h process.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -DRNN_TABLES_PORTABLE -o $@ test_rnnoise_tables.c -lm

clean:
	rm -rf $(BUILD_DIR)
