# Shared stream/ helpers: the header-only stream_mem.h (StreamMemReq, carve, hash, saturate), the
# SDK-free port check stream_bind.{h,c}, which does not depend on stream_mem.h, and the header-only board
# sample helper ai3_model_io.h (row expansion, state pairing, boundary check) over stream_bind.
# Standalone: `make -C stream/common test` builds and runs the three unit tests, the C++ header compile and
# the no-stdio audit.
CC ?= cc
CXX ?= c++
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
WERROR ?= 1
ifeq ($(WERROR),1)
override CFLAGS += -Werror
endif
override CFLAGS += -ffp-contract=off -fno-math-errno -I.
BUILD_DIR ?= build

include ../../lib/deps.mk

.PHONY: all test test-stream-bind test-stream-mem test-model-io check-model-io-cxx audit-no-stdio clean

all: $(BUILD_DIR)/stream_bind.o

$(BUILD_DIR):
	@mkdir -p "$(BUILD_DIR)"

$(BUILD_DIR)/stream_bind.o: stream_bind.c stream_bind.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c stream_bind.c -o "$@"

$(BUILD_DIR)/test_stream_bind: tests/test_stream_bind.c stream_bind.c stream_bind.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) tests/test_stream_bind.c stream_bind.c -lm -o "$@"

test-stream-bind: $(BUILD_DIR)/test_stream_bind
	"$(BUILD_DIR)/test_stream_bind"

# stream_mem.h needs the shared common headers; stream_bind.* is built without them.
$(BUILD_DIR)/test_stream_mem: tests/test_stream_mem.c stream_mem.h stream_bind.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(COMMON_DIR)/include -I$(COMMON_DIR) tests/test_stream_mem.c -lm -o "$@"

test-stream-mem: $(BUILD_DIR)/test_stream_mem
	"$(BUILD_DIR)/test_stream_mem"

$(BUILD_DIR)/test_model_io: tests/test_model_io.c ai3_model_io.h stream_bind.c stream_bind.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) tests/test_model_io.c stream_bind.c -o "$@"

test-model-io: $(BUILD_DIR)/test_model_io
	"$(BUILD_DIR)/test_model_io"

# The header must also compile as C++ (the samples may be built by a C++ driver).
check-model-io-cxx: ai3_model_io.h stream_bind.h | $(BUILD_DIR)
	@printf '#include "ai3_model_io.h"\nint probe(const Ai3ModelIoRow *r, Ai3ModelIoConfig *c) {\n  struct W { int input_count, output_count; struct { const float *data; size_t elements; unsigned w, h, c, b; } inputs[1], outputs[1]; } wk = {0, 0, {{0, 0, 0, 0, 0, 0}}, {{0, 0, 0, 0, 0, 0}}};\n  int rc = ai3_model_io_expand(r, c);\n  AI3_MODEL_IO_CHECK_WORK(rc, c, &wk);\n  return rc + ai3_model_io_frames(256, 128) + ai3_model_io_state_count(c) + (int)ai3_model_io_input_bytes(c, 0, 4);\n}\n' > "$(BUILD_DIR)/model_io_cxx_probe.cpp"
	$(CXX) -std=c++11 -Wall -Wextra -Wpedantic -Werror -I. -fsyntax-only "$(BUILD_DIR)/model_io_cxx_probe.cpp"
	@echo "check-model-io-cxx: PASS"

# stream_bind.o must not reference any printf-family symbol; the positive control proves the scan
# would see one.
audit-no-stdio: $(BUILD_DIR)/stream_bind.o | $(BUILD_DIR)
	@if nm "$(BUILD_DIR)/stream_bind.o" | grep -Eq '(^|[ _])(v?[sf]?n?printf|puts|putchar|fputs|fwrite|stdout|stderr)$$'; then \
	  echo "audit-no-stdio: stream_bind.o references stdio"; exit 1; fi
	@printf 'int printf(const char *, ...);\nint probe(void){return printf("x");}\n' > "$(BUILD_DIR)/stdio_probe.c"
	@$(CC) -c "$(BUILD_DIR)/stdio_probe.c" -o "$(BUILD_DIR)/stdio_probe.o"
	@if ! nm "$(BUILD_DIR)/stdio_probe.o" | grep -Eq '(^|[ _])(v?[sf]?n?printf|puts|putchar|fputs|fwrite|stdout|stderr)$$'; then \
	  echo "audit-no-stdio: positive control did not match"; exit 1; fi
	@echo "audit-no-stdio: PASS"

test: test-stream-bind test-stream-mem test-model-io check-model-io-cxx audit-no-stdio

clean:
	rm -rf "$(BUILD_DIR)"
