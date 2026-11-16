# Shared stream/ helpers: the header-only stream_mem.h (StreamMemReq, carve, hash, saturate), the
# SDK-free port check stream_bind.{h,c}, which does not depend on stream_mem.h, the header-only board
# sample helper graph_io.h (row expansion, state pairing, boundary check) over it, and the header-only run
# contract stream_run.h (StreamWork, STREAM_RUN_* statuses), and the header-only range check stream_range.h
# (stream_plane_within over the audiolib kernel; included only by the stream .c files).
# Standalone: `make -C stream/common test` builds and runs the five unit tests, the C++ header compile, the
# customer header set check and the no-stdio audit.
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

.PHONY: all test test-stream-bind test-stream-mem test-graph-io test-stream-run test-stream-range check-stream-run check-graph-io-cxx check-public-headers audit-no-stdio clean

all: $(BUILD_DIR)/stream_bind.o

$(BUILD_DIR):
	@mkdir -p "$(BUILD_DIR)"

$(BUILD_DIR)/stream_bind.o: stream_bind.c stream_bind.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c stream_bind.c -o "$@"

TEST_WIRINGS = tests/stream_test_wirings.h stream_bind.h

$(BUILD_DIR)/test_stream_bind: tests/test_stream_bind.c stream_bind.c stream_bind.h $(TEST_WIRINGS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) tests/test_stream_bind.c stream_bind.c -lm -o "$@"

test-stream-bind: $(BUILD_DIR)/test_stream_bind
	"$(BUILD_DIR)/test_stream_bind"

# stream_mem.h needs the shared common headers; stream_bind.* is built without them.
$(BUILD_DIR)/test_stream_mem: tests/test_stream_mem.c stream_mem.h stream_bind.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(COMMON_DIR)/include -I$(COMMON_DIR) tests/test_stream_mem.c -lm -o "$@"

test-stream-mem: $(BUILD_DIR)/test_stream_mem
	"$(BUILD_DIR)/test_stream_mem"

$(BUILD_DIR)/test_graph_io: tests/test_graph_io.c graph_io.h stream_bind.c stream_bind.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) tests/test_graph_io.c stream_bind.c -o "$@"

test-graph-io: $(BUILD_DIR)/test_graph_io
	"$(BUILD_DIR)/test_graph_io"

# stream_run.h: constants, layout and the work type under GRAPH_IO_CHECK_WORK.
$(BUILD_DIR)/test_stream_run: tests/test_stream_run.c stream_run.h graph_io.h stream_bind.c stream_bind.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) tests/test_stream_run.c stream_bind.c -o "$@"

test-stream-run: $(BUILD_DIR)/test_stream_run
	"$(BUILD_DIR)/test_stream_run"

# stream_range.h: the shared range predicate over the audiolib kernel.
$(BUILD_DIR)/test_stream_range: tests/test_stream_range.c stream_range.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(COMMON_DIR)/include -I$(COMMON_DIR) tests/test_stream_range.c -lm -o "$@"

test-stream-range: $(BUILD_DIR)/test_stream_range
	"$(BUILD_DIR)/test_stream_range"

# stream_run.h as the only include: C99 -Wpedantic object, and a C++11 program that is compiled and run.
check-stream-run: stream_run.h tests/stream_run_c99_probe.c tests/stream_run_cxx_probe.cpp | $(BUILD_DIR)
	$(CC) -std=c99 -Wall -Wextra -Wpedantic -Werror -I. -c tests/stream_run_c99_probe.c -o "$(BUILD_DIR)/stream_run_c99_probe.o"
	$(CXX) -std=c++11 -Wall -Wextra -Wpedantic -Werror -I. tests/stream_run_cxx_probe.cpp -o "$(BUILD_DIR)/stream_run_cxx_probe"
	"$(BUILD_DIR)/stream_run_cxx_probe"
	@echo "check-stream-run: PASS"

# The header must also compile as C++ (the samples may be built by a C++ driver).
check-graph-io-cxx: graph_io.h stream_bind.h tests/graph_io_cxx_probe.cpp
	$(CXX) -std=c++11 -Wall -Wextra -Wpedantic -Werror -I. -fsyntax-only tests/graph_io_cxx_probe.cpp
	@echo "check-graph-io-cxx: PASS"

# The customer header set compiles as C99 -Wpedantic and as C++11 with only the documented include roots, and
# every function it declares keeps C linkage (tests/check_public_headers.sh).
check-public-headers: check-graph-io-cxx tests/check_public_headers.sh | $(BUILD_DIR)
	sh tests/check_public_headers.sh "$(CC)" "$(CXX)" "$(COMMON_DIR)" "$(BUILD_DIR)/public_headers"

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

test: test-stream-bind test-stream-mem test-graph-io test-stream-run test-stream-range check-stream-run check-public-headers audit-no-stdio

clean:
	rm -rf "$(BUILD_DIR)"
