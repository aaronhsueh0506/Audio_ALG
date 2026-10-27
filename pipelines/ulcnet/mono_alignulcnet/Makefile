.PHONY: all test clean

all:
	$(MAKE) -C ../.. mono_alignulcnet

test: all
	$(MAKE) -C ../.. test_mono_ulcnet \
	  test_ulcnet_accelerator_adapter
	set -e; bin="$$($(MAKE) -s -C ../.. print-bin-dir)"; \
	  "$$bin/test_mono_ulcnet"; \
	  "$$bin/test_ulcnet_accelerator_adapter"; \
	  "$$bin/mono_alignulcnet"; \
	  "$$bin/mono_alignulcnet" --delay-num-filters 1; \
	  rc=0; "$$bin/mono_alignulcnet" --delay-num-filters 3x >/dev/null 2>&1 || rc=$$?; test "$$rc" -eq 2; \
	  rc=0; "$$bin/mono_alignulcnet" --delay-mode fixed --fixed-delay 12x >/dev/null 2>&1 || rc=$$?; test "$$rc" -eq 2

clean:
	$(MAKE) -C ../.. clean
