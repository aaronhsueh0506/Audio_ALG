.PHONY: all test clean

ifeq ($(AEC_NO_ARRAY),1)
all test:
	@echo "pipelines/4ch: SKIPPED -- the four-microphone programs need the array instance; AEC_NO_ARRAY=1 builds the single-channel lib/aec only"
else
all:
	$(MAKE) -C .. 4ch_alignulcnet

test: all
	$(MAKE) -C .. test_4ch_ulcnet
	set -e; bin="$$($(MAKE) -s -C .. print-bin-dir)"; \
	  "$$bin/test_4ch_ulcnet"; \
	  "$$bin/4ch_alignulcnet"; \
	  "$$bin/4ch_alignulcnet" --delay-num-filters 1; \
	  "$$bin/4ch_alignulcnet" --sample-rate 48000; \
	  "$$bin/4ch_alignulcnet" --sample-rate 48000 --delay-num-filters 1; \
	  rc=0; "$$bin/4ch_alignulcnet" --sample-rate 44100 >/dev/null 2>&1 || rc=$$?; test "$$rc" -eq 2; \
	  rc=0; "$$bin/4ch_alignulcnet" --sample-rate 16000x >/dev/null 2>&1 || rc=$$?; test "$$rc" -eq 2; \
	  rc=0; "$$bin/4ch_alignulcnet" --delay-num-filters 3x >/dev/null 2>&1 || rc=$$?; test "$$rc" -eq 2; \
	  rc=0; "$$bin/4ch_alignulcnet" --delay-mode fixed --fixed-delay 12x >/dev/null 2>&1 || rc=$$?; test "$$rc" -eq 2
endif

clean:
	$(MAKE) -C .. clean
