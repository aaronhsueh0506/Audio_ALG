.PHONY: all test clean

all:
	$(MAKE) -C ../4ch_aec_bf_nr_res 4ch_aec_bf_dfn_res

test: all
	$(MAKE) -C ../4ch_aec_bf_nr_res test_4aec_dfn_res
	set -e; bin="$$($(MAKE) -s -C ../4ch_aec_bf_nr_res print-bin-dir)"; \
	  "$$bin/test_4aec_dfn_res"; \
	  "$$bin/4ch_aec_bf_dfn_res"; \
	  "$$bin/4ch_aec_bf_dfn_res" --delay-num-filters 1 --cng --atten-lim 12; \
	  "$$bin/4ch_aec_bf_dfn_res" --sample-rate 48000; \
	  rc=0; "$$bin/4ch_aec_bf_dfn_res" --sample-rate 16000Hz >/dev/null 2>&1 || rc=$$?; test "$$rc" -eq 2; \
	  rc=0; "$$bin/4ch_aec_bf_dfn_res" --sample-rate 8000 >/dev/null 2>&1 || rc=$$?; test "$$rc" -eq 2; \
	  rc=0; "$$bin/4ch_aec_bf_dfn_res" --atten-lim 12dB >/dev/null 2>&1 || rc=$$?; test "$$rc" -eq 2

clean:
	$(MAKE) -C ../4ch_aec_bf_nr_res clean
