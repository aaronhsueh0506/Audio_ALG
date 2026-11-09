.PHONY: all test clean

all:
	$(MAKE) -C .. 4ch_aec_bf_dfn_res

test: all
	$(MAKE) -C .. test_4ch_dfn
	set -e; bin="$$($(MAKE) -s -C .. print-bin-dir)"; \
	  "$$bin/test_4ch_dfn"; \
	  "$$bin/4ch_aec_bf_dfn_res"; \
	  "$$bin/4ch_aec_bf_dfn_res" --delay-num-filters 1 --cng --atten-lim 12; \
	  "$$bin/4ch_aec_bf_dfn_res" --sample-rate 48000; \
	  out="$$("$$bin/4ch_aec_bf_dfn_res" --ai-sample-rate 16000 --ai-fft-size 256)"; \
	  printf '%s\n' "$$out" | grep -q "model grid 16000 Hz / FFT 256 (n_erb 24, df_bins 96, lookahead 0/0), the stage on the host grid"; \
	  rc=0; "$$bin/4ch_aec_bf_dfn_res" --ai-sample-rate 32000 --ai-fft-size 512 >/dev/null 2>&1 || rc=$$?; test "$$rc" -eq 2; \
	  rc=0; "$$bin/4ch_aec_bf_dfn_res" --sample-rate 16000Hz >/dev/null 2>&1 || rc=$$?; test "$$rc" -eq 2; \
	  rc=0; "$$bin/4ch_aec_bf_dfn_res" --sample-rate 8000 >/dev/null 2>&1 || rc=$$?; test "$$rc" -eq 2; \
	  rc=0; "$$bin/4ch_aec_bf_dfn_res" --atten-lim 12dB >/dev/null 2>&1 || rc=$$?; test "$$rc" -eq 2

clean:
	$(MAKE) -C .. clean
