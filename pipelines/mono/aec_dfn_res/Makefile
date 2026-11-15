.PHONY: all test clean

all:
	$(MAKE) -C ../.. mono_aec_dfn_res

test: all
	$(MAKE) -C ../.. test_dfn_res_stage test_dfn_rate_bridge test_mono_dfn
	set -e; bin="$$($(MAKE) -s -C ../.. print-bin-dir)"; \
	  "$$bin/test_dfn_res_stage"; \
	  "$$bin/test_dfn_rate_bridge"; \
	  "$$bin/test_mono_dfn"; \
	  "$$bin/mono_aec_dfn_res"; \
	  "$$bin/mono_aec_dfn_res" --delay-num-filters 1 --cng --atten-lim 12; \
	  "$$bin/mono_aec_dfn_res" --sample-rate 48000; \
	  "$$bin/mono_aec_dfn_res" --sample-rate 8000; \
	  out="$$("$$bin/mono_aec_dfn_res" --ai-sample-rate 16000 --ai-fft-size 256)"; \
	  printf '%s\n' "$$out" | grep -q "model grid 16000 Hz / FFT 256 (n_erb 24, df_bins 96, lookahead 0/0), the model stream on the host grid"; \
	  sh ../../scripts/test_cli_exit_codes.sh mono_aec_dfn_res "$$bin/mono_aec_dfn_res"

clean:
	$(MAKE) -C ../.. clean
