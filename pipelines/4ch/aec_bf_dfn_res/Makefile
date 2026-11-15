.PHONY: all test clean

ifeq ($(AEC_NO_ARRAY),1)
all test:
	@echo "pipelines/4ch: SKIPPED -- the four-microphone programs need the array instance; AEC_NO_ARRAY=1 builds the single-channel lib/aec only"
else
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
	  printf '%s\n' "$$out" | grep -q "model grid 16000 Hz / FFT 256 (n_erb 24, df_bins 96, lookahead 0/0), the model stream on the host grid"; \
	  sh ../../scripts/test_cli_exit_codes.sh 4ch_aec_bf_dfn_res "$$bin/4ch_aec_bf_dfn_res"
endif

clean:
	$(MAKE) -C .. clean
