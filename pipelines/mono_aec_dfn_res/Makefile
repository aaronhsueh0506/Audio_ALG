.PHONY: all test clean

all:
	$(MAKE) -C .. mono_aec_dfn_res

test: all
	$(MAKE) -C .. test_dfn_res_stage test_dfn_rate_bridge test_audio_pipeline_dfn
	set -e; bin="$$($(MAKE) -s -C .. print-bin-dir)"; \
	  "$$bin/test_dfn_res_stage"; \
	  "$$bin/test_dfn_rate_bridge"; \
	  "$$bin/test_audio_pipeline_dfn"; \
	  "$$bin/mono_aec_dfn_res"; \
	  "$$bin/mono_aec_dfn_res" --delay-num-filters 1 --cng --atten-lim 12; \
	  "$$bin/mono_aec_dfn_res" --sample-rate 48000; \
	  "$$bin/mono_aec_dfn_res" --sample-rate 8000; \
	  rc=0; "$$bin/mono_aec_dfn_res" --sample-rate 16000Hz >/dev/null 2>&1 || rc=$$?; test "$$rc" -eq 2; \
	  rc=0; "$$bin/mono_aec_dfn_res" --atten-lim 12dB >/dev/null 2>&1 || rc=$$?; test "$$rc" -eq 2

clean:
	$(MAKE) -C .. clean
