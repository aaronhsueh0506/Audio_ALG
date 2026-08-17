.PHONY: all test clean

all:
	$(MAKE) -C ../4ch_aec_bf_nr_res 4ch_alignulcnet

test: all
	$(MAKE) -C ../4ch_aec_bf_nr_res test_audio_pipeline_4ch_ulcnet
	set -e; bin="$$($(MAKE) -s -C ../4ch_aec_bf_nr_res print-bin-dir)"; \
	  "$$bin/test_audio_pipeline_4ch_ulcnet"; \
	  "$$bin/4ch_alignulcnet"; \
	  "$$bin/4ch_alignulcnet" --delay-num-filters 1

clean:
	$(MAKE) -C ../4ch_aec_bf_nr_res clean
