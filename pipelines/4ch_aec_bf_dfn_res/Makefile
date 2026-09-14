.PHONY: all test clean

all:
	$(MAKE) -C ../4ch_aec_bf_nr_res 4ch_aec_bf_dfn_res

test: all
	$(MAKE) -C ../4ch_aec_bf_nr_res test_4aec_dfn_res
	set -e; bin="$$($(MAKE) -s -C ../4ch_aec_bf_nr_res print-bin-dir)"; \
	  "$$bin/test_4aec_dfn_res"; \
	  "$$bin/4ch_aec_bf_dfn_res"; \
	  "$$bin/4ch_aec_bf_dfn_res" --delay-num-filters 1 --cng --atten-lim 12; \
	  "$$bin/4ch_aec_bf_dfn_res" --sample-rate 48000

clean:
	$(MAKE) -C ../4ch_aec_bf_nr_res clean
