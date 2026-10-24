.PHONY: all test clean

all:
	$(MAKE) -C .. aec_nr_pipeline aec_nr_pipeline_static

test: all
	$(MAKE) -C .. test_mono_chain example-adapter
	bin="$$($(MAKE) -s -C .. print-bin-dir)"; obj="$$($(MAKE) -s -C .. print-obj-dir)"; \
	  "$$bin/test_mono_chain" "$$bin" "$$obj"

clean:
	$(MAKE) -C .. clean
