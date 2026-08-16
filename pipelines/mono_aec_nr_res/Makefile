.PHONY: all test clean

all:
	$(MAKE) -C .. aec_nr_pipeline aec_nr_pipeline_static

test: all
	$(MAKE) -C .. test_audio_pipeline example-adapter
	bin="$$($(MAKE) -s -C .. print-bin-dir)"; \
	  "$$bin/test_audio_pipeline"

clean:
	$(MAKE) -C .. clean
