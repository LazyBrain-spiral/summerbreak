# Run inside WSL2 / Linux with the drone3d env active (see scripts/setup_env.sh).
PY ?= python
SYNTH ?= test_data/synth3d_pass

.PHONY: env check test test-slow synth synth-run demo

env:
	bash scripts/setup_env.sh

check:
	$(PY) scripts/check_env.py --profile live

test:
	$(PY) -m pytest tests/v3 -q -m "not slow"

test-slow:
	$(PY) -m pytest tests/v3 -q -m slow

synth:
	$(PY) tools/synth3d.py --out $(SYNTH) --seconds 16 --fps 10

synth-run:
	$(PY) -m src.pipeline --config configs/synth_oracle.yaml --video $(SYNTH)/flight.mp4 \
	    --telemetry $(SYNTH)/flight.srt --gt $(SYNTH) --run-id synth_oracle
	$(PY) eval/synth_eval.py --run runs/synth_oracle --gt $(SYNTH)

demo:
	$(PY) -m src.pipeline --config configs/live.yaml --video $(VIDEO) --telemetry $(SRT) --run-id $(RUN)
