.PHONY: test clean-tmp ruff sweep
test:
	PYTHONPATH=src/ pytest src/test_*.py

clean-tmp:
	scripts/clean-my-tmp

ruff:
	scripts/ruff.sh

sweep:
	scripts/sweep.sh
