.PHONY: format

format:
	black pose_evaluation
	python -m ruff format .
	python -m ruff check --fix .