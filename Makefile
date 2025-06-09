.PHONY: format

format:
	black .
	isort .
	docformatter -ir .
	ruff check --fix .
	autoflake --in-place --remove-unused-variables --remove-all-unused-imports -r .
