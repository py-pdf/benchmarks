maint:
	uv lock --upgrade
	uv run pre-commit autoupdate

run:
	uv run python benchmark.py
