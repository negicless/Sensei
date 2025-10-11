run:
	python -m bot.main

fmt:
	python -m pip install ruff black && ruff check --fix . && black .
