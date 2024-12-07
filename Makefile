VENV := venv


env:
	python3.11 -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

lint:
	. $(VENV)/bin/activate && ruff check notebook/*.ipynb lib/*.py

format:
	. $(VENV)/bin/activate && black lib/*.py

all: env lint format