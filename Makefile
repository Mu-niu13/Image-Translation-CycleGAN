# Makefile

.PHONY: lint install clean

install:
	@pip install -r requirements.txt

lint:
	@flake8 lib

clean:
	@find . -name '*.pyc' -delete
	@rm -rf __pycache__
