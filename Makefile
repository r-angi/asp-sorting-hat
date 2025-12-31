.PHONY: install test

install:
	python3 -m pip install -r requirements.txt

test:
	python3 -m pytest tests/ -v

