.PHONY: clean test lint init

help:
	@echo "    clean"
	@echo "        Remove python artifacts and build artifacts."
	@echo "    lint"
	@echo "        Check style with flake8."
	@echo "    test"
	@echo "        Run py.test"
	@echo "    check-readme"
	@echo "        Check if the readme can be converted from md to rst for pypi"

init:
	pip install pipenv --upgrade
	pipenv install --dev --skip-lock

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf docs/_build

lint:
	py.test --pep8 -m pep8

test:
	detox

ci:
	pipenv run py.test --pep8 -m pep8
	pipenv run py.test tests/base -n 8 --boxed --cov rasa_nlu -v --cov-append
	pipenv run py.test tests/training -n 8 --boxed --cov rasa_nlu -v --cov-append

livedocs:
	cd docs && make livehtml

check-readme:
	# if this runs through we can be sure the readme is properly shown on pypi
	@pipenv run python setup.py check --restructuredtext --strict

flake8:
	pipenv run flake8 --ignore=E501,F401,E128,E402,E731,F821 rasa_nlu


coverage:
	pipenv run py.test --cov-config .coveragerc --verbose --cov-report term --cov-report xml --cov=requests tests
