.PHONY: install install-ml test eval demo serve
install: ; pip install -e ".[dev]"
install-ml: ; pip install -e ".[dev,ml]"
test: ; pytest -q
eval: ; python -m erag.cli eval
demo: ; python -m erag.cli ask "How many days of annual leave do I get?" --role employee
serve: ; uvicorn erag.api:app --reload --port 8000
