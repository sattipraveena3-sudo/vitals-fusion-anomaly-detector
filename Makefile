.PHONY: install test run demo sample docker
install:
	python -m pip install -r requirements.txt
test:
	pytest
run:
	uvicorn app.main:app --reload
demo:
	python scripts/demo_fusion.py
sample:
	python scripts/generate_sample.py
docker:
	docker compose up --build
