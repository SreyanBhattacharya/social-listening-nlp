.PHONY: install train topics insights all api test docker

install:
	pip install -r requirements-dev.txt && pip install -e .

train:
	python -m sociallistening.train

topics:
	python -m sociallistening.topics

insights:
	python -m sociallistening.insights

all: train topics insights

api:
	uvicorn sociallistening.api:app --reload

test:
	pytest -q

docker:
	docker build -t social-listening-nlp . && docker run -p 8000:8000 social-listening-nlp
