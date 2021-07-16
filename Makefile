.ONESHELL:
SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
CONFIG_FILE=./src/config/mlapi_parameters.json
.PHONY: create_pipelines launch_dashboard build run_all build_docker run_docker

create_pipelines:
	$(CONDA_ACTIVATE) ml_api
	python -m src.make_api_pipelines $(CONFIG_FILE)
launch_dashboard:
	$(CONDA_ACTIVATE) ml_api
	python -m src.launch_dashboard_hub  $(CONFIG_FILE)

build_docker:
	docker build -t api_dashboard .

run_docker:
	docker run -i -t --rm -p 8050:8050 --name api_dashboard api_dashboard

all: create_pipelines launch_dashboard build_docker run_docker
