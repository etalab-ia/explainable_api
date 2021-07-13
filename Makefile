.ONESHELL:
SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
CONFIG_FILE=./src/config/mlapi_parameters.json
.PHONY: create_pipelines launch_dashboard run_all
create_pipelines:
	$(CONDA_ACTIVATE) ml_api
	python -m src.make_api_pipelines $(CONFIG_FILE)
launch_dashboard:
	$(CONDA_ACTIVATE) ml_api
	python -m src.launch_dashboard_hub  $(CONFIG_FILE)
all: create_pipelines launch_dashboard
