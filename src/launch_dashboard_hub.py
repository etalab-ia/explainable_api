import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict

from explainerdashboard import ExplainerHub, ClassifierExplainer, ExplainerDashboard

def read_parameters(parameters_file: Path):
    if parameters_file.exists():
        with open(parameters_file) as fout:
            parameters = json.load(fout)
    else:
        raise FileNotFoundError(f"Config file {parameters_file.as_posix()} does not exist.")
    return parameters


def save_explainer(explainer: ClassifierExplainer, api_name: str, parameters: dict):
    explainer_path = Path(parameters[0]["per_api_data_file"]).parent.joinpath(Path(f"explainers"))
    if not explainer_path.exists():
        explainer_path.mkdir()
    explainer_path = explainer_path / Path(f"explainer_{api_name}.dill")
    try:
        explainer.dump(explainer_path.as_posix())

    except Exception as e:
        print(f"Could not save Explainer for API {api_name}. Error {e}")


def create_explainer_dashboards(per_api_data: Path,
                                parameters: dict,
                                tabs: List[str] = ["importances", "model_summary", "contributions"],
                                use_cache: bool = False):
    dashboard_per_api = {}
    data_per_api = pickle.load(open(per_api_data, "rb"))
    for api, expe in list(data_per_api.items())[:]:
        explainer = None
        print(f"Treating API {api}")
        shap = "guess"
        if expe["algo_name"] in ['XGBoostClassifier', 'CatBoostClassifier', 'RandomForestClassifier']:
            shap = 'tree'
        elif expe["algo_name"] == 'LogisticRegression':
            shap = 'linear'

        if use_cache:
            explainer = load_explainer(api, parameters=parameters)
        if explainer is None:
            explainer = ClassifierExplainer(expe["model"], expe["X_test"], expe["y_test"], shap=shap)
            explainer.get_shap_values_df()
            save_explainer(explainer, api, parameters=parameters)
        dashboard = ExplainerDashboard(explainer, tabs=tabs, title=f"{api.replace('_', ' ')} API")
        dashboard_per_api[api] = dashboard
    return dashboard_per_api


def load_explainer(api_name: str, parameters: dict):
    explainer_path = Path(parameters[0]["per_api_data_file"]).parent.joinpath(
        Path(f"explainers/explainer_{api_name}.dill"))
    explainer = None
    if explainer_path.exists():
        try:
            explainer = ClassifierExplainer.from_file(explainer_path.as_posix())
        except Exception as e:
            print(f"Could not load cached model in {explainer_path}. Error {e}")
        return explainer
    print(f"Cached Explainer for API {api_name} does not exist in path {explainer_path}.")


def create_dashboard_hub(dashboard_per_api: Dict[str, ExplainerDashboard]):
    hub = ExplainerHub(dashboards=list(dashboard_per_api.values()), title="DataPass Explainers Hub")
    hub.run()


def main(parameters_file: Path):
    parameters = read_parameters(parameters_file=parameters_file)

    dashboard_per_api = create_explainer_dashboards(Path(parameters[0]["per_api_data_file"]),
                                                    parameters=parameters,
                                                    use_cache=True)
    create_dashboard_hub(dashboard_per_api)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, default="./src/config/mlapi_parameters.json")
    args = parser.parse_args()
    main(Path(args.config_file))
