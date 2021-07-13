import json
import pickle
from pathlib import Path
from typing import List, Dict

from explainerdashboard import ExplainerHub, ClassifierExplainer, ExplainerDashboard

PARAMETER_FILE = Path("config/mlapi_parameters.json")
if PARAMETER_FILE.exists():
    with open(PARAMETER_FILE) as fout:
        PARAMETERS = json.load(fout)
else:
    raise FileNotFoundError(f"Config file {PARAMETER_FILE.as_posix()} does not exist.")


def save_explainer(explainer: ClassifierExplainer, api_name: str):
    explainer_path = Path(PARAMETERS[0]["per_api_data_file"]).parent.joinpath(Path(f"explainers"))
    if not explainer_path.exists():
        explainer_path.mkdir()
    explainer_path = explainer_path / Path(f"explainer_{api_name}.dill")
    try:
        explainer.dump(explainer_path.as_posix())

    except Exception as e:
        print(f"Could not save Explainer for API {api_name}. Error {e}")


def create_explainer_dashboards(per_api_data: Path,
                                tabs: List[str] = ["importances", "model_summary", "contributions"]):
    dashboard_per_api = {}
    data_per_api = pickle.load(open(per_api_data, "rb"))
    for api, expe in list(data_per_api.items())[:]:
        print(f"Treating API {api}")
        shap = "guess"
        if expe["algo_name"] in ['XGBoostClassifier', 'CatBoostClassifier', 'RandomForestClassifier']:
            shap = 'tree'
        elif expe["algo_name"] == 'LogisticRegression':
            shap = 'linear'

        explainer = ClassifierExplainer(expe["model"], expe["X_test"], expe["y_test"], shap=shap)
        # save_explainer(explainer, api)
        dashboard = ExplainerDashboard(explainer, tabs=tabs, title=f"{api.replace('_', ' ')} API")
        dashboard_per_api[api] = dashboard
    return dashboard_per_api


def create_dashboard_hub(dashboard_per_api: Dict[str, ExplainerDashboard]):
    hub = ExplainerHub(dashboards=list(dashboard_per_api.values()), title="DataPass Explainers Hub")
    hub.run()


def main():
    dashboard_per_api = create_explainer_dashboards(Path(PARAMETERS[0]["per_api_data_file"]))
    create_dashboard_hub(dashboard_per_api)


if __name__ == '__main__':
    main()
