import argparse
import logging
import logging.config
import os
import pickle
import urllib

import pandas as pd
from logging_tree import printout
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {"format": "%(message)s"},
    },
    "root": {"level": "DEBUG"},
}


# All the library modules assume that logger had been configured elsewhere
# Logger configurations or specifications are never initialised in individual modules.
# They only inherit the configurations from the entry point scripts
logger = logging.getLogger(__name__)


def dummy_function():
    """Dummy Function to test logging inside a function"""
    logger.info(f"Logging Test - Function Call Object Done")


def configure_logger(
    logger=None, cfg=None, log_file=None, console=True, log_level="DEBUG"
):
    """Function to setup configurations of logger through function.

    The individual arguments of `log_file`, `console`, `log_level` will overwrite the ones in cfg.

    Parameters
    ----------
            logger:
                    Predefined logger object if present. If None a ew logger object will be created from root.
            cfg: dict()
                    Configuration of the logging to be implemented by default
            log_file: str
                    Path to the log file for logs to be stored
            console: bool
                    To include a console handler(logs printing in console)
            log_level: str
                    One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
                    default - `"DEBUG"`

    Returns
    -------
    logging.Logger
    """

    if not cfg:
        logging.config.dictConfig(LOGGING_DEFAULT_CONFIG)
    else:
        logging.config.dictConfig(cfg)

    logger = logger or logging.getLogger()

    if log_file or console:
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

        if log_file:
            # if not os.path.exists()
            # path = os.path.join(os.getcwd(), 'logs', log_file)
            fh = logging.FileHandler(log_file)
            fh.setLevel(getattr(logging, log_level))
            logger.addHandler(fh)

        if console:
            sh = logging.StreamHandler()
            sh.setLevel(getattr(logging, log_level))
            logger.addHandler(sh)

    return logger


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, help="Training Data Folder")
parser.add_argument("model", type=str, help="Model pickle output folder")
parser.add_argument("log_level", type=str, help="Log_level")
parser.add_argument("log_console", type=bool, help="Log_console")
args = parser.parse_args()
dataset_folder = args.dataset
output_folder = args.model

# outdir = args.path
# if not os.path.exists(output_folder):
#    os.mkdir(output_folder)


train_data = pd.read_csv(os.path.join(dataset_folder, "train_data.csv"))
test_data = pd.read_csv(os.path.join(dataset_folder, "test_data.csv"))

lin_reg = LinearRegression()
lin_reg.fit(
    train_data.drop(columns=["median_house_value"]), train_data["median_house_value"]
)

pickle.dump(lin_reg, open(os.path.join(output_folder, "linear_model"), "wb"))

# housing_predictions = lin_reg.predict(housing_prepared)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# lin_rmse


# lin_mae = mean_absolute_error(housing_labels, housing_predictions)
# lin_mae


tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(
    train_data.drop(columns=["median_house_value"]), train_data["median_house_value"]
)

pickle.dump(tree_reg, open(os.path.join(output_folder, "decisiontree_model"), "wb"))
# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# tree_rmse = np.sqrt(tree_mse)
# tree_rmse


param_distribs = {
    "n_estimators": randint(low=1, high=200),
    "max_features": randint(low=1, high=8),
}

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(
    forest_reg,
    param_distributions=param_distribs,
    n_iter=10,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
)
rnd_search.fit(
    train_data.drop(columns=["median_house_value"]), train_data["median_house_value"]
)
# cvres = rnd_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#    print(np.sqrt(-mean_score), params)


param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(
    forest_reg,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True,
)
grid_search.fit(
    train_data.drop(columns=["median_house_value"]), train_data["median_house_value"]
)

# grid_search.best_params_
# cvres = grid_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#    print(np.sqrt(-mean_score), params)

# feature_importances = grid_search.best_estimator_.feature_importances_
# sorted(zip(feature_importances, housing_prepared.columns), reverse=True)


final_model = grid_search.best_estimator_

pickle.dump(final_model, open(os.path.join(output_folder, "randomforest_model"), "wb"))
