import argparse
import logging
import logging.config
import os
import pickle
import urllib.request

import numpy as np
import pandas as pd
from logging_tree import printout
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
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
parser.add_argument("dataset", type=str, help="Train_Test Data Folder")
parser.add_argument("model", type=str, help="Model Folder")
parser.add_argument("score", type=str, help="Score Output Folder")
parser.add_argument("log_level", type=str, help="Log_level")
parser.add_argument("log_console", type=bool, help="Log_console")
args = parser.parse_args()
dataset_folder = args.dataset
model_folder = args.model
score_folder = args.score


# if not os.path.exists(score_folder):
#   os.mkdir(score_folder)

train_data = pd.read_csv(os.path.join(dataset_folder, "train_data.csv"))
test_data = pd.read_csv(os.path.join(dataset_folder, "test_data.csv"))
logger.debug("Loaded data")
linear_model = pickle.load(open(os.path.join(model_folder, "linear_model"), "rb"))
decisiontree_model = pickle.load(
    open(os.path.join(model_folder, "decisiontree_model"), "rb")
)
randomforest_model = pickle.load(
    open(os.path.join(model_folder, "randomforest_model"), "rb")
)
logger.debug("Loaded model")
lin_predictions = linear_model.predict(test_data.drop(columns=["median_house_value"]))
lin_mse = mean_squared_error(lin_predictions, test_data["median_house_value"])
lin_rmse = np.sqrt(lin_mse)

lin_mae = mean_absolute_error(lin_predictions, test_data["median_house_value"])

with open(os.path.join(score_folder, "lin_score.txt"), "w") as file1:
    # Writing data to a file
    file1.write("Test Scores\n")
    file1.write("Root Mean Squared Error {}".format(lin_rmse))
    file1.write("\n")
    file1.write("Mean Absolute Error {}".format(lin_mae))
    file1.write("\n")

logger.debug("Written data")
tree_predictions = decisiontree_model.predict(
    test_data.drop(columns=["median_house_value"])
)
tree_mse = mean_squared_error(tree_predictions, test_data["median_house_value"])
tree_rmse = np.sqrt(tree_mse)

with open(os.path.join(score_folder, "tree_score.txt"), "w") as file2:
    # Writing data to a file
    file2.write("Decision Tree Scores\n")
    file2.write("Root Mean Squared Error {}".format(tree_rmse))
    file2.write("\n")


forest_predictions = randomforest_model.predict(
    test_data.drop(columns=["median_house_value"])
)
final_mse = mean_squared_error(forest_predictions, test_data["median_house_value"])
final_rmse = np.sqrt(final_mse)

with open(os.path.join(score_folder, "forest_score.txt"), "w") as file3:
    file3.write("Random Forest Scores\n")
    file3.write("Root_Mean_Squared_Score {}".format(final_rmse))
    file3.write("\n")
