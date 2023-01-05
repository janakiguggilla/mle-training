import argparse
import logging
import logging.config
import os
import tarfile
import urllib
import urllib.request

import numpy as np
import pandas as pd
from logging_tree import printout
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

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
parser.add_argument("path", type=str, help="Output Folder")
parser.add_argument("log_level", type=str, help="Log_level")
parser.add_argument("log_console", type=bool, help="Log_console")
args = parser.parse_args()
output_folder = args.path



DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("..", "datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Function to download training Data"""
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


fetch_housing_data()


def load_housing_data(housing_path=HOUSING_PATH):
    """Function to load training data."""
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)

# housing = strat_train_set.drop("median_house_value", axis=1)  # drop labels for training set
housing_labels = housing["median_house_value"].copy()
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
housing_tr["bedrooms_per_room"] = (
    housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
)
housing_tr["population_per_household"] = (
    housing_tr["population"] / housing_tr["households"]
)

housing_cat = housing[["ocean_proximity"]]
housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
strat_train_set = pd.DataFrame()
strat_test_set = pd.DataFrame()
for train_index, test_index in split.split(
    housing_prepared.drop("median_house_value", axis=1), housing_prepared["income_cat"]
):
    strat_train_set = housing_prepared.loc[train_index]
    strat_test_set = housing_prepared.loc[test_index]


# outdir = args.path
# if not os.path.exists(outdir):
#    os.mkdir(outdir)


path = os.path.join(os.getcwd(), output_folder)
strat_train_set.to_csv(os.path.join(path, "train_data.csv"), header=True, index=False)
strat_test_set.to_csv(os.path.join(path, "test_data.csv"), header=True, index=False)
