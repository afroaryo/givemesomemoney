import pandas as pd
import util as utils
from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler

def load_dataset(config_data: dict):
    x_train = utils.pickle_load(config_data["train_set_path"][0])
    y_train = utils.pickle_load(config_data["train_set_path"][1])

    x_valid = utils.pickle_load(config_data["valid_set_path"][0])
    y_valid = utils.pickle_load(config_data["valid_set_path"][1])


    x_test = utils.pickle_load(config_data["test_set_path"][0])
    y_test = utils.pickle_load(config_data["test_set_path"][1])

    train_set = pd.concat([x_train, y_train], axis = 1)
    valid_set = pd.concat([x_valid, y_valid], axis = 1)
    valid_set = valid_set.drop(['NumberOfTimes90DaysLate','NumberOfTime60-89DaysPastDueNotWorse'], axis=1)
    test_set = pd.concat([x_test, y_test], axis = 1)
    test_set = test_set.drop(['NumberOfTimes90DaysLate','NumberOfTime60-89DaysPastDueNotWorse'], axis=1)

    return train_set, valid_set, test_set

def ros_fit_resample(set_data, config):
    sm = SMOTE(random_state = 12, sampling_strategy = 'auto')
    x_ros, y_ros = sm.fit_resample(set_data.drop(columns = config["label"]),set_data[config["label"]])    
    train_set_bal = pd.concat([x_ros, y_ros], axis = 1)

    return train_set_bal

def remove_outliers(set_data):
    set_data = set_data[set_data['RevolvingUtilizationOfUnsecuredLines'] <= 100]
    set_data = set_data[set_data['Age'] > 0]
    set_data = set_data[set_data['DebtRatio'] <= 100]
    set_data = set_data[set_data['MonthlyIncome'] <= 100000]
    set_data = set_data[set_data['NumberOfTimes90DaysLate'] <= 25]
    set_data = set_data[set_data['NumberOfTimes90DaysLate'] <= 10]
    set_data = set_data[set_data['NumberOfDependents'] <= 5]
    
    set_data_cleaned = set_data.drop(['NumberOfTimes90DaysLate','NumberOfTime60-89DaysPastDueNotWorse'], axis=1)

    return set_data_cleaned


if __name__ == "__main__":
    # 1. Load configuration file
    config = utils.load_config()

    # 2. Load dataset
    train_set, valid_set, test_set = load_dataset(config)

    # 3. Removing outliers
    train_set_bal_cleaned = remove_outliers(train_set_bal)
    
    # 4. Oversampling dataset
    train_set_bal = ros_fit_resample(train_set, config)

    # 5. Dump set data
    utils.pickle_dump(
            train_set_bal_cleaned[config["predictors"]],
            config["train_feng_set_path"][0]
    )
    utils.pickle_dump(
            train_set_bal_cleaned[config["label"]],
            config["train_feng_set_path"][1]
    )


    utils.pickle_dump(
            valid_set[config["predictors"]],
            config["valid_feng_set_path"][0]
    )
    utils.pickle_dump(
            valid_set[config["label"]],
            config["valid_feng_set_path"][1]
    )


    utils.pickle_dump(
            test_set[config["predictors"]],
            config["test_feng_set_path"][0]
    )
    utils.pickle_dump(
            test_set[config["label"]],
            config["test_feng_set_path"][1]
    )

    