import pandas as pd
import util as utils
import copy
from sklearn.model_selection import train_test_split

def read_raw_data(config: dict) -> pd.DataFrame:
    # Return raw dataset
    return pd.read_csv(config["dataset_path"])

def check_data(input_data: pd.DataFrame, config: dict, api: bool = False):
    input_data = copy.deepcopy(input_data)
    config = copy.deepcopy(config)

    if not api:
        # Check column data types
        assert input_data.select_dtypes("int").columns.to_list() == \
            config["int_columns"], "an error occurs in int column(s)."
        assert input_data.select_dtypes("float").columns.to_list() == \
            config["float_columns"], "an error occurs in float column(s)."

        # Check range of unsecured lines
        assert input_data[config["float_columns"][0]].between(
                config["range_unsecured_lines"][0],
                config["range_unsecured_lines"][1]
                ).sum() == len(input_data), "an error occurs in range_unsecured_lines."
        
        # Check range of debt ratio
        assert input_data[config["float_columns"][1]].between(
                config["range_debt_ratio"][0],
                config["range_debt_ratio"][1]
                ).sum() == len(input_data), "an error occurs in range_debt_ratio."
        
        # Check range of monthly income
        assert input_data[config["float_columns"][2]].between(
                config["range_monthly_income"][0],
                config["range_monthly_income"][1]
                ).sum() == len(input_data), "an error occurs in range_monthly_income."

        # Check range of fire alarm
        assert input_data[config["int_columns"][5]].between(
                config["range_serious_dlq"][0],
                config["range_serious_dlq"][1]
                ).sum() == len(input_data), "an error occurs in range_serious_dlq."

    else:
        # In case checking data from api
        # Last 2 column names in list of int columns are not used as predictor (CNT and Fire Alarm)
        int_columns = config["int_columns"]
        del int_columns[-2:]

        # Last 4 column names in list of int columns are not used as predictor (NC2.5, NC1.0, NC0.5, and PM2.5)
        float_columns = config["float_columns"]
        del float_columns[-4:]

        # Check column data types
        assert input_data.select_dtypes("int64").columns.to_list() == \
            int_columns, "an error occurs in int column(s)."
        assert input_data.select_dtypes("float64").columns.to_list() == \
            float_columns, "an error occurs in float column(s)."
    
    # Check range of age
    assert input_data[config["int_columns"][0]].between(
            config["range_age"][0],
            config["range_age"][1]
            ).sum() == len(input_data), "an error occurs in range_age."
    
    # Check range of 59_past_worse
    assert input_data[config["int_columns"][1]].between(
            config["range_59_past_worse"][0],
            config["range_59_past_worse"][1]
            ).sum() == len(input_data), "an error occurs in range_59_past_worse."
    
    # Check range of range_open_credit
    assert input_data[config["int_columns"][2]].between(
            config["range_open_credit"][0],
            config["range_open_credit"][1]
            ).sum() == len(input_data), "an error occurs in range_open_credit."
    
    # Check range of range_real_estate_loan
    assert input_data[config["int_columns"][3]].between(
            config["range_real_estate_loan"][0],
            config["range_real_estate_loan"][1]
            ).sum() == len(input_data), "an error occurs in range_real_estate_loan."
    
    # Check range of dependents
    assert input_data[config["int_columns"][4]].between(
            config["range_dependents"][0],
            config["range_dependents"][1]
            ).sum() == len(input_data), "an error occurs in range_dependents."
    

def split_data(input_data: pd.DataFrame, config: dict):
    # Split predictor and label
    x = input_data[config["predictors"]].copy()
    y = input_data[config["label"]].copy()

    # 1st split train and test
    x_train, x_test, \
    y_train, y_test = train_test_split(
        x, y,
        test_size = config["test_size"],
        random_state = 42,
        stratify = y
    )

    # 2nd split test and valid
    x_valid, x_test, \
    y_valid, y_test = train_test_split(
        x_test, y_test,
        test_size = config["valid_size"],
        random_state = 42,
        stratify = y_test
    )

    return x_train, x_valid, x_test, y_train, y_valid, y_test


if __name__ == "__main__":
    # 1. Load configuration file
    config = utils.load_config()

    # 2. Read all raw dataset
    raw_dataset = read_raw_data(config)

    # 3. Convert to datetime
    raw_dataset = convert_datetime(raw_dataset, config)

    # 4. Data defense for non API data
    check_data(raw_dataset, config)

    # 5. Splitting train, valid, and test set
    x_train, x_valid, x_test, \
        y_train, y_valid, y_test = split_data(raw_dataset, config)

    # 6. Save train, valid and test set
    utils.pickle_dump(x_train, config["train_set_path"][0])
    utils.pickle_dump(y_train, config["train_set_path"][1])

    utils.pickle_dump(x_valid, config["valid_set_path"][0])
    utils.pickle_dump(y_valid, config["valid_set_path"][1])

    utils.pickle_dump(x_test, config["test_set_path"][0])
    utils.pickle_dump(y_test, config["test_set_path"][1])

    utils.pickle_dump(raw_dataset, config["dataset_cleaned_path"])