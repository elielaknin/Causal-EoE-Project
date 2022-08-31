import numpy as np
import pandas as pd


def remove_nan_val_from_list(list_of_index, nan_values):
    index_list_without_nan = []
    for value in list_of_index:
        if value not in nan_values:
            index_list_without_nan.append(value)

    return index_list_without_nan


def filter_df(original_df):
    temp_df = original_df.set_index(['type'])
    number_of_rows = temp_df.shape[0]

    min_number_of_samples = 40

    if number_of_rows == 13:# ATE filter
        # in the ATE case, drop only outcomes that have less than 40 samples
        columns_to_drop = list(temp_df.columns[np.where(temp_df.loc['number_active'] < min_number_of_samples)[0]])
        df_after_filter = temp_df.drop(columns_to_drop, axis=1)

    elif number_of_rows == 26:# CATE filter
        # in the CATE case, drop outcomes that have less than 40 samples and ratio of percentage is
        # not lower than 20 or higher than 80.
        cols_above_number_active = list(temp_df.columns[np.where(temp_df.loc['above_TH_num_active'] < min_number_of_samples)[0]])
        cols_above_perc_lower_20 = list(temp_df.columns[np.where(temp_df.loc['above_TH_percentage'] < 20)[0]])
        cols_above_perc_higher_80 = list(temp_df.columns[np.where(temp_df.loc['above_TH_percentage'] > 80)[0]])
        columns_above_to_drop = list(set(cols_above_number_active + cols_above_perc_lower_20 + cols_above_perc_higher_80))

        cols_under_number_active = list(temp_df.columns[np.where(temp_df.loc['under_equal_TH_num_active'] < min_number_of_samples)[0]])
        cols_under_perc_lower_20 = list(temp_df.columns[np.where(temp_df.loc['under_equal_TH_percentage'] < 20)[0]])
        cols_under_perc_higher_80 = list(temp_df.columns[np.where(temp_df.loc['under_equal_TH_percentage'] > 80)[0]])
        columns_under_to_drop = list(set(cols_under_number_active + cols_under_perc_lower_20 + cols_under_perc_higher_80))

        columns_to_drop = list(set(columns_above_to_drop + columns_under_to_drop))
        df_after_filter = temp_df.drop(columns_to_drop, axis=1)

    else:
        raise Exception(f"numbers of rows in the dataframe is incorrect:{number_of_rows}")

    df_after_filter.reset_index(inplace=True)
    return df_after_filter


def main():
    original_df = pd.read_csv("data/cate_tissue_no_op_average_symptoms_score.csv")
    filter_df(original_df)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
