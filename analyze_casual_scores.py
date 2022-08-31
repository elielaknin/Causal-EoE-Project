import argparse
import os
import pandas as pd
import numpy as np
import utils
from tqdm import tqdm


# Code that analyzes the ATE and CATE from the "calc_causal_scores" code.
# The script filter results that have a low number of samples, and return 3 CSVs.
# one for ATE sorted by pvalues (low to high).
# one for the CATE and one for CATE separated (where the under and above threshold) are together.
def parse_args():
    parser = argparse.ArgumentParser(description='This script is ...'
                                     , formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("data_dir_path")
    args = parser.parse_args()
    return args


def analyze_ate(data_type_dir, data_type_name):
    df = pd.read_csv(os.path.join(data_type_dir, f"ate_{data_type_name}.csv"))
    df = utils.filter_df(df)
    data = np.array(df[df['type'] == 'ate'].iloc[0, 1:])
    headers = np.array(df.columns[1:])
    res_df = pd.DataFrame()
    res_df['outcome'] = headers
    res_df['ate'] = data
    res_df['pvalue'] = np.array(df[df['type'] == 'pvalue_hist'].iloc[0, 1:])
    res_df['CI_95_low'] = np.array(df[df['type'] == 'CI_95_low'].iloc[0, 1:])
    res_df['CI_95_high'] = np.array(df[df['type'] == 'CI_95_high'].iloc[0, 1:])
    res_df = res_df.sort_values(by=['pvalue'])
    return res_df


def analyze_cate(data_type_dir, data_type_name):
    cate_data_path = os.path.join(data_type_dir, 'cate')

    res_df = pd.DataFrame()
    cate_seperated_df = pd.DataFrame()
    for outcome_file_name in os.listdir(cate_data_path):
        outcome = outcome_file_name.split(".csv")[0].split(f"{data_type_name}_")[1]
        outcome_file_path = os.path.join(cate_data_path, outcome_file_name)
        outcome_df = pd.read_csv(outcome_file_path)
        outcome_df = utils.filter_df(outcome_df)

        outcome_res_df = pd.DataFrame()
        outcome_res_df['outcome'] = np.array([outcome for _ in range(len(outcome_df.columns[1:]))])
        outcome_res_df['condition'] = np.array(outcome_df.columns[1:])
        outcome_res_df['above_TH_cate'] = np.array(outcome_df[outcome_df['type'] == 'above_TH_ate'].iloc[0, 1:])
        outcome_res_df['under_equal_TH_cate'] = np.array(
            outcome_df[outcome_df['type'] == 'under_equal_TH_ate'].iloc[0, 1:])
        outcome_res_df['distance'] = np.abs(outcome_res_df['above_TH_cate'] - outcome_res_df['under_equal_TH_cate'])
        outcome_res_df['above_pvalues'] = np.array(outcome_df[outcome_df['type'] == 'above_TH_pvalue_hist'].iloc[0, 1:])
        outcome_res_df['under_equal_pvalues'] = np.array(
            outcome_df[outcome_df['type'] == 'under_equal_TH_pvalue_hist'].iloc[0, 1:])
        outcome_res_df['pvalues_under_0.05'] = (outcome_res_df['above_pvalues'] <= 0.05) | (
                    outcome_res_df['under_equal_pvalues'] <= 0.05)
        outcome_res_df['above_min'] = np.array(outcome_df[outcome_df['type'] == 'above_TH_min'].iloc[0, 1:])
        outcome_res_df['above_max'] = np.array(outcome_df[outcome_df['type'] == 'above_TH_max'].iloc[0, 1:])
        outcome_res_df['under_equal_min'] = np.array(outcome_df[outcome_df['type'] == 'under_equal_TH_min'].iloc[0, 1:])
        outcome_res_df['under_equal_max'] = np.array(outcome_df[outcome_df['type'] == 'under_equal_TH_max'].iloc[0, 1:])
        outcome_res_df['above_CI_95_low'] = np.array(outcome_df[outcome_df['type'] == 'above_TH_CI_95_low'].iloc[0, 1:])
        outcome_res_df['above_CI_95_high'] = np.array(
            outcome_df[outcome_df['type'] == 'above_TH_CI_95_high'].iloc[0, 1:])
        outcome_res_df['under_equal_CI_95_low'] = np.array(
            outcome_df[outcome_df['type'] == 'under_equal_TH_CI_95_low'].iloc[0, 1:])
        outcome_res_df['under_equal_CI_95_high'] = np.array(
            outcome_df[outcome_df['type'] == 'under_equal_TH_CI_95_high'].iloc[0, 1:])
        res_df = pd.concat([res_df, outcome_res_df], axis=0, ignore_index=True)

        outcome_separeted_res_df = pd.DataFrame()
        outcome_separeted_res_df['outcome'] = np.array([outcome for _ in range(2 * len(outcome_df.columns[1:]))])

        above_condition_names = "above_" + np.array(outcome_df.columns[1:])
        under_condition_names = "under_equal_" + np.array(outcome_df.columns[1:])
        outcome_separeted_res_df['condition'] = np.concatenate((above_condition_names, under_condition_names))

        above_cate = np.array(outcome_df[outcome_df['type'] == 'above_TH_ate'].iloc[0, 1:])
        under_cate = np.array(outcome_df[outcome_df['type'] == 'under_equal_TH_ate'].iloc[0, 1:])
        outcome_separeted_res_df['cate'] = np.concatenate((above_cate, under_cate))

        above_pvalue = np.array(outcome_df[outcome_df['type'] == 'above_TH_pvalue_hist'].iloc[0, 1:])
        under_pvalue = np.array(outcome_df[outcome_df['type'] == 'under_equal_TH_pvalue_hist'].iloc[0, 1:])
        outcome_separeted_res_df['pvalue'] = np.concatenate((above_pvalue, under_pvalue))

        above_min = np.array(outcome_df[outcome_df['type'] == 'above_TH_min'].iloc[0, 1:])
        under_min = np.array(outcome_df[outcome_df['type'] == 'under_equal_TH_min'].iloc[0, 1:])
        outcome_separeted_res_df['min'] = np.concatenate((above_min, under_min))

        above_max = np.array(outcome_df[outcome_df['type'] == 'above_TH_max'].iloc[0, 1:])
        under_max = np.array(outcome_df[outcome_df['type'] == 'under_equal_TH_max'].iloc[0, 1:])
        outcome_separeted_res_df['max'] = np.concatenate((above_max, under_max))

        above_CI_95_low = np.array(outcome_df[outcome_df['type'] == 'above_TH_CI_95_low'].iloc[0, 1:])
        under_CI_95_low = np.array(outcome_df[outcome_df['type'] == 'under_equal_TH_CI_95_low'].iloc[0, 1:])
        outcome_separeted_res_df['CI_95_low'] = np.concatenate((above_CI_95_low, under_CI_95_low))

        above_CI_95_high = np.array(outcome_df[outcome_df['type'] == 'above_TH_CI_95_high'].iloc[0, 1:])
        under_CI_95_high = np.array(outcome_df[outcome_df['type'] == 'under_equal_TH_CI_95_high'].iloc[0, 1:])
        outcome_separeted_res_df['CI_95_high'] = np.concatenate((above_CI_95_high, under_CI_95_high))
        cate_seperated_df = pd.concat([cate_seperated_df, outcome_separeted_res_df], axis=0, ignore_index=True)

    res_df = res_df.iloc[res_df[['above_TH_cate', 'under_equal_TH_cate']].min(axis=1).sort_values(ascending=True).index,
             :]
    cate_seperated_df.sort_values(by=['pvalue', 'cate'], ascending=True, inplace=True)
    return res_df, cate_seperated_df


def analyze_data_type(data_main_path, data_type_name):
    data_path = os.path.join(data_main_path, data_type_name)
    analysis_res_dir = os.path.join(data_path, 'analysis')
    if not os.path.exists(analysis_res_dir):
        os.makedirs(analysis_res_dir)

    ate_df = analyze_ate(data_path, data_type_name)
    ate_df.to_csv(os.path.join(analysis_res_dir, "sorted_ate.csv"), index=False)

    cate_df, cate_seperated_df = analyze_cate(data_path, data_type_name)
    cate_df.to_csv(os.path.join(analysis_res_dir, "sorted_cate.csv"), index=False)
    cate_seperated_df.to_csv(os.path.join(analysis_res_dir, "sorted_cate_seperated.csv"), index=False)


def main():
    args = parse_args()
    data_dir_path = args.data_dir_path
    for data_type in tqdm(os.listdir(data_dir_path)):
        if os.path.isdir(os.path.join(data_dir_path, data_type)):
            if data_type == 'patients_standardization':
                analyze_data_type(data_dir_path, data_type)


if __name__ == '__main__':
    main()
