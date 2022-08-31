import argparse
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

from casual_parser import CasualParser

# main script that calculates the ATE and CATE scores for all features.
# the code returns a CSV with apo, ate, CI, pvalues and other statistics for each outcome (in ATE) and for
# each outcome + condition (for CATE). In the CATE different thresholds of the conditional are applied.
def parse_args():
    parser = argparse.ArgumentParser(description='This script is ...'
                                     , formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("data_dir_path")
    parser.add_argument("output_dir_path")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_dir_path = args.data_dir_path

    patient_6weeks_path = os.path.join(data_dir_path, "patients_6weeks_causal_data.csv")
    patient_start_path = os.path.join(data_dir_path, "patients_baseline_causal_data.csv")
    tissue_6weeks_path = os.path.join(data_dir_path, "samples_6weeks_causal_data.csv")
    tissue_start_path = os.path.join(data_dir_path, "samples_baseline_causal_data.csv")

    res_dir = os.path.join(args.output_dir_path, 'casual_results')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # choose which of data to use (patient or tissue) and also the type of outcome operation:
    # None, standardization, abs_distance, ratio_distance

    analyze_data_type('patients_standardization', patient_6weeks_path, patient_start_path, res_dir, 'standardization')
    # analyze_data_type('patients_no_op', patient_6weeks_path, patient_start_path, res_dir, None)
    # analyze_data_type('patients_abs_dist', patient_6weeks_path, patient_start_path, res_dir, 'abs_distance')
    # analyze_data_type('patients_percentage_dist', patient_6weeks_path, patient_start_path, res_dir, 'ratio_distance')
    # analyze_data_type('tissue_no_op', tissue_6weeks_path, tissue_start_path, res_dir, None)
    # analyze_data_type('tissue_abs_dist', tissue_6weeks_path, tissue_start_path, res_dir, 'abs_distance')
    # analyze_data_type('tissue_percentage_dist', tissue_6weeks_path, tissue_start_path, res_dir, 'ratio_distance')


def calc_all_ate(casual_parser):
    res_df = pd.DataFrame(columns=['type'], data=['treated_apo', 'untreated_apo', 'ate', 'CI_95_low', 'CI_95_high',
                                                  'CI_99_low', 'CI_99_high', 'min', 'max', 'pvalue_hist',
                                                  'pvalue_ttest2', 'percentage', 'number_active'])
    for outcome in tqdm(casual_parser.features):
        outcome_df = casual_parser.calc_cate(outcome)
        res_df[outcome] = outcome_df['value']

    return res_df


def calc_all_cate(casual_parser):
    quantile_list = [0.25, 0.5, 0.75]
    res_dfs_ls = []
    for outcome in tqdm(casual_parser.features):

        outcome_df = casual_parser.iterate_condition_th(outcome, casual_parser.features[0], quantile_list=quantile_list)
        for condition in casual_parser.features[1:]:
            condition_df = casual_parser.iterate_condition_th(outcome, condition, quantile_list=quantile_list).iloc[:,
                           1:]
            outcome_df = pd.concat([outcome_df, condition_df], axis=1)

        res_dfs_ls.append(outcome_df)

    return res_dfs_ls


def analyze_data_type(name, end_path, start_path, res_dir, outcome_operation):
    data_type_res_dir = os.path.join(res_dir, name)
    if not os.path.exists(data_type_res_dir):
        os.makedirs(data_type_res_dir)

    casual_parser = CasualParser(start_path, end_path, outcome_operation=outcome_operation)

    # ATE for all features
    ate_df = calc_all_ate(casual_parser)
    ate_df.to_csv(os.path.join(data_type_res_dir, f"ate_{name}.csv"), index=False)

    # CATE for all features
    cate_res_dir = os.path.join(data_type_res_dir, 'cate')
    if not os.path.exists(cate_res_dir):
        os.makedirs(cate_res_dir)
    cate_dfs_ls = calc_all_cate(casual_parser)
    for i, cate_df in enumerate(cate_dfs_ls):
        cate_df.to_csv(os.path.join(cate_res_dir, f"cate_{name}_{casual_parser.features[i]}.csv"), index=False)


def analyze_roc_cate_condition(name, end_path, start_path, res_dir, outcome_operation, condition, th_list):

    casual_parser = CasualParser(start_path, end_path, outcome_operation=outcome_operation)

    dict_outcome = {}
    for outcome in tqdm(casual_parser.features):
        condition_df = casual_parser.iterate_condition_th(outcome, condition, th_list=th_list)
        condition_df.set_index('type', inplace=True)
        dict_outcome[outcome] = condition_df

    th_dict = {}
    for th in th_list:
        th_dict[str(round(th, 4))] = pd.DataFrame()

    for outcome in dict_outcome:
        for thresold_outcome in dict_outcome[outcome]:
            column = dict_outcome[outcome][[thresold_outcome]]
            current_outcome, current_thres = column.columns.values[0].split('_cond_th:')
            th_dict[current_thres][outcome] = column


    final_df = pd.DataFrame(index=['above_significant_outcomes', 'above_avg_ate_outcomes', 'above_percentage',
                                   'under_significant_outcomes', 'under_avg_ate_outcomes', 'under_percentage',])
    for thres in th_dict:
        above_sig_ate_outcomes = np.sum(th_dict[thres].T['above_TH_pvalue_hist'] <= 0.05)
        above_average_ate_outcomes = np.mean(th_dict[thres].T['above_TH_ate'])
        above_percentage_outcomes = np.mean(th_dict[thres].T['above_TH_percentage'])

        under_sig_ate_outcomes = np.sum(th_dict[thres].T['under_equal_TH_pvalue_hist'] <= 0.05)
        under_average_ate_outcomes = np.mean(th_dict[thres].T['under_equal_TH_ate'])
        under_percentage_outcomes = np.mean(th_dict[thres].T['under_equal_TH_percentage'])

        final_df[thres] = (above_sig_ate_outcomes, above_average_ate_outcomes, above_percentage_outcomes,
                           under_sig_ate_outcomes, under_average_ate_outcomes, under_percentage_outcomes)

    data_type_res_dir = os.path.join(res_dir, name, 'analysis')
    if not os.path.exists(data_type_res_dir):
        os.makedirs(data_type_res_dir)

    final_df.to_csv(os.path.join(data_type_res_dir, f"{condition}_different_threshold.csv"))

if __name__ == '__main__':
    main()
