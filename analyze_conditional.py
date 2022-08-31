import argparse
import pandas as pd
import numpy as np
import os


# script that analyze the CATE results and find the best condition that have a
# maximum number of significant outcomes.
def parse_args():
    parser = argparse.ArgumentParser(description='This script is ...'
                                     , formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("data_file_path")
    args = parser.parse_args()
    return args


def analyze_conditional(data_cate_file_path):
    cate_df = pd.read_csv(data_cate_file_path)
    avg_cate = np.mean(cate_df['cate'])
    cate_df['sig_<0_condition'] = np.logical_and(cate_df['cate'] < 0, cate_df['pvalue'] <= 0.05)
    cate_df['sig_>0_condition'] = np.logical_and(cate_df['cate'] > 0, cate_df['pvalue'] <= 0.05)

    avg_cate_pval_05 = np.mean(cate_df[cate_df['pvalue'] <= 0.05]['cate'])
    cate_sign_under_zero_df = cate_df[cate_df['sig_<0_condition'] == True]
    cate_sign_above_zero_df = cate_df[cate_df['sig_>0_condition'] == True]

    condition_list_under_zero = list(cate_sign_under_zero_df['condition'].unique())
    condition_list_above_zero = list(cate_sign_above_zero_df['condition'].unique())

    dict_cond_under = {}
    for cond in condition_list_under_zero:
        tmp_df = cate_sign_under_zero_df[cate_sign_under_zero_df['condition'] == cond]
        dict_cond_under[cond] = (len(tmp_df), np.mean(tmp_df['cate']), np.std(tmp_df['cate']))

    condi_under_df = pd.DataFrame(dict_cond_under, index=['number_of_outcomes', 'avg_cate', 'std_cate']).T
    condi_under_df.sort_values('number_of_outcomes', ascending=False, inplace=True)

    dict_cond_above = {}
    for cond in condition_list_above_zero:
        tmp_df = cate_sign_above_zero_df[cate_sign_above_zero_df['condition'] == cond]
        dict_cond_above[cond] = (len(tmp_df), np.mean(tmp_df['cate']), np.std(tmp_df['cate']))

    condi_above_df = pd.DataFrame(dict_cond_above, index=['number_of_outcomes', 'avg_cate', 'std_cate']).T
    condi_above_df.sort_values('number_of_outcomes', ascending=False, inplace=True)

    conditinal_df = pd.concat([condi_above_df, condi_under_df])
    conditinal_df.sort_values('number_of_outcomes', ascending=False, inplace=True)

    analysis_res_dir = os.path.dirname(data_cate_file_path)
    conditinal_df.to_csv(os.path.join(analysis_res_dir, "number_of_outcomes_for_all_conditional.csv"))


def main():
    args = parse_args()
    data_cate_file_path = args.data_file_path
    analyze_conditional(data_cate_file_path)


if __name__ == '__main__':
    main()
