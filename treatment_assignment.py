import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

from casual_parser import CasualParser


def parse_args():
    parser = argparse.ArgumentParser(description='This script is ...'
                                     , formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("data_dir_path")
    parser.add_argument("output_dir_path")
    args = parser.parse_args()
    return args


def predict_data_type(name, end_path, start_path, res_dir, outcome_operation):
    data_type_res_dir = os.path.join(res_dir, name)
    if not os.path.exists(data_type_res_dir):
        os.makedirs(data_type_res_dir)

    raw_res_dir = os.path.join(data_type_res_dir, 'raw_predictions')
    if not os.path.exists(raw_res_dir):
        os.makedirs(raw_res_dir)

    params_res_dir = os.path.join(data_type_res_dir, 'best_params')
    if not os.path.exists(params_res_dir):
        os.makedirs(params_res_dir)

    casual_parser = CasualParser(start_path, end_path, outcome_operation=outcome_operation)

    overall_df = pd.DataFrame()
    overall_df["policy_values"] = ["outcome", "inputs_type", "random_policy_value", "all_treated_policy_value",
                                   "all_untreated_policy_value", "s_learner_policy_value", "t_learner_policy_value"]

    # data_to_use = 'AI' # All, AI, HSS, Symptoms, EREF
    # casual_parser.features['PEC_distribution max score', 'PeakCount']
    for model_type in ['mlp', 'xgboost']:
        for outcome in tqdm(['PEC_distribution max score', 'PeakCount']):
            print(outcome)
            outcome_dir = os.path.join(raw_res_dir, outcome)
            if not os.path.exists(outcome_dir):
                os.makedirs(outcome_dir)

            for data_to_use in ['AI', 'HSS']:
                outcome_data = np.array(casual_parser.outcome_df[outcome])

                fill_empty = 'mean'
                if model_type == 'xgboost':
                    fill_empty = None
                seed = 1
                s_pred, t_pred, ind_not_nan = casual_parser.predict_total_policy_value_grid_search(outcome, 6,
                                                                                                   data_to_use=data_to_use,
                                                                                                   model_type=model_type,
                                                                                                   seed=seed,
                                                                                                   fill_empty=fill_empty)
                s_policy = s_pred['best_s_policy']
                t_policy = t_pred['best_t_policy']
                np.random.seed(seed)
                rand_policy = np.random.randint(0, 2, size=len(s_policy)).astype(int)
                treated_policy = np.ones(len(s_policy), dtype=int)
                untreated_policy = np.zeros(len(s_policy), dtype=int)
                gt = np.array((casual_parser.meta_data_df[
                                   casual_parser.treatment_assignment_header] == casual_parser.treated_name)).astype(
                    int)

                raw_df = casual_parser.meta_data_df
                raw_df['outcome'] = outcome_data
                raw_df['real_treatment_assignment'] = gt
                raw_df['random_assignment'] = rand_policy
                raw_df['all_treated_assignment'] = treated_policy
                raw_df['all_untreated_assignment'] = untreated_policy
                raw_df['s_learner_assignment'] = s_policy
                raw_df['t_learner_assignment'] = t_policy
                raw_df.to_csv(os.path.join(outcome_dir, f"{data_to_use}.csv"), index_label=False)

                # get policy values
                outcome_data_f = outcome_data[ind_not_nan]
                rand_policy_val = np.mean(outcome_data_f[rand_policy[ind_not_nan] == gt[ind_not_nan]])
                treated_policy_val = np.mean(outcome_data_f[treated_policy[ind_not_nan] == gt[ind_not_nan]])
                untreated_policy_val = np.mean(outcome_data_f[untreated_policy[ind_not_nan] == gt[ind_not_nan]])
                s_policy_val = np.mean(outcome_data_f[s_policy[ind_not_nan] == gt[ind_not_nan]])
                t_policy_val = np.mean(outcome_data_f[t_policy[ind_not_nan] == gt[ind_not_nan]])

                overall_df[f"{outcome}-{data_to_use}"] = [outcome, data_to_use, rand_policy_val, treated_policy_val,
                                                          untreated_policy_val, s_policy_val, t_policy_val]

                with open(os.path.join(params_res_dir,
                                       f'outcome_{outcome}_model_{model_type}_input_{data_to_use}_s_learner.txt'),
                          'w') as convert_file:
                    convert_file.write(json.dumps(s_pred['best_s_params']))
                with open(os.path.join(params_res_dir,
                                       f'outcome_{outcome}_model_{model_type}_input_{data_to_use}_t_learner.txt'),
                          'w') as convert_file:
                    convert_file.write(json.dumps(t_pred['best_t_params']))

        overall_df.to_csv(os.path.join(data_type_res_dir, f"overall_policy_values_{model_type}.csv"), index_label=False)


def main():
    args = parse_args()
    data_dir_path = args.data_dir_path

    patient_6weeks_path = os.path.join(data_dir_path, "patients_6weeks_causal_data.csv")
    patient_start_path = os.path.join(data_dir_path, "patients_baseline_causal_data.csv")

    res_dir = os.path.join(args.output_dir_path, 'treatment_assignment_results')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    predict_data_type('overall_policy', patient_6weeks_path, patient_start_path, res_dir, 'standardization')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
