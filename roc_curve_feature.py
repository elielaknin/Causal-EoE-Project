import argparse
import os
import pandas as pd
import numpy as np
import utils
from tqdm import tqdm
from calc_casual_scores import analyze_roc_cate_condition


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

    # condition = 'BZH_spatial_th_0_score'
    # quantile_list = np.linspace(0.845, 1, 100)

    condition = 'average_symptoms_score'
    quantile_list = np.linspace(0.18, 0.73, 100)

    # condition = 'GradeLaminaPropriaFibros'
    # quantile_list = np.linspace(0, 3, 10)

    # condition = 'StageLaminaPropriaFibros'
    # quantile_list = np.linspace(0, 10, 10)

    res_dir = os.path.join(args.output_dir_path, 'casual_results')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    analyze_roc_cate_condition('patients_standardization', patient_6weeks_path, patient_start_path, res_dir,
                               'standardization', condition, quantile_list)



if __name__ == '__main__':
    main()
