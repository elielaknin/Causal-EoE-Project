import pandas as pd
import numpy as np
import random
from scipy import stats
import xgboost
from sklearn import preprocessing, metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import itertools

random.seed(0)


class CasualParser:

    def __init__(self, df_start_path, df_end_path, outcome_operation=None):

        self.df_start_path = df_start_path
        self.df_end_path = df_end_path
        self.load_and_arrange_dfs()

        self.treatment_assignment_header = 'Treatment'
        self.treated_name = '6FED'
        self.untreated_name = '1FED'

        self.start_mean = self.df_start.mean()
        self.start_std = self.df_start.std()

        if outcome_operation == 'abs_distance':
            self.outcome_df = self.df_end - self.df_start
        elif outcome_operation == 'ratio_distance':
            self.outcome_df = (self.df_end - self.df_start) / self.df_start.mean()
        elif outcome_operation == 'standardization':
            self.outcome_df = (self.df_end - self.df_start.mean()) / self.df_start.std()
        elif outcome_operation is None:
            self.outcome_df = self.df_end
        else:
            raise Exception("Not supported operation")

        self.num_samples = len(self.outcome_df)

    def get_start_mean(self, outcome):
        return self.start_mean[outcome]

    def get_start_std(self, outcome):
        return self.start_std[outcome]

    def load_and_arrange_dfs(self):
        meta_data_end_idx = 7
        df_start = pd.read_csv(self.df_start_path)
        df_end = pd.read_csv(self.df_end_path)

        # make sure all meta-data except visit-time are matching
        if not (df_start.iloc[:, :meta_data_end_idx - 4] == df_end.iloc[:, :meta_data_end_idx - 4]).all().all():
            raise Exception("meta data isn't corresponding in two dataframes")

        self.meta_data_df = df_start.iloc[:, :meta_data_end_idx]
        self.df_start = df_start.iloc[:, meta_data_end_idx:]
        self.df_end = df_end.iloc[:, meta_data_end_idx:]
        self.features = list(self.df_start.columns)

    def split_treatment(self, df):

        treated_df = df[self.meta_data_df[self.treatment_assignment_header] == self.treated_name]
        untreated_df = df[self.meta_data_df[self.treatment_assignment_header] == self.untreated_name]

        return treated_df, untreated_df

    def calc_ate(self, dfs_dict, outcome):

        # calc total size of dataset
        dataset_size = 0
        for key in sorted(dfs_dict.keys()):
            dataset_size += len(dfs_dict[key])
        ate_dict = {}
        for key in sorted(dfs_dict.keys()):
            treated_df, untreated_df = self.split_treatment(dfs_dict[key])

            treated_outcome_df = treated_df[outcome].dropna()
            untreated_outcome_df = untreated_df[outcome].dropna()

            pval_on_ate_ttest2_less = stats.ttest_ind(treated_outcome_df, untreated_outcome_df, equal_var=False,
                                                      alternative='less').pvalue
            pval_on_ate_ttest2_greater = stats.ttest_ind(treated_outcome_df, untreated_outcome_df, equal_var=False,
                                                         alternative='greater').pvalue

            treated_apo_ls, untreated_apo_ls, ate_ls, pval_ls = [], [], [], []
            for i in range(1000):
                random_treated_sample = random.choices(list(treated_outcome_df), k=len(treated_outcome_df))
                random_treated_mean = np.mean(random_treated_sample)
                random_untreated_sample = random.choices(list(untreated_outcome_df), k=len(untreated_outcome_df))
                random_untreated_mean = np.mean(random_untreated_sample)

                treated_apo_ls.append(random_treated_mean)
                untreated_apo_ls.append(random_untreated_mean)
                ate_ls.append(random_treated_mean - random_untreated_mean)

            # import matplotlib.pyplot as plt
            # plt.hist(ate_ls, bins=20)
            # plt.show()

            ate_dict[f"{key}_treated_apo"] = np.mean(treated_apo_ls)
            ate_dict[f"{key}_untreated_apo"] = np.mean(untreated_apo_ls)
            ate_dict[f"{key}_ate"] = np.mean(ate_ls)
            ate_dict[f"{key}_CI_95_low"] = np.quantile(ate_ls, 0.025)
            ate_dict[f"{key}_CI_95_high"] = np.quantile(ate_ls, 0.975)
            ate_dict[f"{key}_CI_99_low"] = np.quantile(ate_ls, 0.005)
            ate_dict[f"{key}_CI_99_high"] = np.quantile(ate_ls, 0.995)
            ate_dict[f"{key}_min"] = np.min(ate_ls)
            ate_dict[f"{key}_max"] = np.max(ate_ls)
            if np.mean(ate_ls) > 0:
                ate_dict[f"{key}_pvalue_hist"] = np.sum(np.array(ate_ls) <= 0) / len(ate_ls)
                ate_dict[f"{key}_pvalue_ttest2"] = pval_on_ate_ttest2_greater
            else:
                ate_dict[f"{key}_pvalue_hist"] = np.sum(np.array(ate_ls) >= 0) / len(ate_ls)
                ate_dict[f"{key}_pvalue_ttest2"] = pval_on_ate_ttest2_less

            ate_dict[f"{key}_percentage"] = (len(dfs_dict[key]) / dataset_size) * 100
            ate_dict[f"{key}_num_active"] = len(dfs_dict[key].dropna(subset=[outcome]))

        return ate_dict

    def calc_cate(self, outcome_col, conditional_col=None, conditional_th=None):

        outcome_df_dict = {}

        # apply condition
        if conditional_col is not None:

            # apply binary condition
            if conditional_th is not None:
                outcome_df_dict[f'above_{conditional_th}'] = self.outcome_df[
                    self.df_start[conditional_col] > conditional_th].dropna(subset=[conditional_col])
                outcome_df_dict[f'under_equal_{conditional_th}'] = self.outcome_df[
                    self.df_start[conditional_col] <= conditional_th].dropna(subset=[conditional_col])

            # apply condition to all classes
            else:
                condition_vals = list(self.df_start[conditional_col].unique())
                for val in condition_vals:
                    if np.isnan(val):
                        continue
                    else:
                        outcome_df_dict[str(val)] = self.outcome_df[self.df_start[conditional_col] == val].dropna(
                            subset=[conditional_col])

        # no condition
        else:
            outcome_df_dict['all'] = self.outcome_df

        cate_dict = self.calc_ate(outcome_df_dict, outcome_col)
        cate_df = pd.DataFrame(columns=['type', outcome_col])
        cate_df['type'] = list(cate_dict.keys())
        cate_df['value'] = [cate_dict[key] for key in cate_dict.keys()]
        return cate_df

    def iterate_condition_th(self, outcome_col, conditional_col, th_list=None, quantile_list=None):

        if th_list is None and quantile_list is None:
            th_list = np.linspace(self.df_start[conditional_col].min(), self.df_start[conditional_col].max(), 12)

        elif quantile_list is not None:
            th_list = list(round(self.df_start[conditional_col].quantile(quantile_list), 4))

        res_df = pd.DataFrame()
        res_df['type'] = ['above_TH_apo_treated', 'above_TH_apo_untreated', 'above_TH_ate',
                          'above_TH_CI_95_low', 'above_TH_CI_95_high', 'above_TH_CI_99_low', 'above_TH_CI_99_high',
                          'above_TH_min', 'above_TH_max', 'above_TH_pvalue_hist', 'above_TH_pvalue_ttest2',
                          'above_TH_percentage', 'above_TH_num_active',
                          'under_equal_TH_apo_treated', 'under_equal_TH_apo_untreated', 'under_equal_TH_ate',
                          'under_equal_TH_CI_95_low', 'under_equal_TH_CI_95_high', 'under_equal_TH_CI_99_low',
                          'under_equal_TH_CI_99_high', 'under_equal_TH_min', 'under_equal_TH_max',
                          'under_equal_TH_pvalue_hist', 'under_equal_TH_pvalue_ttest2', 'under_equal_TH_percentage',
                          'under_equal_TH_num_active']

        for conditional_th in th_list:
            cate_df = self.calc_cate(outcome_col, conditional_col=conditional_col,
                                     conditional_th=conditional_th)

            res_df[f"{conditional_col}_cond_th:{round(conditional_th, 4)}"] = cate_df['value']

        return res_df

    def predict_ate_from_learner_features(self, outcome, num_folds, data_to_use):
        # train on all data

        policy_dict = {}
        slearner_policy_val_ls, tlearner_policy_val_ls, treated_apo_ls, untreated_apo_ls = [], [], [], []
        slearner_r2_score_ls, tlearner_r2_score_ls, s_learner_mse_ls, t_learner_mse_ls = [], [], [], []

        ind_not_nan_ls = list(self.outcome_df[outcome].dropna().index)

        if data_to_use == 'All':
            df_start_filtered = self.df_start
        elif data_to_use == 'AI':
            df_start_filtered = self.df_start.iloc[:, 0:45]
        elif data_to_use == 'HSS':
            df_start_filtered = self.df_start.iloc[:, 45:66]
        elif data_to_use == 'Symptoms':
            df_start_filtered = self.df_start.iloc[:, 67:75]
        elif data_to_use == 'EREF':
            df_start_filtered = self.df_start.iloc[:, 76:]
        else:
            raise Exception("Not supported data type")

        features_start_norm = preprocessing.StandardScaler().fit_transform(df_start_filtered)
        data = np.concatenate([np.expand_dims(np.array(self.outcome_df[outcome]), 1), features_start_norm], axis=1)[
               ind_not_nan_ls, :]
        treatment = np.array((self.meta_data_df[self.treatment_assignment_header] == self.treated_name)).astype(int)[
            ind_not_nan_ls]

        skf = StratifiedKFold(n_splits=num_folds, random_state=1, shuffle=True)
        for train_index, test_index in skf.split(data, treatment):
            train_all, test_all = data[train_index], data[test_index]
            train_data = train_all[:, 1:]
            test_data = test_all[:, 1:]
            train_outcome = train_all[:, :1]
            test_outcome = test_all[:, :1]
            train_treatment, test_treatment = treatment[train_index], treatment[test_index]

            # 'S-learner': 'booster': ['gbtree', 'gblinear', 'dart'], 'max_depth': [4, 6, 8, 10]
            s_learner_model = xgboost.XGBRegressor(booster='gbtree', max_depth=4)
            s_learner_in = np.concatenate([train_data, np.expand_dims(train_treatment, 1)], axis=1)
            s_learner_model.fit(s_learner_in, train_outcome)

            s_learner_in_predict = np.concatenate([test_data, np.expand_dims(test_treatment, 1)], axis=1)
            s_learner_predict = s_learner_model.predict(s_learner_in_predict)
            s_learner_r2 = metrics.r2_score(test_outcome, s_learner_predict)
            s_learner_mse = metrics.mean_squared_error(test_outcome, s_learner_predict)

            # predict S-learner policy value
            one_cols = np.ones((len(test_treatment), 1))
            zero_cols = np.zeros((len(test_treatment), 1))
            treated_predict_outcome = s_learner_model.predict(np.concatenate([test_data, one_cols], axis=1))
            untreated_predict_outcome = s_learner_model.predict(np.concatenate([test_data, zero_cols], axis=1))
            policy = (treated_predict_outcome < untreated_predict_outcome).astype(int)
            slearner_policy_value = np.mean(test_outcome[policy == test_treatment])

            # 'T-learner':
            t_learner_treated_model = xgboost.XGBRegressor(booster='gbtree', max_depth=4)
            t_learner_treated_in = train_data[train_treatment == 1]
            t_learner_treated_outcome = train_outcome[train_treatment == 1]
            t_learner_treated_model.fit(t_learner_treated_in, t_learner_treated_outcome)
            t_learner_treated_predict = t_learner_treated_model.predict(test_data[test_treatment == 1])
            t_learner_treated_r2 = metrics.r2_score(test_outcome[test_treatment == 1], t_learner_treated_predict)
            t_learner_treated_mse = metrics.mean_squared_error(test_outcome[test_treatment == 1],
                                                               t_learner_treated_predict)

            t_learner_untreated_model = xgboost.XGBRegressor(booster='gbtree', max_depth=4)
            t_learner_untreated_in = train_data[train_treatment == 0]
            t_learner_untreated_outcome = train_outcome[train_treatment == 0]
            t_learner_untreated_model.fit(t_learner_untreated_in, t_learner_untreated_outcome)
            t_learner_untreated_predict = t_learner_untreated_model.predict(test_data[test_treatment == 0])
            t_learner_untreated_r2 = metrics.r2_score(test_outcome[test_treatment == 0], t_learner_untreated_predict)
            t_learner_untreated_mse = metrics.mean_squared_error(test_outcome[test_treatment == 0],
                                                                 t_learner_untreated_predict)

            # predict T-learner policy value
            treated_predict_outcome = t_learner_treated_model.predict(test_data)
            untreated_predict_outcome = t_learner_untreated_model.predict(test_data)
            policy = (treated_predict_outcome < untreated_predict_outcome).astype(int)
            tlearner_policy_value = np.mean(test_outcome[policy == test_treatment])

            # all assign value
            treated_apo = np.mean(test_outcome[test_treatment == 1])
            untreated_apo = np.mean(test_outcome[test_treatment == 0])

            t_learner_r2 = (t_learner_treated_r2 + t_learner_untreated_r2) / 2
            t_learner_mse = (t_learner_treated_mse + t_learner_untreated_mse) / 2

            slearner_policy_val_ls.append(slearner_policy_value)
            tlearner_policy_val_ls.append(tlearner_policy_value)
            treated_apo_ls.append(treated_apo)
            untreated_apo_ls.append(untreated_apo)
            slearner_r2_score_ls.append(s_learner_r2)
            tlearner_r2_score_ls.append(t_learner_r2)
            s_learner_mse_ls.append(s_learner_mse)
            t_learner_mse_ls.append(t_learner_mse)

        policy_dict[f"{outcome}_treated_apo_mean"] = np.mean(treated_apo_ls)
        policy_dict[f"{outcome}_treated_apo_std"] = np.std(treated_apo_ls)
        policy_dict[f"{outcome}_untreated_apo_mean"] = np.mean(untreated_apo_ls)
        policy_dict[f"{outcome}_untreated_apo_std"] = np.std(untreated_apo_ls)
        policy_dict[f"{outcome}_policy_val_t_mean"] = np.mean(tlearner_policy_val_ls)
        policy_dict[f"{outcome}_policy_val_t_std"] = np.std(tlearner_policy_val_ls)
        policy_dict[f"{outcome}_policy_val_s_mean"] = np.mean(slearner_policy_val_ls)
        policy_dict[f"{outcome}_policy_val_s_std"] = np.std(slearner_policy_val_ls)
        policy_dict[f"{outcome}_t_r2_cv_mean"] = np.mean(tlearner_r2_score_ls)
        policy_dict[f"{outcome}_t_mse_mean"] = np.std(t_learner_mse_ls)
        policy_dict[f"{outcome}_s_r2_cv_mean"] = np.mean(slearner_r2_score_ls)
        policy_dict[f"{outcome}_s_mse_mean"] = np.std(s_learner_mse_ls)
        policy_dict[f"{outcome}_start_mean"] = self.get_start_mean(outcome)
        policy_dict[f"{outcome}_start_std"] = self.get_start_std(outcome)

        return policy_dict

    def predict_ate_from_learner_features_GridSearchCV(self, outcome, num_folds, data_to_use):
        # train on all data

        policy_dict = {}
        boostrap_policy_dict = {}

        ind_not_nan_ls = list(self.outcome_df[outcome].dropna().index)

        if data_to_use == 'All':
            df_start_filtered = self.df_start
        elif data_to_use == 'AI':
            df_start_filtered = self.df_start.iloc[:, 0:38]
        elif data_to_use == 'HSS':
            df_start_filtered = self.df_start.iloc[:, 38:60]
        elif data_to_use == 'Symptoms':
            df_start_filtered = self.df_start.iloc[:, 60:67]
        elif data_to_use == 'EREF':
            df_start_filtered = self.df_start.iloc[:, 67:]
        else:
            raise Exception("Not supported data type")

        param = {'booster': ['gbtree', 'gblinear', 'dart'],
                 'max_depth': [2, 4, 6, 8, 10],
                 'eta': [0.1, 0.3, 0.5],
                 'gamma': [0, 0.1, 0.25]}

        # param = {'booster': ['gbtree', 'dart'],
        #          'max_depth': [2, 10],
        #          'eta': [0.5],
        #          'gamma': [0, 0.1]}

        features_start_norm = preprocessing.StandardScaler().fit_transform(df_start_filtered)
        data_outcome_features = np.concatenate(
            [np.expand_dims(np.array(self.outcome_df[outcome]), 1), features_start_norm], axis=1)[ind_not_nan_ls, :]
        features_data = data_outcome_features[:, 1:]
        outcome_in = data_outcome_features[:, :1]

        treatment = np.array((self.meta_data_df[self.treatment_assignment_header] == self.treated_name)).astype(int)[
            ind_not_nan_ls]

        # find best params for 'S-learner':
        clf = xgboost.XGBRegressor(verbosity=0, use_label_encoder=False)
        s_learner_model = GridSearchCV(clf, param, scoring='r2', cv=5, verbose=0)
        s_learner_in = np.concatenate([features_data, np.expand_dims(treatment, 1)], axis=1)
        s_learner_model.fit(s_learner_in, outcome_in)
        best_params_s_learner = s_learner_model.best_params_

        # find best params for treated 'T-learner':
        clf_treated = xgboost.XGBRegressor(verbosity=0, use_label_encoder=False)
        t_learner_treated_model = GridSearchCV(clf_treated, param, scoring='r2', cv=5, verbose=0)
        t_learner_treated_in = features_data[treatment == 1]
        t_learner_treated_outcome = outcome_in[treatment == 1]
        t_learner_treated_model.fit(t_learner_treated_in, t_learner_treated_outcome)
        best_params_t_learner_treated = t_learner_treated_model.best_params_

        # find best params for untreated 'T-learner':
        clf_untreated = xgboost.XGBRegressor(verbosity=0, use_label_encoder=False)
        t_learner_untreated_model = GridSearchCV(clf_untreated, param, scoring='r2', cv=5, verbose=0)
        t_learner_untreated_in = features_data[treatment == 0]
        t_learner_untreated_outcome = outcome_in[treatment == 0]
        t_learner_untreated_model.fit(t_learner_untreated_in, t_learner_untreated_outcome)
        best_params_t_learner_untreated = t_learner_untreated_model.best_params_

        slearner_policy_val_ls, tlearner_policy_val_ls, treated_apo_ls, untreated_apo_ls = [], [], [], []
        slearner_r2_score_ls, tlearner_r2_score_ls, s_learner_mse_ls, t_learner_mse_ls = [], [], [], []
        random_apo_ls = []

        boostrap_number = 100
        for seed in range(boostrap_number):
            seed_slearner_policy_val_ls, seed_tlearner_policy_val_ls, seed_treated_apo_ls, seed_untreated_apo_ls = [], [], [], []
            seed_slearner_r2_score_ls, seed_tlearner_r2_score_ls, seed_s_learner_mse_ls, seed_t_learner_mse_ls = [], [], [], []
            seed_random_apo_ls = []

            skf = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True)
            for train_index, test_index in skf.split(data_outcome_features, treatment):
                train_all, test_all = data_outcome_features[train_index], data_outcome_features[test_index]
                train_data = train_all[:, 1:]
                test_data = test_all[:, 1:]
                train_outcome = train_all[:, :1]
                test_outcome = test_all[:, :1]
                train_treatment, test_treatment = treatment[train_index], treatment[test_index]

                # 'S-learner':
                s_learner_model = xgboost.XGBRegressor(booster=best_params_s_learner['booster'],
                                                       max_depth=best_params_s_learner['max_depth'],
                                                       eta=best_params_s_learner['eta'],
                                                       gamma=best_params_s_learner['gamma'])
                s_learner_in = np.concatenate([train_data, np.expand_dims(train_treatment, 1)], axis=1)
                s_learner_model.fit(s_learner_in, train_outcome)

                s_learner_in_predict = np.concatenate([test_data, np.expand_dims(test_treatment, 1)], axis=1)
                s_learner_predict = s_learner_model.predict(s_learner_in_predict)
                s_learner_r2 = metrics.r2_score(test_outcome, s_learner_predict)
                s_learner_mse = metrics.mean_squared_error(test_outcome, s_learner_predict)

                # predict S-learner policy value
                one_cols = np.ones((len(test_treatment), 1))
                zero_cols = np.zeros((len(test_treatment), 1))
                treated_predict_outcome = s_learner_model.predict(np.concatenate([test_data, one_cols], axis=1))
                untreated_predict_outcome = s_learner_model.predict(np.concatenate([test_data, zero_cols], axis=1))
                policy = (treated_predict_outcome < untreated_predict_outcome).astype(int)
                slearner_policy_value = np.mean(test_outcome[policy == test_treatment])

                # 'T-learner':
                t_learner_treated_model = xgboost.XGBRegressor(booster=best_params_t_learner_treated['booster'],
                                                               max_depth=best_params_t_learner_treated['max_depth'],
                                                               eta=best_params_t_learner_treated['eta'],
                                                               gamma=best_params_t_learner_treated['gamma'])
                t_learner_treated_in = train_data[train_treatment == 1]
                t_learner_treated_outcome = train_outcome[train_treatment == 1]
                t_learner_treated_model.fit(t_learner_treated_in, t_learner_treated_outcome)
                t_learner_treated_predict = t_learner_treated_model.predict(test_data[test_treatment == 1])
                t_learner_treated_r2 = metrics.r2_score(test_outcome[test_treatment == 1], t_learner_treated_predict)
                t_learner_treated_mse = metrics.mean_squared_error(test_outcome[test_treatment == 1],
                                                                   t_learner_treated_predict)

                t_learner_untreated_model = xgboost.XGBRegressor(booster=best_params_t_learner_untreated['booster'],
                                                                 max_depth=best_params_t_learner_untreated['max_depth'],
                                                                 eta=best_params_t_learner_untreated['eta'],
                                                                 gamma=best_params_t_learner_untreated['gamma'])
                t_learner_untreated_in = train_data[train_treatment == 0]
                t_learner_untreated_outcome = train_outcome[train_treatment == 0]
                t_learner_untreated_model.fit(t_learner_untreated_in, t_learner_untreated_outcome)
                t_learner_untreated_predict = t_learner_untreated_model.predict(test_data[test_treatment == 0])
                t_learner_untreated_r2 = metrics.r2_score(test_outcome[test_treatment == 0],
                                                          t_learner_untreated_predict)
                t_learner_untreated_mse = metrics.mean_squared_error(test_outcome[test_treatment == 0],
                                                                     t_learner_untreated_predict)

                # predict T-learner policy value
                treated_predict_outcome = t_learner_treated_model.predict(test_data)
                untreated_predict_outcome = t_learner_untreated_model.predict(test_data)
                policy = (treated_predict_outcome < untreated_predict_outcome).astype(int)
                tlearner_policy_value = np.mean(test_outcome[policy == test_treatment])

                # random policy assigned
                random_treatment = np.array(test_treatment, copy=True)
                np.random.shuffle(random_treatment)
                random_apo = np.mean(test_outcome[random_treatment == test_treatment])

                # all assign value
                treated_apo = np.mean(test_outcome[test_treatment == 1])
                untreated_apo = np.mean(test_outcome[test_treatment == 0])

                t_learner_r2 = (t_learner_treated_r2 + t_learner_untreated_r2) / 2
                t_learner_mse = (t_learner_treated_mse + t_learner_untreated_mse) / 2

                seed_slearner_policy_val_ls.append(slearner_policy_value)
                seed_tlearner_policy_val_ls.append(tlearner_policy_value)
                seed_treated_apo_ls.append(treated_apo)
                seed_untreated_apo_ls.append(untreated_apo)
                seed_random_apo_ls.append(random_apo)
                seed_slearner_r2_score_ls.append(s_learner_r2)
                seed_tlearner_r2_score_ls.append(t_learner_r2)
                seed_s_learner_mse_ls.append(s_learner_mse)
                seed_t_learner_mse_ls.append(t_learner_mse)

            slearner_policy_val_ls.append(np.mean(seed_slearner_policy_val_ls))
            tlearner_policy_val_ls.append(np.mean(seed_tlearner_policy_val_ls))
            treated_apo_ls.append(np.mean(seed_treated_apo_ls))
            untreated_apo_ls.append(np.mean(seed_untreated_apo_ls))
            slearner_r2_score_ls.append(np.mean(seed_slearner_r2_score_ls))
            tlearner_r2_score_ls.append(np.mean(seed_tlearner_r2_score_ls))
            s_learner_mse_ls.append(np.mean(seed_s_learner_mse_ls))
            t_learner_mse_ls.append(np.mean(seed_t_learner_mse_ls))
            random_apo_ls.append(np.mean(seed_random_apo_ls))

        policy_dict[f"{outcome}_treated_apo_mean"] = np.mean(treated_apo_ls)
        policy_dict[f"{outcome}_treated_apo_std"] = np.std(treated_apo_ls)
        policy_dict[f"{outcome}_untreated_apo_mean"] = np.mean(untreated_apo_ls)
        policy_dict[f"{outcome}_untreated_apo_std"] = np.std(untreated_apo_ls)
        policy_dict[f"{outcome}_policy_val_t_mean"] = np.mean(tlearner_policy_val_ls)
        policy_dict[f"{outcome}_policy_val_t_std"] = np.std(tlearner_policy_val_ls)
        policy_dict[f"{outcome}_policy_val_s_mean"] = np.mean(slearner_policy_val_ls)
        policy_dict[f"{outcome}_policy_val_s_std"] = np.std(slearner_policy_val_ls)
        policy_dict[f"{outcome}_policy_val_random_mean"] = np.mean(random_apo_ls)
        policy_dict[f"{outcome}_policy_val_random_std"] = np.std(random_apo_ls)
        policy_dict[f"{outcome}_t_r2_cv_mean"] = np.mean(tlearner_r2_score_ls)
        policy_dict[f"{outcome}_t_mse_mean"] = np.std(t_learner_mse_ls)
        policy_dict[f"{outcome}_s_r2_cv_mean"] = np.mean(slearner_r2_score_ls)
        policy_dict[f"{outcome}_s_mse_mean"] = np.std(s_learner_mse_ls)
        policy_dict[f"{outcome}_start_mean"] = self.get_start_mean(outcome)
        policy_dict[f"{outcome}_start_std"] = self.get_start_std(outcome)

        boostrap_policy_dict[f"{outcome}_treated_apo"] = treated_apo_ls
        boostrap_policy_dict[f"{outcome}_untreated_apo"] = untreated_apo_ls
        boostrap_policy_dict[f"{outcome}_policy_val_t"] = tlearner_policy_val_ls
        boostrap_policy_dict[f"{outcome}_policy_val_s"] = slearner_policy_val_ls
        boostrap_policy_dict[f"{outcome}_policy_val_random"] = random_apo_ls

        t_minus_treated_pvalue = np.sum(
            [np.array(tlearner_policy_val_ls) - np.array(treated_apo_ls) > 0]) / boostrap_number
        t_minus_untreated_pvalue = np.sum(
            [np.array(tlearner_policy_val_ls) - np.array(untreated_apo_ls) > 0]) / boostrap_number
        t_minus_random_pvalue = np.sum(
            [np.array(tlearner_policy_val_ls) - np.array(random_apo_ls) > 0]) / boostrap_number
        s_minus_treated_pvalue = np.sum(
            [np.array(slearner_policy_val_ls) - np.array(treated_apo_ls) > 0]) / boostrap_number
        s_minus_untreated_pvalue = np.sum(
            [np.array(slearner_policy_val_ls) - np.array(untreated_apo_ls) > 0]) / boostrap_number
        s_minus_random_pvalue = np.sum(
            [np.array(slearner_policy_val_ls) - np.array(random_apo_ls) > 0]) / boostrap_number

        pvalue_ls = [t_minus_treated_pvalue, t_minus_untreated_pvalue, t_minus_random_pvalue,
                     s_minus_treated_pvalue, s_minus_untreated_pvalue, s_minus_random_pvalue]

        return policy_dict, pvalue_ls, boostrap_policy_dict

    def predict_total_policy_value_grid_search(self, outcome, num_folds, data_to_use, seed=1, fill_empty='mean',
                                               model_type='mlp'):

        random.seed(seed)
        # train on all data
        ind_not_nan_ls = list(self.outcome_df[outcome].dropna().index)

        if data_to_use == 'All':
            df_start_filtered = self.df_start
        elif data_to_use == 'AI':
            df_start_filtered = self.df_start.iloc[:, 0:38]
        elif data_to_use == 'HSS':
            df_start_filtered = self.df_start.iloc[:, 38:60]
        elif data_to_use == 'Symptoms':
            df_start_filtered = self.df_start.iloc[:, 60:67]
        elif data_to_use == 'EREF':
            df_start_filtered = self.df_start.iloc[:, 67:]
        else:
            raise Exception("Not supported data type")

        if fill_empty == 'mean':
            for feature_name in df_start_filtered.columns:
                mean_value = df_start_filtered[feature_name].mean()
                df_start_filtered[feature_name].fillna(value=mean_value, inplace=True)

        features_start_norm = preprocessing.StandardScaler().fit_transform(df_start_filtered)
        data_outcome_features = np.concatenate(
            [np.expand_dims(np.array(self.outcome_df[outcome]), 1), features_start_norm], axis=1)[ind_not_nan_ls, :]
        features_data = data_outcome_features[:, 1:]
        outcome_in = data_outcome_features[:, :1]

        treatment = np.array((self.meta_data_df[self.treatment_assignment_header] == self.treated_name)).astype(int)[
            ind_not_nan_ls]

        # find best params for 'S-learner':
        if model_type == 'xgboost':
            # gc_model = xgboost.XGBRegressor(verbosity=0, use_label_encoder=False, random_state=seed)
            # gc_model_treated = xgboost.XGBRegressor(verbosity=0, use_label_encoder=False, random_state=seed)
            # gc_model_untreated = xgboost.XGBRegressor(verbosity=0, use_label_encoder=False, random_state=seed)
            gc_params = {
                'booster': ['gbtree', 'gblinear', 'dart'],
                'max_depth': [2, 4, 6, 8, 10],
                'eta': [0.1, 0.2, 0.3, 0.4, 0.5],
                'gamma': [0, 0.1, 0.25, 0.5]}
            ml_cls = xgboost.XGBRegressor
        elif model_type == 'mlp':
            # gc_model = MLPRegressor(max_iter=100, random_state=seed)
            # gc_model_treated = MLPRegressor(max_iter=100, random_state=seed)
            # gc_model_untreated = MLPRegressor(max_iter=100, random_state=seed)
            gc_params = {
                'hidden_layer_sizes': [(), (10,), (50, 50, 50), (50, 100, 50), (100,), (20,), (50), (20, 50), (20, 20)],
                'activation': ['tanh', 'relu'],
                'solver': ['sgd', 'adam'],
                'alpha': [0.0005, 0.005, 0.05],
                'learning_rate': ['constant', 'adaptive'],
            }
            # gc_params = {
            #     'hidden_layer_sizes': [()],
            #     'activation': ['tanh'],
            #     'solver': ['sgd'],
            #     'alpha': [0.0001],
            #     'learning_rate': ['constant'],
            # }
            ml_cls = MLPRegressor

        keys, values = zip(*gc_params.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # s_learner_model = GridSearchCV(gc_model, gc_params, scoring='r2', cv=num_folds, verbose=0)
        # s_learner_in = np.concatenate([features_data, np.expand_dims(treatment, 1)], axis=1)
        # s_learner_model.fit(s_learner_in, outcome_in)
        # best_params_s_learner = s_learner_model.best_params_
        #
        # # find best params for treated 'T-learner':
        # t_learner_treated_model = GridSearchCV(gc_model_treated, gc_params, scoring='r2', cv=num_folds, verbose=0)
        # t_learner_treated_in = features_data[treatment == 1]
        # t_learner_treated_outcome = outcome_in[treatment == 1]
        # t_learner_treated_model.fit(t_learner_treated_in, t_learner_treated_outcome)
        # best_params_t_learner_treated = t_learner_treated_model.best_params_
        #
        # # find best params for untreated 'T-learner':
        # t_learner_untreated_model = GridSearchCV(gc_model_untreated, gc_params, scoring='r2', cv=num_folds, verbose=0)
        # t_learner_untreated_in = features_data[treatment == 0]
        # t_learner_untreated_outcome = outcome_in[treatment == 0]
        # t_learner_untreated_model.fit(t_learner_untreated_in, t_learner_untreated_outcome)
        # best_params_t_learner_untreated = t_learner_untreated_model.best_params_

        outcome_data = np.array(self.outcome_df[outcome])
        outcome_data_f = outcome_data[ind_not_nan_ls]
        gt = np.array((self.meta_data_df[
                           self.treatment_assignment_header] == self.treated_name)).astype(int)

        s_pred = {"best_s_policy_val": np.Inf, "best_s_policy": [], "best_s_params": []}
        skf = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True)
        for params_perm in permutations_dicts:
            s_policy = np.array([None for _ in range(self.num_samples)])
            for train_index, test_index in skf.split(data_outcome_features, treatment):
                all_data_test_index = np.array(ind_not_nan_ls)[test_index]
                train_all, test_all = data_outcome_features[train_index], data_outcome_features[test_index]
                train_data = train_all[:, 1:]
                test_data = test_all[:, 1:]
                train_outcome = train_all[:, :1]
                train_treatment, test_treatment = treatment[train_index], treatment[test_index]

                s_learner_model = ml_cls(**params_perm)

                # 'S-learner':
                s_learner_in = np.concatenate([train_data, np.expand_dims(train_treatment, 1)], axis=1)
                s_learner_model.fit(s_learner_in, train_outcome)

                # predict S-learner policy
                one_cols = np.ones((len(test_treatment), 1))
                zero_cols = np.zeros((len(test_treatment), 1))
                treated_predict_outcome = s_learner_model.predict(np.concatenate([test_data, one_cols], axis=1))
                untreated_predict_outcome = s_learner_model.predict(np.concatenate([test_data, zero_cols], axis=1))
                policy = (treated_predict_outcome < untreated_predict_outcome).astype(int)
                s_policy[all_data_test_index] = policy

            s_policy_val = np.mean(outcome_data_f[s_policy[ind_not_nan_ls] == gt[ind_not_nan_ls]])
            if s_policy_val < s_pred["best_s_policy_val"]:
                s_pred["best_s_policy_val"] = s_policy_val
                s_pred["best_s_policy"] = s_policy
                s_pred["best_s_params"] = params_perm

        t_pred = {"best_t_policy_val": np.Inf, "best_t_policy": [], "best_t_params": []}
        skf = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True)
        for treated_params_perm in permutations_dicts:
            for untreated_params_perm in permutations_dicts:
                t_policy = np.array([None for _ in range(self.num_samples)])
                for train_index, test_index in skf.split(data_outcome_features, treatment):
                    all_data_test_index = np.array(ind_not_nan_ls)[test_index]
                    train_all, test_all = data_outcome_features[train_index], data_outcome_features[test_index]
                    train_data = train_all[:, 1:]
                    test_data = test_all[:, 1:]
                    train_outcome = train_all[:, :1]
                    train_treatment, test_treatment = treatment[train_index], treatment[test_index]

                    t_learner_treated_model = ml_cls(**treated_params_perm)
                    t_learner_untreated_model = ml_cls(**untreated_params_perm)

                    # 'T-learner':
                    t_learner_treated_in = train_data[train_treatment == 1]
                    t_learner_treated_outcome = train_outcome[train_treatment == 1]
                    t_learner_treated_model.fit(t_learner_treated_in, t_learner_treated_outcome)

                    t_learner_untreated_in = train_data[train_treatment == 0]
                    t_learner_untreated_outcome = train_outcome[train_treatment == 0]
                    t_learner_untreated_model.fit(t_learner_untreated_in, t_learner_untreated_outcome)

                    # predict T-learner policy value
                    treated_predict_outcome = t_learner_treated_model.predict(test_data)
                    untreated_predict_outcome = t_learner_untreated_model.predict(test_data)
                    policy = (treated_predict_outcome < untreated_predict_outcome).astype(int)
                    t_policy[all_data_test_index] = policy

                t_policy_val = np.mean(outcome_data_f[t_policy[ind_not_nan_ls] == gt[ind_not_nan_ls]])
                if t_policy_val < t_pred["best_t_policy_val"]:
                    t_pred["best_t_policy_val"] = t_policy_val
                    t_pred["best_t_policy"] = t_policy
                    t_pred["best_t_params"] = {**{f"treated_{k}": v for k, v in treated_params_perm.items()},
                                               **{f"untreated_{k}": v for k, v in untreated_params_perm.items()}}

        return s_pred, t_pred, ind_not_nan_ls
