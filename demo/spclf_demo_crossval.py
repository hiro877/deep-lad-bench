#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import argparse
import logging
import numpy as np

from collections import defaultdict

from deeploglizer.models import SPClassifier
from deeploglizer.common.preprocess import FeatureExtractor
from deeploglizer.common.dataloader import load_sessions, log_dataset
from deeploglizer.common.utils import seed_everything, dump_final_results, dump_params, dump_store_results, dump_store_results_cm

import os


parser = argparse.ArgumentParser()

##### Model params
parser.add_argument("--model_name", default="SPClassifier", type=str)

parser.add_argument("--emb_dim", default=500, type=int)
parser.add_argument("--emb_sparsity", default=0.15, type=float)

parser.add_argument("--potential_radius", default=7, type=int)
parser.add_argument("--duty_cycle", default=1402, type=int)
parser.add_argument("--stimulus_thresh", default=6, type=int)
parser.add_argument("--potential_pct", default=0.1, type=float)
parser.add_argument("--local_density", default=0.2, type=float)
parser.add_argument("--boost_strength", default=7.0, type=float)
parser.add_argument("--syn_inc", default=0.14, type=float)
parser.add_argument("--syn_connect", default=0.5, type=float)
parser.add_argument("--syn_dec", default=0.02, type=float)
parser.add_argument("--min_overlap_duty", default=0.2, type=float)

##### Dataset params
parser.add_argument("--dataset", default="HDFS", type=str)
parser.add_argument(
    "--data_dir", default="../data/processed/HDFS_100k/hdfs_0.0_tar", type=str
)
parser.add_argument("--window_size", default=10, type=int)
parser.add_argument("--stride", default=1, type=int)

##### Input params
parser.add_argument("--feature_type", default="sequentials", type=str, choices=["sequentials", "semantics"])
parser.add_argument("--label_type", default="anomaly", type=str)
parser.add_argument("--use_tfidf", action="store_true")
parser.add_argument("--max_token_len", default=50, type=int)
parser.add_argument("--min_token_count", default=1, type=int)
parser.add_argument("--eval_type", default="window", type=str)

##### Others
parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument("--n_experiment", default=5, type=int)
parser.add_argument("--epoches", default=100, type=int)
parser.add_argument("--patience", default=3, type=int)
parser.add_argument("--is_validation", action='store_true')
parser.add_argument("--use_val_template_in_learning", action='store_true')
parser.add_argument("--wtp_records_path", default="wtp_records", type=str)
parser.add_argument("--is_epoches", action='store_true')

params = vars(parser.parse_args())



if __name__ == '__main__':
    is_shuffle = True
    use_all_template_in_learning = False
    print("is_shuffle: ", is_shuffle)
    print("use_all_template_in_learning: ", use_all_template_in_learning)
    seed_everything(params["random_seed"])

    session_train, session_test, session_val = load_sessions(data_dir=params["data_dir"], is_validation=params["is_validation"])

    ext = FeatureExtractor(**params)
    ext.is_print_dataset=True
    session_train = ext.fit_transform(session_train)

    dataset_train = log_dataset(
        session_train, feature_type=params["feature_type"], data_pct=1, shuffle=is_shuffle)

    os.makedirs(params["wtp_records_path"], exist_ok=True)
    if params["is_validation"]:
        if params["use_val_template_in_learning"]:
            ext.update_id2log_train(session_val)
        session_val = ext.transform(session_val, datatype="test")
        dataset_val = log_dataset(
            session_val, feature_type=params["feature_type"])

    if use_all_template_in_learning:
        ext.update_id2log_train(session_test)
    session_test = ext.transform(session_test, datatype="test")
    dataset_test = log_dataset(
        session_test, feature_type=params["feature_type"])

    # for BGL (2/3 is better for HDFS)
    if params["dataset"] == "BGL":
        params["column_dims"] = ((params["emb_dim"] // 3) * 5, (params["window_size"] // 3) * 5)
    elif params["dataset"] == "HDFS":
        params["column_dims"] = (params["emb_dim"], params["window_size"])
    params["n_templates"] = len(ext.ulog_train) + len(ext.ulog_new) + 2

    # model_save_path = dump_params(params)
    # dataset_train = log_dataset(
    #     session_train, feature_type=params["feature_type"], data_pct=1.0, shuffle=is_shuffle)
    # model = SPClassifier(meta_data=ext.meta_data, model_save_path=model_save_path, **params)
    # model.mem_prof(dataset_train=dataset_train)

    validation_ratio = 0.11
    print("validation_ratio: ", validation_ratio)
    # for pct in [0.1 * (i+1) for i in range(10)]:
    for pct in [1]:
        params["data_pct"] = pct
        model_save_path = dump_params(params)

        logging.info("{}Experiment data_pct:{}%{}".format("=" * 15, pct, "=" * 15))
        # dataset_train = log_dataset(
        #     session_train, feature_type=params["feature_type"], data_pct=pct, shuffle=is_shuffle)

        if params["is_validation"]:
            print("len(dataset_val): ", len(dataset_val))
            print("len(dataset_train): ", len(dataset_train))
            print("len(dataset_test): ", len(dataset_test))
        else:
            print("len(dataset_train): ", len(dataset_train))
            print("len(dataset_test): ", len(dataset_test))

        store_dict = defaultdict(list)
        best_f1 = -float("inf")
        for exp_num in range(params["n_experiment"]):
            logging.info("{}Experiment #{}{}".format("-" * 15, exp_num, "-" * 15))
            dataset_train = log_dataset(
                session_train, feature_type=params["feature_type"], data_pct=1.0, shuffle=is_shuffle)
            model = SPClassifier(meta_data=ext.meta_data, model_save_path=model_save_path, **params)
            model.is_learning_curve = True
            model.wtp_records_path=params["wtp_records_path"]

            # eval_results = model.fit(
            #     dataset_train=dataset_train,
            #     dataset_test=dataset_test
            # )
            if params["is_validation"]:
                if params["is_epoches"]:
                    eval_results = model.fit_epoches(
                        dataset_train=dataset_train,
                        dataset_test=dataset_val
                    )
                else:
                    eval_results = model.fit(
                        dataset_train=dataset_train,
                        dataset_test=dataset_val
                    )
            else:
                if params["is_epoches"]:
                    eval_results = model.fit_epoches(
                        dataset_train=dataset_train,
                        dataset_test=dataset_test
                    )
                else:
                    eval_results = model.fit(
                        dataset_train=dataset_train,
                        dataset_test=dataset_test
                    )
            logging.info("-" * 40)

            if params["is_validation"]:
                logging.info('\033[91m'+"##############################"+'\033[0m')
                logging.info("TestDataset inference. ")
                eval_results = model.evaluate_roc(dataset_test, exp_num)
                logging.info('\033[91m'+"##############################"+'\033[0m')

            if eval_results["f1"] > best_f1:
                best_f1 = eval_results["f1"]
                model.save_model()

            for k, v in eval_results.items():
                store_dict[k].append(v)


        model.load_model(model_save_path=model_save_path)
        whole_results = defaultdict(float)
        for k, v in store_dict.items():
            mean_score = np.mean(store_dict[k])
            whole_results[k] = mean_score

        result_str = "Final Result: " + \
                     "\t".join(["{}-{:.4f}".format(k, v) for k, v in whole_results.items()])
        logging.info(result_str)
        logging.info("=" * 40)


        # dump_store_results("BGL_Store", store_dict)
        if params["is_validation"]:
            dump_store_results_cm("BGL_VAL_Store_SPCLF", store_dict)
        else:
            dump_store_results_cm("BGL_Store", store_dict)

        dump_final_results(params, whole_results)