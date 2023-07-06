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
from deeploglizer.common.utils import seed_everything, dump_final_results, dump_params, dump_store_results



parser = argparse.ArgumentParser()

##### Model params
parser.add_argument("--model_name", default="SPClassifier", type=str)

parser.add_argument("--emb_dim", default=500, type=int)
parser.add_argument("--emb_sparsity", default=0.15, type=float)

parser.add_argument("--potential_radius", default=7, type=int)
parser.add_argument("--duty_cycle", default=1402, type=int)
parser.add_argument("--stimulus_thresh", default=6, type=int)
parser.add_argument("--potential_pct", default=0.1, type=float)
parser.add_argument("--local_density", default=0.1, type=float)
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

##### Others
parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument("--n_experiment", default=5, type=int)
parser.add_argument("--epoches", default=100, type=int)
parser.add_argument("--patience", default=3, type=int)
parser.add_argument("--is_validation", action='store_true')

params = vars(parser.parse_args())



if __name__ == '__main__':
    is_shuffle = False
    use_all_template_in_learning = False
    print("is_shuffle: ", is_shuffle)
    print("use_all_template_in_learning: ", use_all_template_in_learning)
    seed_everything(params["random_seed"])

    session_train, session_test, _ = load_sessions(data_dir=params["data_dir"])

    ext = FeatureExtractor(**params)

    session_train = ext.fit_transform(session_train)
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

    model_save_path = dump_params(params)
    dataset_train = log_dataset(
        session_train, feature_type=params["feature_type"], data_pct=1.0, shuffle=is_shuffle)
    model = SPClassifier(meta_data=ext.meta_data, model_save_path=model_save_path, **params)
    model.mem_prof(dataset_train=dataset_train)

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
            print(len(dataset_train))
            dataset_val = dataset_train[-int(len(dataset_train)*validation_ratio):]
            dataset_train = dataset_train[:-int(len(dataset_train)*validation_ratio)]
            print("len(dataset_val): ", len(dataset_val))
            print("len(dataset_train): ", len(dataset_train))
            print("len(dataset_test): ", len(dataset_test))

        store_dict = defaultdict(list)
        best_f1 = -float("inf")
        for i in range(params["n_experiment"]):
            logging.info("{}Experiment #{}{}".format("-" * 15, i, "-" * 15))
            model = SPClassifier(meta_data=ext.meta_data, model_save_path=model_save_path, **params)

            # eval_results = model.fit(
            #     dataset_train=dataset_train,
            #     dataset_test=dataset_test
            # )
            if params["is_validation"]:
                eval_results = model.fit(
                    dataset_train=dataset_train,
                    dataset_test=dataset_val
                )
            else:
                eval_results = model.fit(
                    dataset_train=dataset_train,
                    dataset_test=dataset_test
                )
            logging.info("-" * 40)

            if params["is_validation"]:
                logging.info("##############################")
                logging.info("TestDataset inference. ")
                eval_results = model.evaluate(dataset_test)
                logging.info("##############################")

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

        dump_final_results(params, whole_results, model)
        # dump_store_results("BGL_Store", store_dict)
        if params["is_validation"]:
            dump_store_results("BGL_VAL_Store", store_dict)
        else:
            dump_store_results("BGL_Store", store_dict)