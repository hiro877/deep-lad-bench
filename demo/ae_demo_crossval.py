#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

sys.path.append("../")
import argparse
import logging
from torch.utils.data import DataLoader
from collections import defaultdict

from deeploglizer.models import AutoEncoder
from deeploglizer.common.preprocess import FeatureExtractor
from deeploglizer.common.dataloader import load_sessions, log_dataset, log_dataset_gen
from deeploglizer.common.utils import seed_everything, dump_params, dump_final_results, dump_store_results_next_log


parser = argparse.ArgumentParser()

##### Model params
parser.add_argument("--model_name", default="Autoencoder", type=str)
parser.add_argument("--hidden_size", default=128, type=int)
parser.add_argument("--num_directions", default=2, type=int)
parser.add_argument("--num_layers", default=2, type=int)
parser.add_argument("--embedding_dim", default=32, type=int)

##### Dataset params
parser.add_argument("--dataset", default="HDFS", type=str)
parser.add_argument(
    "--data_dir", default="../data/processed/HDFS_100k/hdfs_0.0_tar", type=str
)
parser.add_argument("--window_size", default=10, type=int)
parser.add_argument("--stride", default=1, type=int)
parser.add_argument("--data_pct", default=1.0, type=float)


##### Input params
parser.add_argument("--feature_type", default="sequentials", type=str, choices=["sequentials", "semantics"])
parser.add_argument("--use_tfidf", action="store_true")
parser.add_argument("--max_token_len", default=50, type=int)
parser.add_argument("--min_token_count", default=1, type=int)
parser.add_argument("--eval_type", default="window", type=str)
# Uncomment the following to use pretrained word embeddings. The "embedding_dim" should be set as 300
# parser.add_argument(
#     "--pretrain_path", default="../data/pretrain/wiki-news-300d-1M.vec", type=str
# )

##### Training params
parser.add_argument("--epoches", default=100, type=int)
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--test_batch_size", default=4096, type=int)
parser.add_argument("--learning_rate", default=0.01, type=float)
parser.add_argument("--anomaly_ratio", default=0.1, type=float)
parser.add_argument("--patience", default=3, type=int)

##### Others
parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--cache", default=False, type=bool)
parser.add_argument("--n_experiment", default=5, type=int)
parser.add_argument("--is_validation", action='store_true')
parser.add_argument("--use_val_template_in_learning", action='store_true')
parser.add_argument("--wtp_records_path", default="wtp_records", type=str)

params = vars(parser.parse_args())



if __name__ == "__main__":
    is_shuffle = True
    use_all_template_in_learning = False
    print("is_shuffle: ", is_shuffle)
    print("use_all_template_in_learning: ", use_all_template_in_learning)
    seed_everything(params["random_seed"])

    os.makedirs(params["wtp_records_path"], exist_ok=True)

    session_train, session_test, session_val = load_sessions(data_dir=params["data_dir"], is_validation=params["is_validation"])
    ext = FeatureExtractor(**params)
    ext.is_print_dataset = True

    session_train = ext.fit_transform(session_train)
    dataset_train = log_dataset(
        session_train, feature_type=params["feature_type"],
        data_pct=1, shuffle=is_shuffle)
    dataloader_train = DataLoader(
        dataset_train, batch_size=params["batch_size"], shuffle=is_shuffle, pin_memory=True)

    if params["is_validation"]:
        if params["use_val_template_in_learning"]:
            ext.update_id2log_train(session_val)
        session_val = ext.transform(session_val, datatype="test")
        dataset_val = log_dataset(
            session_val, feature_type=params["feature_type"])
        dataloader_val = DataLoader(
            dataset_val, batch_size=params["test_batch_size"], shuffle=is_shuffle, pin_memory=True)
    if use_all_template_in_learning:
        ext.update_id2log_train(session_test)
    session_test = ext.transform(session_test, datatype="test")


    session_test = ext.transform(session_test, datatype="test")

    dataset_test = log_dataset(
        session_test, feature_type=params["feature_type"])

    dataloader_test = DataLoader(
        dataset_test, batch_size=params["test_batch_size"], shuffle=is_shuffle, pin_memory=True)


    # for pct in [0.1 * (i + 1) for i in range(10)]:
    for pct in [1.0]:
        params["data_pct"] = pct
        model_save_path = dump_params(params)

        logging.info("{}Experiment data_pct:{}%{}".format("=" * 15, pct, "=" * 15))
        # dataset_train = log_dataset(
        #     session_train, feature_type=params["feature_type"], data_pct=pct, shuffle=True)
        # dataloader_train = DataLoader(
        #     dataset_train, batch_size=params["batch_size"], shuffle=True, pin_memory=True)

        if params["is_validation"]:
            print("len(dataset_val): ", len(dataset_val))
            print("len(dataset_train): ", len(dataset_train))
            print("len(dataset_test): ", len(dataset_test))
        else:
            print("len(dataset_train): ", len(dataset_train))
            print("len(dataset_test): ", len(dataset_test))


        store_dict = defaultdict(list)
        best_f1 = -float("inf")
        for i in range(params["n_experiment"]):
            logging.info("{}Experiment #{}{}".format("-" * 15, i, "-" * 15))

            model = AutoEncoder(
                meta_data=ext.meta_data, model_save_path=model_save_path, **params
            )
            model.is_learning_curve = True
            model.wtp_records_path=params["wtp_records_path"]
            #
            # eval_results = model.fit(
            #     dataloader_train,
            #     test_loader=dataloader_test,
            #     epoches=params["epoches"],
            #     learning_rate=params["learning_rate"],
            # )
            if params["is_validation"]:
                eval_results = model.fit(
                    dataloader_train,
                    test_loader=dataloader_val,
                    epoches=params["epoches"],
                    learning_rate=params["learning_rate"],
                )
            else:
                eval_results = model.fit(
                    dataloader_train,
                    test_loader=dataloader_test,
                    epoches=params["epoches"],
                    learning_rate=params["learning_rate"],
                )

            logging.info("-" * 40)

            if params["is_validation"]:
                logging.info("##############################")
                logging.info("TestLoader inference. ")
                model.is_exp_analisys=True
                eval_results = model.evaluate(dataloader_test)
                model.is_exp_analisys = False
                logging.info("##############################")
                # with open(params["wtp_records_path"]+'/miss_pred_exp{}.csv'.format(i), 'w') as f:
                #     writer = csv.writer(f)
                #     for k, v in model.exp_analisys_dict["miss_pred"].items():
                #         writer.writerow([k, v, wtp_window_labels[v], wtp_window_labels[v]])
                # with open(params["wtp_records_path"]+'/miss_pred_wtp_exp{}.csv'.format(i), 'w') as f:
                #     writer = csv.writer(f)
                #     for k, v in model.exp_analisys_dict["miss_pred_wtp"].items():
                #         writer.writerow([k, v, wtp_window_labels[k], wtp_window_labels[k]])

            if eval_results["f1"] > best_f1:
                best_f1 = eval_results["f1"]
                model.save_model()

            for k, v in eval_results.items():
                store_dict[k].append(v)

        whole_results = defaultdict(float)
        for k, v in store_dict.items():
            mean_score = np.mean(store_dict[k])
            whole_results[k] = mean_score

        result_str = "Final Result: " + \
                     "\t".join(["{}-{:.4f}".format(k, v) for k, v in whole_results.items()])
        logging.info(result_str)
        logging.info("=" * 40)

        if params["is_validation"]:
            dump_store_results_next_log("BGL_VAL_Store", store_dict)
        else:
            dump_store_results_next_log("BGL_Store", store_dict)