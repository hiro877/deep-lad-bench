import os
import sys
import time
import torch
import logging
import numpy as np
import pandas as pd
from torch import nn
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

from deeploglizer.common.utils import set_device, tensor2flatten_arr, check_memory

import csv


class Embedder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        pretrain_matrix=None,
        freeze=False,
        use_tfidf=False,
    ):
        super(Embedder, self).__init__()
        self.use_tfidf = use_tfidf
        if pretrain_matrix is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(
                pretrain_matrix, padding_idx=1, freeze=freeze
            )
        else:
            self.embedding_layer = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=1
            )

    def forward(self, x):
        if self.use_tfidf:
            return torch.matmul(x, self.embedding_layer.weight.double())
        else:
            return self.embedding_layer(x.long())


class ForcastBasedModel(nn.Module):
    def __init__(
        self,
        meta_data,
        model_save_path,
        feature_type,
        label_type,
        eval_type,
        topk,
        use_tfidf,
        embedding_dim,
        freeze=False,
        gpu=-1,
        anomaly_ratio=None,
        patience=3,
        **kwargs,
    ):
        super(ForcastBasedModel, self).__init__()
        self.device = set_device(gpu)
        self.topk = topk
        self.meta_data = meta_data
        self.feature_type = feature_type
        self.label_type = label_type
        self.eval_type = eval_type
        self.anomaly_ratio = anomaly_ratio  # only used for auto encoder
        self.patience = patience
        self.time_tracker = {}
        self.exp_analisys_dict={"result":[], "miss_pred":{}, "miss_pred_wtp": {}}
        self.is_exp_analisys=False
        self.is_learning_curve=False
        self.wtp_records_path="wtp_records"

        os.makedirs(model_save_path, exist_ok=True)
        self.model_save_file = os.path.join(model_save_path, "model.ckpt")
        if feature_type in ["sequentials", "semantics"]:
            self.embedder = Embedder(
                meta_data["vocab_size"],
                embedding_dim=embedding_dim,
                pretrain_matrix=meta_data.get("pretrain_matrix", None),
                freeze=freeze,
                use_tfidf=use_tfidf,
            )
        else:
            logging.info(f'Unrecognized feature type, except sequentials or semantics, got {feature_type}')

    def evaluate(self, test_loader, dtype="test"):
        logging.info("Evaluating {} data.".format(dtype))

        if self.label_type == "next_log":
            return self.__evaluate_next_log(test_loader, dtype=dtype)
        elif self.label_type == "anomaly":
            return self.__evaluate_anomaly(test_loader, dtype=dtype)
        elif self.label_type == "none":
            return self.__evaluate_recst(test_loader, dtype=dtype)

    def __evaluate_recst(self, test_loader, dtype="test"):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()
            for batch_input in test_loader:
                return_dict = self.forward(self.__input2device(batch_input))
                y_pred = return_dict["y_pred"]
                store_dict["session_idx"].extend(
                    tensor2flatten_arr(batch_input["session_idx"])
                )
                store_dict["window_anomalies"].extend(
                    tensor2flatten_arr(batch_input["window_anomalies"])
                )
                store_dict["window_preds"].extend(tensor2flatten_arr(y_pred))
            infer_end = time.time()
            logging.info("Finish inference [{:.2f}s]".format(infer_end - infer_start))
            self.time_tracker["test"] = infer_end - infer_start

            store_df = pd.DataFrame(store_dict)

            if self.eval_type == "session":
                use_cols = ["session_idx", "window_anomalies", "window_preds"]
                session_df = (
                    store_df[use_cols]
                    .groupby("session_idx", as_index=False)
                    .max()  # most anomalous window
                )
                assert (
                    self.anomaly_ratio is not None
                ), "anomaly_ratio should be specified for autoencoder!"
                print("self.eval_type == session:")
            else:
                session_df = store_df
                print("self.eval_type != session:")
            thre = np.percentile(
                session_df[f"window_preds"].values, 100 - self.anomaly_ratio * 100
            )
            pred = (session_df[f"window_preds"] > thre).astype(int)
            y = (session_df["window_anomalies"] > 0).astype(int)

            cm = confusion_matrix(y, pred)
            tn, fp, fn, tp = cm.flatten()
            eval_results = {
                "f1": f1_score(y, pred),
                "rc": recall_score(y, pred),
                "pc": precision_score(y, pred),
                "acc": accuracy_score(y, pred),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
            logging.info({k: f"{v:.3f}" for k, v in eval_results.items()})
            return eval_results

    def __evaluate_anomaly(self, test_loader, dtype="test"):

        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()
            for batch_input in test_loader:
                return_dict = self.forward(self.__input2device(batch_input))
                y_prob, y_pred = return_dict["y_pred"].max(dim=1)
                store_dict["session_idx"].extend(
                    tensor2flatten_arr(batch_input["session_idx"])
                )
                store_dict["window_anomalies"].extend(
                    tensor2flatten_arr(batch_input["window_anomalies"])
                )
                store_dict["window_preds"].extend(tensor2flatten_arr(y_pred))
                # print(batch_input["features"])
                # print(y_pred)
                # print(batch_input["window_anomalies"])
                # print(len(batch_input["features"]), len(y_pred), len(batch_input["window_anomalies"]))
                if self.is_exp_analisys:
                    for i in range(len(y_pred)):
                        if y_pred[i] == batch_input["window_anomalies"][i]:
                            self.exp_analisys_dict["result"].append(0) # 0: True
                        else:
                            self.exp_analisys_dict["result"].append(1) # 1: False
                            window_template_pattern = " ".join(map(str, batch_input["features"][i].tolist()))
                            self.exp_analisys_dict["miss_pred"][str(len(self.exp_analisys_dict["result"]))]=window_template_pattern
                            if self.exp_analisys_dict["miss_pred_wtp"].get(window_template_pattern):
                                self.exp_analisys_dict["miss_pred_wtp"][window_template_pattern] += 1
                            else:
                                self.exp_analisys_dict["miss_pred_wtp"][window_template_pattern] = 1

                    # print(self.exp_analisys_dict)
                    # sys.exit()
            infer_end = time.time()
            logging.info("Finish inference. [{:.2f}s]".format(infer_end - infer_start))
            self.time_tracker["test"] = infer_end - infer_start

            store_df = pd.DataFrame(store_dict)
            use_cols = ["session_idx", "window_anomalies", "window_preds"]
            if self.eval_type == "session":
                session_df = store_df[use_cols].groupby("session_idx", as_index=False).sum()
                pred = (session_df[f"window_preds"] > 0).astype(int)
                y = (session_df["window_anomalies"] > 0).astype(int)
            else:
                print("eval_type: {}".format(self.eval_type))
                pred = (store_df[f"window_preds"] > 0).astype(int)
                y = (store_df["window_anomalies"] > 0).astype(int)
                # print(f1_score(y, pred))
                # print(pred.compare(y))
                # sys.exit()

            cm = confusion_matrix(y, pred)
            tn, fp, fn, tp = cm.flatten()
            eval_results = {
                "f1": f1_score(y, pred),
                "rc": recall_score(y, pred),
                "pc": precision_score(y, pred),
                "acc": accuracy_score(y, pred),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
            logging.info({k: f"{v:.3f}" for k, v in eval_results.items()})
            return eval_results

    def __evaluate_next_log(self, test_loader, dtype="test"):
        model = self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()
            for batch_input in test_loader:
                return_dict = model.forward(self.__input2device(batch_input))
                y_pred = return_dict["y_pred"]
                y_prob_topk, y_pred_topk = torch.topk(y_pred, self.topk)  # b x topk
                # print("===== evaluate =====")
                # print(y_pred)
                # print("==================")
                # print(y_prob_topk)
                # print("==================")
                # print(y_pred_topk)

                store_dict["session_idx"].extend(
                    tensor2flatten_arr(batch_input["session_idx"])
                )
                store_dict["window_anomalies"].extend(
                    tensor2flatten_arr(batch_input["window_anomalies"])
                )
                store_dict["window_labels"].extend(
                    tensor2flatten_arr(batch_input["window_labels"])
                )
                store_dict["x"].extend(batch_input["features"].data.cpu().numpy())
                store_dict["y_pred_topk"].extend(y_pred_topk.data.cpu().numpy())
                store_dict["y_prob_topk"].extend(y_prob_topk.data.cpu().numpy())
                # print(store_dict["window_anomalies"])
                # print(y_pred.shape)
                # print(y_prob_topk.shape)
                # print(y_pred_topk.shape)
                # print(len(store_dict["window_labels"]))
                # sys.exit()
            infer_end = time.time()
            logging.info("Finish inference. [{:.2f}s]".format(infer_end - infer_start))
            self.time_tracker["test"] = infer_end - infer_start
            store_df = pd.DataFrame(store_dict)
            best_result = None
            best_f1 = -float("inf")

            count_start = time.time()

            topkdf = pd.DataFrame(store_df["y_pred_topk"].tolist())
            logging.info("Calculating acc sum.")
            hit_df = pd.DataFrame()
            # print(topkdf.shape)
            # i=0
            for col in sorted(topkdf.columns):
                topk = col + 1
                hit = (topkdf[col] == store_df["window_labels"]).astype(int)
                # print(topkdf[col])
                # print(store_df["window_labels"])
                # print(hit)
                # sys.exit()
                hit_df[topk] = hit
                if col == 0:
                    acc_sum = 2 ** topk * hit
                    # print("topk: {}, hit: {}".format(topk, hit))
                    # print("2 ** topk * hit: ", 2 ** topk * hit)
                else:
                    acc_sum += 2 ** topk * hit
                # i+=1
            # print(i)
            # sys.exit()
            # print("acc_sum: ", acc_sum)
            # print("=========================")
            acc_sum[acc_sum == 0] = 2 ** (1 + len(topkdf.columns))
            # print("acc_sum: ", acc_sum)
            hit_df["acc_num"] = acc_sum
            # sys.exit()

            for col in sorted(topkdf.columns):
                topk = col + 1
                check_num = 2 ** topk
                store_df["window_pred_anomaly_{}".format(topk)] = (
                    ~(hit_df["acc_num"] <= check_num)
                ).astype(int)
                # print('hit_df["acc_num"] <= check_num): ', hit_df["acc_num"] <= check_num)
                # print('~(hit_df["acc_num"] <= check_num: ', ~(hit_df["acc_num"] <= check_num))
                # print(store_df["window_pred_anomaly_{}".format(topk)])
                # sys.exit()
            # store_df.to_csv("store_{}_2.csv".format(dtype), index=False)

            logging.info("Finish generating store_df.")

            if self.eval_type == "session":
                use_cols = ["session_idx", "window_anomalies"] + [
                    f"window_pred_anomaly_{topk}" for topk in range(1, self.topk + 1)
                ]
                session_df = (
                    store_df[use_cols].groupby("session_idx", as_index=False).sum()
                )
                print("self.eval_type == session:")
            else:
                session_df = store_df
                print("self.eval_type != session:")
            session_df.to_csv("session_{}_2.csv".format(dtype), index=False)

            for topk in range(1, self.topk + 1):
                pred = (session_df[f"window_pred_anomaly_{topk}"] > 0).astype(int)
                y = (session_df["window_anomalies"] > 0).astype(int)
                # print(session_df[f"window_pred_anomaly_{topk}"])
                # print(session_df["window_anomalies"])
                # print("=======================")
                # print(pred)
                # print(y)
                # sys.exit()
                window_topk_acc = 1 - store_df["window_anomalies"].sum() / len(store_df)

                cm = confusion_matrix(y, pred)
                tn, fp, fn, tp = cm.flatten()
                eval_results = {
                    "f1": f1_score(y, pred),
                    "rc": recall_score(y, pred),
                    "pc": precision_score(y, pred),
                    "top{}-acc".format(topk): window_topk_acc,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                }
                logging.info({k: f"{v:.3f}" for k, v in eval_results.items()})
                if eval_results["f1"] >= best_f1:
                    best_result = eval_results
                    best_f1 = eval_results["f1"]
            count_end = time.time()
            logging.info("Finish counting [{:.2f}s]".format(count_end - count_start))
            return best_result

    def __input2device(self, batch_input):
        return {k: v.to(self.device) for k, v in batch_input.items()}

    def save_model(self):
        logging.info("Saving model to {}".format(self.model_save_file))
        try:
            torch.save(
                self.state_dict(),
                self.model_save_file,
                _use_new_zipfile_serialization=False,
            )
        except:
            torch.save(self.state_dict(), self.model_save_file)

    def load_model(self, model_save_file=""):
        logging.info("Loading model from {}".format(self.model_save_file))
        self.load_state_dict(torch.load(model_save_file, map_location=self.device))

    def fit(self, train_loader, test_loader=None, epoches=10, learning_rate=1.0e-3):
        if self.is_learning_curve:
            return self.fit_learning_curve(train_loader, test_loader, epoches, learning_rate)

        self.to(self.device)
        logging.info(
            "Start training on {} batches with {}.".format(
                len(train_loader), self.device
            )
        )
        best_f1 = -float("inf")
        best_results = None
        worse_count = 0
        for epoch in range(1, epoches + 1):
            epoch_time_start = time.time()
            model = self.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            batch_cnt = 0
            epoch_loss = 0
            for batch_input in train_loader:
                loss = model.forward(self.__input2device(batch_input))["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                batch_cnt += 1
                # check_memory(batch_cnt)

            epoch_loss = epoch_loss / batch_cnt
            epoch_time_elapsed = time.time() - epoch_time_start
            logging.info(
                "Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, epoches, epoch_loss, epoch_time_elapsed)
            )
            self.time_tracker["train"] = epoch_time_elapsed

            if test_loader is not None and (epoch % 1 == 0):
                eval_results = self.evaluate(test_loader)
                if eval_results["f1"] > best_f1:
                    best_f1 = eval_results["f1"]
                    best_results = eval_results
                    best_results["converge"] = int(epoch)
                    self.save_model()
                    worse_count = 0
                else:
                    worse_count += 1
                    if worse_count >= self.patience:
                        logging.info("Early stop at epoch: {}".format(epoch))
                        break

        self.load_model(self.model_save_file)
        return best_results

    def fit_learning_curve(self, train_loader, test_loader=None, epoches=10, learning_rate=1.0e-3):
        """
        Learning Curve
        ・交差検証なし
        ・交差検証あり（Validation）
        ・交差検証あり（Test）
        Parameters
        ----------
        train_loader
        test_loader
        epoches
        learning_rate

        Returns
        -------

        """
        lc_datas_train=[]
        lc_datas_test=[]

        self.to(self.device)

        best_f1 = -float("inf")
        best_results = None
        for epoch in range(1, epoches + 1):
            epoch_time_start = time.time()
            model = self.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            batch_cnt = 0
            epoch_loss = 0
            for batch_input in train_loader:
                loss = model.forward(self.__input2device(batch_input))["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                batch_cnt += 1

            epoch_loss = epoch_loss / batch_cnt
            epoch_time_elapsed = time.time() - epoch_time_start
            logging.info(
                "Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, epoches, epoch_loss, epoch_time_elapsed)
            )
            self.time_tracker["train"] = epoch_time_elapsed

            if test_loader is not None and (epoch % 1 == 0):
                eval_results = self.evaluate(test_loader)
                print("Epoch Test_Loader")
                print(eval_results)
                if eval_results["f1"] > best_f1:
                    best_f1 = eval_results["f1"]
                    best_results = eval_results
                    best_results["converge"] = int(epoch)
                    self.save_model()
                print("Evaluate Train_Loader for Learning Curve")
                eval_results_train = self.evaluate(train_loader)
                print(eval_results_train)
                if self.label_type == "next_log":
                    # lc_datas_train.append([epoch, eval_results_train["f1"], eval_results_train["rc"],
                    #                        eval_results_train["pc"], eval_results_train["top1-acc"]])
                    # lc_datas_test.append([epoch, eval_results["f1"], eval_results["rc"],
                    #                        eval_results["pc"], eval_results["top1-acc"]])
                    datas=[epoch]
                    for v in eval_results_train.values():
                        datas.append(v)
                    lc_datas_train.append(datas)
                    datas=[epoch]
                    for v in eval_results.values():
                        datas.append(v)
                    lc_datas_test.append(datas)
                elif self.label_type == "anomaly":
                    lc_datas_train.append([epoch, eval_results_train["f1"], eval_results_train["rc"],
                                           eval_results_train["pc"], eval_results_train["acc"]])
                    lc_datas_test.append([epoch, eval_results["f1"], eval_results["rc"],
                                           eval_results["pc"], eval_results["acc"]])
                elif self.label_type == "none":
                    pass
                else:
                    pass

        self.load_model(self.model_save_file)

        with open(self.wtp_records_path+'/Learning_Curve.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(["best_rersult", str(best_results), "converge", str(best_results["converge"])])
            writer.writerow(["train"])
            writer.writerow(["epoch", "f1", "rc", "pc", "acc"])
            for index in range(len(lc_datas_train)):
                writer.writerow(lc_datas_train[index])
            writer.writerow(["test"])
            writer.writerow(["epoch", "f1", "rc", "pc", "acc"])
            for index in range(len(lc_datas_test)):
                writer.writerow(lc_datas_test[index])

        # self.load_model(self.model_save_file)
        return best_results