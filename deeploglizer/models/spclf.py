import os
import sys
import time
import logging
import numpy as np
from memory_profiler import profile
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt

from htm.encoders.rdse import RDSE_Parameters, RDSE
from htm.bindings.algorithms import SpatialPooler, Classifier
from htm.bindings.sdr import SDR, Metrics

import csv
from concurrent.futures import ProcessPoolExecutor


class Encoder:
    def __init__(self, params):
        self.enc = SDR((params["emb_dim"], params["window_size"]))
        self.dimensions = (params["emb_dim"], params["window_size"])
        p = RDSE_Parameters()
        p.size = params["emb_dim"]
        p.sparsity = params["emb_sparsity"]
        p.category = True
        for _ in range(3):
            try:
                self.encoder = RDSE(p)
                self.feature_map = {str(i): self.encoder.encode(i).dense[:, np.newaxis]
                                    for i in range(params["n_templates"])}
            except RuntimeError as e:
                logging.error("RDSE init error: retry.")
            else:
                break
        else:
            sys.exit()

    def encode(self, i):
        return self.feature_map.get(
            str(i), self.encoder.encode(i).dense[:, np.newaxis])

    def __call__(self, data):
        """
        encode the data
        @param data - raw data
        @param out  - return SDR with encoded data
        """
        self.enc.dense = np.concatenate(
            [self.encode(data[i]) for i in range(len(data))], axis=1)
        return self.enc



class BaseModel:
    def __init__(
            self,
            meta_data,
            model_save_path,
            feature_type,
            label_type,
            **kwargs,
    ):
        self.meta_data = meta_data
        self.feature_type = feature_type
        self.label_type = label_type
        self.params = kwargs

        self.time_tracker = {}
        self.sp = None
        self.learn = True
        self.sdrc = None

        os.makedirs(model_save_path, exist_ok=True)
        self.model_save_file = os.path.join(model_save_path, "sp.pickle")
        self.model_save_file_sdrc = os.path.join(model_save_path, "sdrc.pickle")
        self.model_save_dir = model_save_path
        if feature_type in ["sequentials", "semantics"]:
            self.encoder = Encoder(kwargs)
        else:
            logging.info(f'Unrecognized feature type, except sequentials or semantics, got {feature_type}')

        # print(self.model_save_file)
        # sys.exit()

    def train(self):
        self.learn = True

    def eval(self):
        self.learn = False

    def save_model(self):
        if self.sp:
            model_name = self.model_save_file
            logging.info("Saving model(sp) to {}".format(model_name))
            self.sp.saveToFile(model_name)
        if self.sdrc:
            model_name = self.model_save_file_sdrc
            logging.info("Saving model(sdrc) to {}".format(model_name))
            self.sdrc.saveToFile(model_name)

    def load_model(self, model_save_path=""):
        # logging.info("Loading model from {}".format(model_save_path))
        model_name = os.path.join(model_save_path, "sp.pickle")
        print("Loading model from {}".format(model_name))
        try:
            self.sp.loadFromFile(model_name)
        except Exception as e:
            # logging.info(e)
            print(e)
        else:
            # logging.info("Loaded model from {}".format(model_name))
            print("Loaded model from {}".format(model_name))
            logging.info(str(self.sp))

        # logging.info("Loading model from {}".format(model_save_path))
        model_name = os.path.join(model_save_path, "sdrc.pickle")
        print("Loading model from {}".format(model_name))
        try:
            self.sdrc.loadFromFile(model_name)
        except Exception as e:
            # logging.info(e)
            print(e)
        else:
            # logging.info("Loaded model from {}".format(model_name))
            print("Loaded model from {}".format(model_name))
            logging.info(str(self.sdrc))



class SPClassifier(BaseModel):
    def __init__(
            self,
            meta_data,
            model_save_path,
            feature_type,
            label_type,
            **kwargs
    ):
        super().__init__(
            meta_data=meta_data,
            model_save_path=model_save_path,
            feature_type=feature_type,
            label_type=label_type,
            **kwargs
        )
        self.build_layer(kwargs)

    @profile(precision=7)
    def build_layer(self, params):
        self.sp = SpatialPooler(
            inputDimensions=self.encoder.dimensions,
            columnDimensions=params['column_dims'],
            potentialRadius=params['potential_radius'],
            potentialPct=params['potential_pct'],
            globalInhibition=True,
            localAreaDensity=params['local_density'],
            stimulusThreshold=int(round(params['stimulus_thresh'])),
            synPermInactiveDec=params['syn_dec'],
            synPermActiveInc=params['syn_inc'],
            synPermConnected=params['syn_connect'],
            minPctOverlapDutyCycle=params['min_overlap_duty'],
            dutyCyclePeriod=int(round(params['duty_cycle'])),
            boostStrength=params['boost_strength'],
            seed=0,  # this is important, 0="random" seed which changes on each invocation
            spVerbosity=99,
            wrapAround=False)
        self.columns = SDR(self.sp.getColumnDimensions())
        self.columns_stats = Metrics(self.columns, 99999999)
        self.sdrc = Classifier()




    @profile(precision=9)
    def mem_prof(self, dataset_train):
        logging.info("Start profiling")

        for d in dataset_train[:3]:
            feat, lbl = d["features"], d["window_anomalies"]
            enc = self.encoder(feat)
            self.sp.compute(enc, self.learn, self.columns)
            self.sdrc.learn(self.columns, lbl)

        logging.info("End profiling")

    def fit(self, dataset_train, dataset_test=None):

        if self.is_learning_curve:
            return self.fit_learning_curve(dataset_train, dataset_test)

        self.train()
        logging.info("Start training")
        time_start = time.time()

        # Training Loop
        for d in dataset_train:
            feat, lbl = d["features"], d["window_anomalies"]
            enc = self.encoder(feat)
            self.sp.compute(enc, self.learn, self.columns)
            self.sdrc.learn(self.columns, lbl)

        logging.info(str(self.sp))
        logging.info(str(self.columns_stats))

        time_elapsed = time.time() - time_start
        self.time_tracker["train"] = time_elapsed
        logging.info("End training [{:.2f}s]".format(self.time_tracker["train"]))

        if dataset_test:
            eval_results = self.evaluate(dataset_test)

        return eval_results

    def fit_learning_curve(self, dataset_train, dataset_test=None):
        self.train()
        logging.info("Start training")
        time_start = time.time()

        # Training Loop
        for d in dataset_train:
            feat, lbl = d["features"], d["window_anomalies"]
            enc = self.encoder(feat)
            self.sp.compute(enc, self.learn, self.columns)
            self.sdrc.learn(self.columns, lbl)

        logging.info(str(self.sp))
        logging.info(str(self.columns_stats))

        time_elapsed = time.time() - time_start
        self.time_tracker["train"] = time_elapsed
        logging.info("End training [{:.2f}s]".format(self.time_tracker["train"]))

        if dataset_test:
            lc_datas_train = []
            lc_datas_test = []
            eval_results = self.evaluate(dataset_test)

            eval_results_train = self.evaluate(dataset_train)
            lc_datas_train.append([-1, eval_results_train["f1"], eval_results_train["rc"],
                                   eval_results_train["pc"], eval_results_train["acc"]])
            lc_datas_test.append([-1, eval_results["f1"], eval_results["rc"],
                                  eval_results["pc"], eval_results["acc"]])

            with open(self.wtp_records_path + '/Learning_Curve.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(["best_rersult", str(eval_results)])
                writer.writerow(["train"])
                writer.writerow(["epoch", "f1", "rc", "pc", "acc"])
                for index in range(len(lc_datas_train)):
                    writer.writerow(lc_datas_train[index])
                writer.writerow(["test"])
                writer.writerow(["epoch", "f1", "rc", "pc", "acc"])
                for index in range(len(lc_datas_test)):
                    writer.writerow(lc_datas_test[index])

        return eval_results

    def fit_epoches(self, dataset_train, dataset_test=None, epoches=10, patience=3):
        if self.is_learning_curve:
            return self.fit_epoches_learning_curve(dataset_train, dataset_test, epoches)
        logging.info("Start training")

        best_f1 = -float("inf")
        best_results = None
        worse_count = 0
        for epoch in range(1, epoches + 1):
            self.train()
            time_start = time.time()
            # Training Loop
            for d in dataset_train:
                feat, lbl = d["features"], d["window_anomalies"]
                enc = self.encoder(feat)
                self.sp.compute(enc, self.learn, self.columns)
                self.sdrc.learn(self.columns, lbl)

            logging.info(str(self.sp))
            logging.info(str(self.columns_stats))

            time_elapsed = time.time() - time_start
            self.time_tracker["train"] = time_elapsed
            logging.info("End training [{:.2f}s]".format(self.time_tracker["train"]))

            if dataset_test is not None and (epoch % 1 == 0):
                eval_results = self.evaluate(dataset_test)
                if eval_results["f1"] > best_f1:
                    best_f1 = eval_results["f1"]
                    best_results = eval_results
                    best_results["converge"] = int(epoch)
                    self.save_model()
                    worse_count = 0
                else:
                    worse_count += 1
                    if worse_count >= patience:
                        logging.info("Early stop at epoch: {}".format(epoch))
                        break
        self.load_model(self.model_save_file)
        return best_results

    def fit_epoches_learning_curve(self, dataset_train, dataset_test=None, epoches=10):
        logging.info("Start training")

        best_f1 = -float("inf")
        best_results = None
        worse_count = 0
        lc_datas_train = []
        lc_datas_test = []
        for epoch in range(1, epoches + 1):
            self.train()
            time_start = time.time()
            # Training Loop
            for d in dataset_train:
                feat, lbl = d["features"], d["window_anomalies"]
                enc = self.encoder(feat)
                self.sp.compute(enc, self.learn, self.columns)
                self.sdrc.learn(self.columns, lbl)

            logging.info(str(self.sp))
            logging.info(str(self.columns_stats))
            time_elapsed = time.time() - time_start
            self.time_tracker["train"] = time_elapsed
            logging.info("End training [{:.2f}s]".format(self.time_tracker["train"]))

            if dataset_test is not None and (epoch % 1 == 0):

                eval_results = self.evaluate(dataset_test)
                if eval_results["f1"] > best_f1:
                    best_f1 = eval_results["f1"]
                    best_results = eval_results
                    best_results["converge"] = int(epoch)
                    self.save_model()
                eval_results_train = self.evaluate(dataset_train)

                lc_datas_train.append([epoch, eval_results_train["f1"], eval_results_train["rc"],
                                       eval_results_train["pc"], eval_results_train["acc"]])
                lc_datas_test.append([epoch, eval_results["f1"], eval_results["rc"],
                                      eval_results["pc"], eval_results["acc"]])



                #     worse_count = 0
                # else:
                #     worse_count += 1
                    # if worse_count >= patience:
                    #     logging.info("Early stop at epoch: {}".format(epoch))
                    #     break

        with open(self.wtp_records_path + '/Learning_Curve.csv', 'a') as f:
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

        self.load_model(self.model_save_file)
        return best_results

    def evaluate(self, dataset_test, dtype="test"):
        self.eval()  # set to evaluation mode

        logging.info("Start inference")
        infer_start = time.time()

        # Testing Loop
        score = 0
        y_true = []
        y_pred = []
        for d in dataset_test:
            feat, lbl = d["features"], d["window_anomalies"]
            enc = self.encoder(feat)
            self.sp.compute(enc, self.learn, self.columns)
            y_true.append(lbl)
            y_pred.append(np.argmax(self.sdrc.infer(self.columns)))

        infer_end = time.time()
        logging.info("Finish inference. [{:.2f}s]".format(infer_end - infer_start))
        self.time_tracker["test"] = infer_end - infer_start

        eval_results = {
            "f1": f1_score(y_true, y_pred),
            "rc": recall_score(y_true, y_pred),
            "pc": precision_score(y_true, y_pred),
            "acc": accuracy_score(y_true, y_pred),
        }
        logging.info({k: f"{v:.3f}" for k, v in eval_results.items()})
        return eval_results

    def evaluate_roc(self, dataset_test, exp_num, dtype="test"):
        self.eval()  # set to evaluation mode

        logging.info("Start inference")
        infer_start = time.time()

        # Testing Loop
        score = 0
        y_true = []
        y_pred = []
        # y_prob = []
        for d in dataset_test:
            feat, lbl = d["features"], d["window_anomalies"]
            enc = self.encoder(feat)
            self.sp.compute(enc, self.learn, self.columns)
            y_true.append(lbl)
            prob = self.sdrc.infer(self.columns)
            # y_prob.append(prob)
            y_pred.append(np.argmax(prob))

        infer_end = time.time()
        logging.info("Finish inference. [{:.2f}s]".format(infer_end - infer_start))
        self.time_tracker["test"] = infer_end - infer_start

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.flatten()

        eval_results = {
            "f1": f1_score(y_true, y_pred),
            "rc": recall_score(y_true, y_pred),
            "pc": precision_score(y_true, y_pred),
            "acc": accuracy_score(y_true, y_pred),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "roc_auc": roc_auc_score(y_true, y_pred)
        }

        # ROC曲線を描写
        os.makedirs("Results_ROC/Temp/", exist_ok=True)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, label='spclf')
        plt.fill_between(fpr, tpr, 0, alpha=0.1)

        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.savefig("Results_ROC/Temp/ROC{}.png".format(exp_num))  # プロットしたグラフをファイルsin.pngに保存する

        logging.info({k: f"{v:.3f}" for k, v in eval_results.items()})

        return eval_results

class SPClassifierProcess(BaseModel):
    def __init__(
            self,
            meta_data,
            model_save_path,
            feature_type,
            label_type,
            **kwargs
    ):
        super().__init__(
            meta_data=meta_data,
            model_save_path=model_save_path,
            feature_type=feature_type,
            label_type=label_type,
            **kwargs
        )
        self.build_layer(kwargs)

    @profile(precision=7)
    def build_layer(self, params):
        self.sp = SpatialPooler(
            inputDimensions=self.encoder.dimensions,
            columnDimensions=params['column_dims'],
            potentialRadius=params['potential_radius'],
            potentialPct=params['potential_pct'],
            globalInhibition=True,
            localAreaDensity=params['local_density'],
            stimulusThreshold=int(round(params['stimulus_thresh'])),
            synPermInactiveDec=params['syn_dec'],
            synPermActiveInc=params['syn_inc'],
            synPermConnected=params['syn_connect'],
            minPctOverlapDutyCycle=params['min_overlap_duty'],
            dutyCyclePeriod=int(round(params['duty_cycle'])),
            boostStrength=params['boost_strength'],
            seed=0,  # this is important, 0="random" seed which changes on each invocation
            spVerbosity=99,
            wrapAround=False)
        self.columns = SDR(self.sp.getColumnDimensions())
        # self.columns_stats = Metrics(self.columns, 99999999)
        self.sdrc = Classifier()
        self.is_learning_curve = False
        self.is_argmax_pred = True




    # @profile(precision=9)
    # def mem_prof(self, dataset_train):
    #     logging.info("Start profiling")
    #
    #     for d in dataset_train[:3]:
    #         feat, lbl = d["features"], d["window_anomalies"]
    #         enc = self.encoder(feat)
    #         self.sp.compute(enc, self.learn, self.columns)
    #         self.sdrc.learn(self.columns, lbl)
    #
    #     logging.info("End profiling")

    def fit(self, dataset_train, dataset_test=None):

        self.train()
        print("Start training. pid={}".format(os.getpid()))
        # time_start = time.time()

        # Training Loop
        for d in dataset_train:
            feat, lbl = d["features"], d["window_anomalies"]
            enc = self.encoder(feat)
            self.sp.compute(enc, self.learn, self.columns)
            self.sdrc.learn(self.columns, lbl)

        # logging.info(str(self.sp))
        # logging.info(str(self.columns_stats))

        # time_elapsed = time.time() - time_start
        # self.time_tracker["train"] = time_elapsed
        print("End training. pid={}".format(os.getpid()))

        if dataset_test:
            # eval_results = self.evaluate(dataset_test)
            eval_results = self.evaluate_process(dataset_test)

        return eval_results

    def fit_learning_curve(self, dataset_train, dataset_test=None):
        self.train()
        logging.info("Start training")
        time_start = time.time()

        # Training Loop
        for d in dataset_train:
            feat, lbl = d["features"], d["window_anomalies"]
            enc = self.encoder(feat)
            self.sp.compute(enc, self.learn, self.columns)
            self.sdrc.learn(self.columns, lbl)

        logging.info(str(self.sp))
        logging.info(str(self.columns_stats))

        time_elapsed = time.time() - time_start
        self.time_tracker["train"] = time_elapsed
        logging.info("End training [{:.2f}s]".format(self.time_tracker["train"]))

        if dataset_test:
            lc_datas_train = []
            lc_datas_test = []
            eval_results = self.evaluate(dataset_test)

            eval_results_train = self.evaluate(dataset_train)
            lc_datas_train.append([-1, eval_results_train["f1"], eval_results_train["rc"],
                                   eval_results_train["pc"], eval_results_train["acc"]])
            lc_datas_test.append([-1, eval_results["f1"], eval_results["rc"],
                                  eval_results["pc"], eval_results["acc"]])

            with open(self.wtp_records_path + '/Learning_Curve.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(["best_rersult", str(eval_results)])
                writer.writerow(["train"])
                writer.writerow(["epoch", "f1", "rc", "pc", "acc"])
                for index in range(len(lc_datas_train)):
                    writer.writerow(lc_datas_train[index])
                writer.writerow(["test"])
                writer.writerow(["epoch", "f1", "rc", "pc", "acc"])
                for index in range(len(lc_datas_test)):
                    writer.writerow(lc_datas_test[index])

        return eval_results

    def fit_epoches(self, dataset_train, dataset_test=None, epoches=10, patience=3):
        if self.is_learning_curve:
            return self.fit_epoches_learning_curve(dataset_train, dataset_test, epoches)
        logging.info("Start training")

        best_f1 = -float("inf")
        best_results = None
        worse_count = 0
        for epoch in range(1, epoches + 1):
            self.train()
            time_start = time.time()
            # Training Loop
            for d in dataset_train:
                feat, lbl = d["features"], d["window_anomalies"]
                enc = self.encoder(feat)
                self.sp.compute(enc, self.learn, self.columns)
                self.sdrc.learn(self.columns, lbl)

            logging.info(str(self.sp))
            logging.info(str(self.columns_stats))

            time_elapsed = time.time() - time_start
            self.time_tracker["train"] = time_elapsed
            logging.info("End training [{:.2f}s]".format(self.time_tracker["train"]))

            if dataset_test is not None and (epoch % 1 == 0):
                eval_results = self.evaluate(dataset_test)
                if eval_results["f1"] > best_f1:
                    best_f1 = eval_results["f1"]
                    best_results = eval_results
                    best_results["converge"] = int(epoch)
                    self.save_model()
                    worse_count = 0
                else:
                    worse_count += 1
                    if worse_count >= patience:
                        logging.info("Early stop at epoch: {}".format(epoch))
                        break
        self.load_model(self.model_save_file)
        return best_results

    def fit_epoches_learning_curve(self, dataset_train, dataset_test=None, epoches=10):
        logging.info("Start training")

        best_f1 = -float("inf")
        best_results = None
        worse_count = 0
        lc_datas_train = []
        lc_datas_test = []
        for epoch in range(1, epoches + 1):
            self.train()
            time_start = time.time()
            # Training Loop
            for d in dataset_train:
                feat, lbl = d["features"], d["window_anomalies"]
                enc = self.encoder(feat)
                self.sp.compute(enc, self.learn, self.columns)
                self.sdrc.learn(self.columns, lbl)

            logging.info(str(self.sp))
            logging.info(str(self.columns_stats))
            time_elapsed = time.time() - time_start
            self.time_tracker["train"] = time_elapsed
            logging.info("End training [{:.2f}s]".format(self.time_tracker["train"]))

            if dataset_test is not None and (epoch % 1 == 0):

                eval_results = self.evaluate(dataset_test)
                if eval_results["f1"] > best_f1:
                    best_f1 = eval_results["f1"]
                    best_results = eval_results
                    best_results["converge"] = int(epoch)
                    self.save_model()
                eval_results_train = self.evaluate(dataset_train)

                lc_datas_train.append([epoch, eval_results_train["f1"], eval_results_train["rc"],
                                       eval_results_train["pc"], eval_results_train["acc"]])
                lc_datas_test.append([epoch, eval_results["f1"], eval_results["rc"],
                                      eval_results["pc"], eval_results["acc"]])



                #     worse_count = 0
                # else:
                #     worse_count += 1
                    # if worse_count >= patience:
                    #     logging.info("Early stop at epoch: {}".format(epoch))
                    #     break

        with open(self.wtp_records_path + '/Learning_Curve.csv', 'a') as f:
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

        self.load_model(self.model_save_file)
        return best_results

    def evaluate(self, dataset_test, dtype="test"):
        self.eval()  # set to evaluation mode

        logging.info("Start inference")
        infer_start = time.time()

        # Testing Loop
        score = 0
        y_true = []
        y_pred = []
        for d in dataset_test:
            feat, lbl = d["features"], d["window_anomalies"]
            enc = self.encoder(feat)
            self.sp.compute(enc, self.learn, self.columns)
            y_true.append(lbl)
            y_pred.append(np.argmax(self.sdrc.infer(self.columns)))

        infer_end = time.time()
        logging.info("Finish inference. [{:.2f}s]".format(infer_end - infer_start))
        self.time_tracker["test"] = infer_end - infer_start

        eval_results = {
            "f1": f1_score(y_true, y_pred),
            "rc": recall_score(y_true, y_pred),
            "pc": precision_score(y_true, y_pred),
            "acc": accuracy_score(y_true, y_pred),
        }
        logging.info({k: f"{v:.3f}" for k, v in eval_results.items()})
        return eval_results

    def evaluate_process(self, dataset_test, dtype="test"):
        self.eval()  # set to evaluation mode

        logging.info("evaluate_process(): Start inference")
        # infer_start = time.time()

        # Testing Loop
        score = 0
        # y_true = []
        y_pred = []
        for d in dataset_test:
            feat, lbl = d["features"], d["window_anomalies"]
            enc = self.encoder(feat)
            self.sp.compute(enc, self.learn, self.columns)
            # y_true.append(lbl)
            if self.is_argmax_pred:
                y_pred.append(np.argmax(self.sdrc.infer(self.columns)))
            else:
                pred = self.sdrc.infer(self.columns)
                pred_argmax = np.argmax[pred]
                y_pred.append(pred[pred_argmax] * pred_argmax)
            # sys.exit()

        # infer_end = time.time()
        # logging.info("Finish inference. [{:.2f}s]".format(infer_end - infer_start))
        # self.time_tracker["test"] = infer_end - infer_start

        eval_results = {
            # "y_true": y_true,
            "y_pred": y_pred,
        }
        # logging.info({k: f"{v:.3f}" for k, v in eval_results.items()})
        return eval_results

# class SPClassifierMultiProcess(BaseModel):
#     def __init__(
#             self,
#             meta_data,
#             model_save_path,
#             feature_type,
#             label_type,
#             **kwargs
#     ):
#         super().__init__(
#             meta_data=meta_data,
#             model_save_path=model_save_path,
#             feature_type=feature_type,
#             label_type=label_type,
#             **kwargs
#         )
#         self.build_layer(kwargs, model_save_path)
#
#     @profile(precision=7)
#     def build_layer(self, params, model_save_path, process_num=3):
#         self.sp = SpatialPooler(
#             inputDimensions=self.encoder.dimensions,
#             columnDimensions=params['column_dims'],
#             potentialRadius=params['potential_radius'],
#             potentialPct=params['potential_pct'],
#             globalInhibition=True,
#             localAreaDensity=params['local_density'],
#             stimulusThreshold=int(round(params['stimulus_thresh'])),
#             synPermInactiveDec=params['syn_dec'],
#             synPermActiveInc=params['syn_inc'],
#             synPermConnected=params['syn_connect'],
#             minPctOverlapDutyCycle=params['min_overlap_duty'],
#             dutyCyclePeriod=int(round(params['duty_cycle'])),
#             boostStrength=params['boost_strength'],
#             seed=0,  # this is important, 0="random" seed which changes on each invocation
#             spVerbosity=99,
#             wrapAround=False)
#         self.columns = SDR(self.sp.getColumnDimensions())
#         self.columns_stats = Metrics(self.columns, 99999999)
#         self.sdrc = Classifier()
#
#         self.spatialpoolers = []
#
#         for i in range(process_num):
#             self.spatialpoolers.append(model = SPClassifier(meta_data=ext.meta_data, model_save_path=model_save_path, **params))
#
#     @profile(precision=9)
#     def mem_prof(self, dataset_train):
#         logging.info("Start profiling")
#
#         for d in dataset_train[:3]:
#             feat, lbl = d["features"], d["window_anomalies"]
#             enc = self.encoder(feat)
#             self.sp.compute(enc, self.learn, self.columns)
#             self.sdrc.learn(self.columns, lbl)
#
#         logging.info("End profiling")
#
#     def set_dataset(self, dataset_train, dataset_test, dataset_val=None):
#         self.dataset_train = dataset_train
#         self.dataset_test = dataset_test
#         self.dataset_val = dataset_val
#     def fit(self, dataset_train, dataset_test=None):
#         with ProcessPoolExecutor(max_workers=3) as executor:
#             # 『関数』と『引数リスト』を渡して、実行します。
#             results = executor.map(
#                 fit_wrapper, src_datas, timeout=None)
#         if self.is_learning_curve:
#             return self.fit_learning_curve(dataset_train, dataset_test)
#
#         self.train()
#         logging.info("Start training")
#         time_start = time.time()
#
#         # Training Loop
#         for d in tqdm(dataset_train):
#             feat, lbl = d["features"], d["window_anomalies"]
#             enc = self.encoder(feat)
#             self.sp.compute(enc, self.learn, self.columns)
#             self.sdrc.learn(self.columns, lbl)
#
#         logging.info(str(self.sp))
#         logging.info(str(self.columns_stats))
#
#         time_elapsed = time.time() - time_start
#         self.time_tracker["train"] = time_elapsed
#         logging.info("End training [{:.2f}s]".format(self.time_tracker["train"]))
#
#         if dataset_test:
#             eval_results = self.evaluate(dataset_test)
#
#         return eval_results
#
#     def fit_wrapper(self, i):
#         return self.spatialpoolers[i].fit(self.dataset_train, self.dataset_test)
#
#     def fit_learning_curve(self, dataset_train, dataset_test=None):
#         self.train()
#         logging.info("Start training")
#         time_start = time.time()
#
#         # Training Loop
#         for d in tqdm(dataset_train):
#             feat, lbl = d["features"], d["window_anomalies"]
#             enc = self.encoder(feat)
#             self.sp.compute(enc, self.learn, self.columns)
#             self.sdrc.learn(self.columns, lbl)
#
#         logging.info(str(self.sp))
#         logging.info(str(self.columns_stats))
#
#         time_elapsed = time.time() - time_start
#         self.time_tracker["train"] = time_elapsed
#         logging.info("End training [{:.2f}s]".format(self.time_tracker["train"]))
#
#         if dataset_test:
#             lc_datas_train = []
#             lc_datas_test = []
#             eval_results = self.evaluate(dataset_test)
#
#             eval_results_train = self.evaluate(dataset_train)
#             lc_datas_train.append([-1, eval_results_train["f1"], eval_results_train["rc"],
#                                    eval_results_train["pc"], eval_results_train["acc"]])
#             lc_datas_test.append([-1, eval_results["f1"], eval_results["rc"],
#                                   eval_results["pc"], eval_results["acc"]])
#
#             with open(self.wtp_records_path + '/Learning_Curve.csv', 'a') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(["best_rersult", str(eval_results)])
#                 writer.writerow(["train"])
#                 writer.writerow(["epoch", "f1", "rc", "pc", "acc"])
#                 for index in range(len(lc_datas_train)):
#                     writer.writerow(lc_datas_train[index])
#                 writer.writerow(["test"])
#                 writer.writerow(["epoch", "f1", "rc", "pc", "acc"])
#                 for index in range(len(lc_datas_test)):
#                     writer.writerow(lc_datas_test[index])
#
#         return eval_results
#
#     def fit_epoches(self, dataset_train, dataset_test=None, epoches=10, patience=3):
#         if self.is_learning_curve:
#             return self.fit_epoches_learning_curve(dataset_train, dataset_test, epoches)
#         logging.info("Start training")
#
#         best_f1 = -float("inf")
#         best_results = None
#         worse_count = 0
#         for epoch in range(1, epoches + 1):
#             self.train()
#             time_start = time.time()
#             # Training Loop
#             for d in tqdm(dataset_train):
#                 feat, lbl = d["features"], d["window_anomalies"]
#                 enc = self.encoder(feat)
#                 self.sp.compute(enc, self.learn, self.columns)
#                 self.sdrc.learn(self.columns, lbl)
#
#             logging.info(str(self.sp))
#             logging.info(str(self.columns_stats))
#
#             time_elapsed = time.time() - time_start
#             self.time_tracker["train"] = time_elapsed
#             logging.info("End training [{:.2f}s]".format(self.time_tracker["train"]))
#
#             if dataset_test is not None and (epoch % 1 == 0):
#                 eval_results = self.evaluate(dataset_test)
#                 if eval_results["f1"] > best_f1:
#                     best_f1 = eval_results["f1"]
#                     best_results = eval_results
#                     best_results["converge"] = int(epoch)
#                     self.save_model()
#                     worse_count = 0
#                 else:
#                     worse_count += 1
#                     if worse_count >= patience:
#                         logging.info("Early stop at epoch: {}".format(epoch))
#                         break
#         self.load_model(self.model_save_file)
#         return best_results
#
#     def fit_epoches_learning_curve(self, dataset_train, dataset_test=None, epoches=10):
#         logging.info("Start training")
#
#         best_f1 = -float("inf")
#         best_results = None
#         worse_count = 0
#         lc_datas_train = []
#         lc_datas_test = []
#         for epoch in range(1, epoches + 1):
#             self.train()
#             time_start = time.time()
#             # Training Loop
#             for d in tqdm(dataset_train):
#                 feat, lbl = d["features"], d["window_anomalies"]
#                 enc = self.encoder(feat)
#                 self.sp.compute(enc, self.learn, self.columns)
#                 self.sdrc.learn(self.columns, lbl)
#
#             logging.info(str(self.sp))
#             logging.info(str(self.columns_stats))
#             time_elapsed = time.time() - time_start
#             self.time_tracker["train"] = time_elapsed
#             logging.info("End training [{:.2f}s]".format(self.time_tracker["train"]))
#
#             if dataset_test is not None and (epoch % 1 == 0):
#
#                 eval_results = self.evaluate(dataset_test)
#                 if eval_results["f1"] > best_f1:
#                     best_f1 = eval_results["f1"]
#                     best_results = eval_results
#                     best_results["converge"] = int(epoch)
#                     self.save_model()
#                 eval_results_train = self.evaluate(dataset_train)
#
#                 lc_datas_train.append([epoch, eval_results_train["f1"], eval_results_train["rc"],
#                                        eval_results_train["pc"], eval_results_train["acc"]])
#                 lc_datas_test.append([epoch, eval_results["f1"], eval_results["rc"],
#                                       eval_results["pc"], eval_results["acc"]])
#
#
#
#                 #     worse_count = 0
#                 # else:
#                 #     worse_count += 1
#                     # if worse_count >= patience:
#                     #     logging.info("Early stop at epoch: {}".format(epoch))
#                     #     break
#
#         with open(self.wtp_records_path + '/Learning_Curve.csv', 'a') as f:
#             writer = csv.writer(f)
#             writer.writerow(["best_rersult", str(best_results), "converge", str(best_results["converge"])])
#             writer.writerow(["train"])
#             writer.writerow(["epoch", "f1", "rc", "pc", "acc"])
#             for index in range(len(lc_datas_train)):
#                 writer.writerow(lc_datas_train[index])
#             writer.writerow(["test"])
#             writer.writerow(["epoch", "f1", "rc", "pc", "acc"])
#             for index in range(len(lc_datas_test)):
#                 writer.writerow(lc_datas_test[index])
#
#         self.load_model(self.model_save_file)
#         return best_results
#
#     def evaluate(self, dataset_test, dtype="test"):
#         self.eval()  # set to evaluation mode
#
#         logging.info("Start inference")
#         infer_start = time.time()
#
#         # Testing Loop
#         score = 0
#         y_true = []
#         y_pred = []
#         for d in tqdm(dataset_test):
#             feat, lbl = d["features"], d["window_anomalies"]
#             enc = self.encoder(feat)
#             self.sp.compute(enc, self.learn, self.columns)
#             y_true.append(lbl)
#             y_pred.append(np.argmax(self.sdrc.infer(self.columns)))
#
#         infer_end = time.time()
#         logging.info("Finish inference. [{:.2f}s]".format(infer_end - infer_start))
#         self.time_tracker["test"] = infer_end - infer_start
#
#         eval_results = {
#             "f1": f1_score(y_true, y_pred),
#             "rc": recall_score(y_true, y_pred),
#             "pc": precision_score(y_true, y_pred),
#             "acc": accuracy_score(y_true, y_pred),
#         }
#         logging.info({k: f"{v:.3f}" for k, v in eval_results.items()})
#         return eval_results