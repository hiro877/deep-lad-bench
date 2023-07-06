import os
import sys
import time
import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
from itertools import chain

from htm.encoders.rdse import RDSE_Parameters, RDSE
from htm.bindings.algorithms import SpatialPooler, TemporalMemory
from htm.bindings.sdr import SDR



class Encoder:
    def __init__(self, embedding_dim, sparsity):
        p = RDSE_Parameters()
        p.size = embedding_dim
        p.sparsity = sparsity
        p.category = True
        for _ in range(3):
            try:
                self.encoder = RDSE(p)
            except RuntimeError as e:
                logging.error("RDSE init error: retry.")
            else:
                break
        else:
            sys.exit()

    def __call__(self, x):
        return self.encoder.encode(x)



class BaseModel:
    def __init__(
            self,
            meta_data,
            model_save_path,
            feature_type,
            label_type,
            eval_type="session",
            **kwargs,
    ):
        self.meta_data = meta_data
        self.feature_type = feature_type
        self.label_type = label_type
        self.eval_type = eval_type
        self.params = kwargs

        self.time_tracker = {}
        self.sp = None
        self.tm = None
        self.learn = True

        os.makedirs(model_save_path, exist_ok=True)
        self.model_save_file = os.path.join(model_save_path, "model_{}.pickle")
        self.model_save_dir = model_save_path
        if feature_type in ["sequentials", "semantics"]:
            self.encoder = Encoder(
                embedding_dim=kwargs["column_size"],
                sparsity=kwargs.get("enc_sparsity", 0.2)
            )
        else:
            logging.info(f'Unrecognized feature type, except sequentials or semantics, got {feature_type}')

    def train(self):
        self.learn = True

    def eval(self):
        self.learn = False

    def anomaly(self):
        return float(self.tm.anomaly)

    def reset(self):
        self.tm.reset()

    def save_model(self):
        if self.sp:
            model_name = self.model_save_file.format("sp")
            logging.info("Saving model to {}".format(model_name))
            self.sp.saveToFile(model_name)
        if self.tm:
            model_name = self.model_save_file.format("tm")
            logging.info("Saving model to {}".format(model_name))
            self.tm.saveToFile(model_name)

    def load_model(self, model_save_path=""):
        logging.info("Loading model from {}".format(model_save_path))
        if self.params["use_spatial"]:
            model_name = os.path.join(model_save_path, "model_sp.pickle")
            try:
                self.sp.loadFromFile(model_name)
            except Exception as e:
                logging.info(e)
            else:
                logging.info("Loaded model from {}".format(model_name))
                logging.info(str(self.sp))

        if self.params["use_temporal"]:
            model_name = os.path.join(model_save_path, "model_tm.pickle")
            try:
                self.tm.loadFromFile(model_name)
            except Exception as e:
                logging.info(e)
            else:
                logging.info("Loaded model from {}".format(model_name))
                logging.info(str(self.tm))


class HTM(BaseModel):
    def __init__(
            self,
            meta_data,
            model_save_path,
            feature_type,
            label_type,
            eval_type="session",
            **kwargs
    ):
        super().__init__(
            meta_data=meta_data,
            model_save_path=model_save_path,
            feature_type=feature_type,
            label_type=label_type,
            eval_type=eval_type,
            **kwargs
        )
        self.trp_size = (kwargs["pad_size"], kwargs["truncate_size"])
        self.input_shape = (kwargs["column_size"],)
        self.output_shape = kwargs["column_size"] if kwargs["use_spatial"] else (kwargs["column_size"],)

        assert kwargs["use_spatial"] or kwargs["use_temporal"]
        if kwargs["use_spatial"]:
            if kwargs["use_temporal"]:
                self.step = self.step_SPTM
            else:
                self.step = self.step_SP
        else:
            self.step = self.step_TM
        self.build_layer(kwargs)
        self.forward = np.frompyfunc(self._forward, 1, 1)

    def build_layer(self, params):
        if params["use_spatial"]:
            self.sp = SpatialPooler(
                inputDimensions=self.input_shape,
                columnDimensions=self.output_shape,
                potentialPct=params["potential_pct"],
                potentialRadius=params["potential_radius"],
                globalInhibition=True if len(self.output_shape) == 1 else False,
                localAreaDensity=params["local_density"],
                synPermInactiveDec=params["syn_dec"],
                synPermActiveInc=params["syn_inc"],
                synPermConnected=params["syn_connect"],
                boostStrength=params["boost_strength"],
                wrapAround=True,
            )
        if params["use_temporal"]:
            self.tm = TemporalMemory(
                columnDimensions=self.output_shape,
                cellsPerColumn=params["num_cells"],
                activationThreshold=params["act_threshold"],
                initialPermanence=params["init_perm"],
                connectedPermanence=params["syn_connect"],
                minThreshold=params["min_threshold"],
                maxNewSynapseCount=params["new_synapse_count"],
                permanenceIncrement=params["perm_inc"],
                permanenceDecrement=params["perm_dec"],
                predictedSegmentDecrement=0.0,
                maxSegmentsPerCell=params["max_segments"],
                maxSynapsesPerSegment=params["max_synapses"]
            )

    def pooling(self, sdr):
        activeColumns = SDR(self.output_shape)
        self.sp.compute(sdr, self.learn, activeColumns)
        return activeColumns

    def get_prediction(self):
        self.tm.activateDendrites(self.learn)
        predictedColumnIndices = {self.tm.columnForCell(i)
                                  for i in self.tm.getPredictiveCells().sparse}
        predictedColumns = SDR(self.output_shape)
        predictedColumns.sparse = list(predictedColumnIndices)
        return predictedColumns

    def memory(self, sdr):
        self.tm.compute(sdr, self.learn)

    def step_SPTM(self, encoding):
        activeColumns = self.pooling(encoding)
        self.memory(activeColumns)

    def step_SP(self, encoding):
        return self.pooling(encoding)

    def step_TM(self, encoding):
        self.memory(encoding)

    def trp(self, features, n_min, n_max):
        features = features[:n_max-1]  # truncate
        pad_size = max(n_min - 1 - len(features), 0)  # padding
        return np.pad(features, [1, pad_size], 'constant', constant_values=(1, 0))

    def _forward(self, x):
        encoding = self.encoder(x)
        self.step(encoding)
        return self.anomaly()

    def calc_session_anomalies(self, input):
        features = self.trp(input["features"], self.trp_size[0], self.trp_size[1])
        scores = self.forward(features)
        self.reset()
        return np.mean(scores[1:])

    def calc_message_anomalies(self, input):
        features = np.insert(input["features"], 0, 1)
        scores = self.forward(features)
        self.reset()
        return scores[1:].tolist()

    def fit(self, dataset_train, dataset_test=None):
        self.train()
        logging.info("Start training")
        time_start = time.time()

        if self.params["dataset"] == "HDFS":
            y_pred = [self.calc_session_anomalies(input) for input in tqdm(dataset_train)]
        elif self.params["dataset"] == "BGL":
            y_pred = [self.calc_message_anomalies(input) for input in tqdm(dataset_train)]

        time_elapsed = time.time() - time_start
        self.time_tracker["train"] = time_elapsed
        logging.info("End training [{:.2f}s]".format(self.time_tracker["train"]))

        if dataset_test:
            eval_results = self.evaluate(dataset_test)

        return eval_results

    def optimize_threshold(self, y_pred, y_true):
        vals = np.unique(y_pred)
        scores = []
        for val in vals:
            pred = np.where(y_pred > val, 1, 0)
            scores.append((val, f1_score(pred, y_true)))

        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        threshold = sorted_scores[0][0]
        logging.info(f'optimized threshold:{sorted_scores[0][0]},'
                     f' f-score:{sorted_scores[0][1]}')
        return threshold

    def evaluate(self, dataset_test, dtype="test"):
        self.eval()  # set to evaluation mode

        logging.info("Start inference")
        infer_start = time.time()

        if self.params["dataset"] == "HDFS":
            scores = [self.calc_session_anomalies(input) for input in tqdm(dataset_test)]
            y_true = [input["window_labels"] for input in dataset_test]
        elif self.params["dataset"] == "BGL":
            scores = [self.calc_message_anomalies(input) for input in tqdm(dataset_test)]
            y_true = [input["window_labels"] for input in dataset_test]
            scores = list(chain.from_iterable(scores))
            y_true = list(chain.from_iterable(y_true))

        threshold = self.optimize_threshold(scores, y_true)
        y_pred = np.where(scores > threshold, 1, 0)

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