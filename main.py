import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import MinMaxScaler
from utils.env import get_device, set_device
from utils.preprocess import build_loc_net, construct_data
from utils.net_struct import get_feature_map, get_graph_struc_from_txt
from sklearn.metrics import precision_score, recall_score, f1_score
from dataset.data_loader import TimeDataset
from models.GRN import GRN
from train import train
from test import test
from evaluate import (
    get_err_scores,
    get_best_performance_data,
    get_val_performance_data,
    get_full_err_scores,
)

from datetime import datetime
import os
import argparse
from pathlib import Path
import random

class Main:
    def __init__(self, train_config, env_config, debug=False):
        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config["dataset"]
        train_orig = pd.read_csv(f"./data/{dataset}/train.csv", sep=",", index_col=0, low_memory=False)
        test_orig = pd.read_csv(f"./data/{dataset}/test.csv", sep=",", index_col=0, low_memory=False)

        numeric_train = train_orig.select_dtypes(include=['number'])
        numeric_test = test_orig.select_dtypes(include=['number'])
        self.scaler = MinMaxScaler()

        train_scaled_array = self.scaler.fit_transform(numeric_train)
        train_scaled = pd.DataFrame(train_scaled_array, columns=numeric_train.columns, index=numeric_train.index)

        test_scaled_array = self.scaler.transform(numeric_test)
        test_scaled = pd.DataFrame(test_scaled_array, columns=numeric_test.columns, index=numeric_test.index)

        self.train_data = train_scaled
        self.test_data = test_scaled

        train, test = train_scaled, test_scaled

        if "attack" in train.columns:
            train = train.drop(columns=["attack"])

        feature_map = get_feature_map(dataset)
        fc_struc = get_graph_struc_from_txt(dataset)
        set_device(env_config["device"])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())

        cfg = {
            "slide_win": train_config["slide_win"],
            "slide_stride": train_config["slide_stride"],
            "loss_type": train_config["loss_type"]
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode="train", config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode="test", config=cfg)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        train_dataloader, val_dataloader = self.get_loaders(
            train_dataset,
            train_config["seed"],
            train_config["batch"],
            val_ratio=train_config["val_ratio"],
        )
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config["batch"], shuffle=False, num_workers=10)

        edge_index_sets = [fc_edge_index]

        self.model = GRN(
            edge_index_sets,
            len(feature_map),
            train_config,
            dim=train_config["dim"],
            input_dim=train_config["slide_win"],
            out_layer_num=train_config["out_layer_num"],
            out_layer_inter_dim=train_config["out_layer_inter_dim"],
            topk=train_config["topk"],
        ).to(self.device)

    def run(self):
        if len(self.env_config["load_model_path"]) > 3:
            model_save_path = self.env_config["load_model_path"]
            path = self.get_save_path()
            result_save_path = path[1]
            print(f"模型保存路径：{model_save_path}")
            print(f"结果保存路径：{result_save_path}")
        else:
            path = self.get_save_path()
            model_save_path = path[0]
            result_save_path = path[1]
            self.result_save_path = result_save_path
            print(f"模型保存路径：{model_save_path}")
            print(f"结果保存路径：{result_save_path}")

            self.train_log = train(
                self.model,
                model_save_path,
                config=train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                dataset_name=self.env_config["dataset"],
            )

        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        _, self.test_result = test(best_model, self.test_dataloader, result_save_path, flag="test")
        _, self.val_result = test(best_model, self.val_dataloader, result_save_path, flag="val")

    def get_score2(self, test_result, val_result):
        np_test_result = np.array(test_result)
        gt_labels = np_test_result[2, :, 0].tolist()

        normal_scores = np.abs(val_result[0] - val_result[1]).max(-1)
        test_scores = np.abs(test_result[0] - test_result[1]).max(-1)
        threshold = np.max(normal_scores)

        pred_labels = np.zeros(len(test_scores))
        pred_labels[test_scores > threshold] = 1

        pred_labels = pred_labels.astype(int).tolist()
        gt_labels = [int(x) for x in gt_labels]

        pre = precision_score(gt_labels, pred_labels)
        rec = recall_score(gt_labels, pred_labels)
        f1 = f1_score(gt_labels, pred_labels)

        print(f"F1 score: {f1}")
        print(f"precision: {pre}")
        print(f"recall: {rec}")
        print(f"threshold: {threshold}\n")

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.01):
        dataset_len = len(train_dataset)
        if val_ratio == "te":
            train_use_len = 492
        else:
            val_ratio = float(val_ratio)
            train_use_len = int(dataset_len * (1 - val_ratio))
            val_use_len = int(dataset_len * val_ratio)

        indices = torch.arange(dataset_len)

        train_sub_indices = indices[:train_use_len]
        train_subset = Subset(train_dataset, train_sub_indices)

        if val_ratio == "te":
            val_sub_indices = indices[500:]
        else:
            val_sub_indices = indices[train_use_len:]
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True, num_workers=10)
        val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False, num_workers=10)

        print(f"train set: {len(train_subset)}")
        print(f"val set: {len(val_subset)}")

        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result):
        np_test_result = np.array(test_result)
        test_labels = np_test_result[2, :, 0].tolist()

        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        if self.env_config["report"] == "best":
            info = get_best_performance_data(test_scores, test_labels, topk=1, save_path=self.result_save_path)
        else:
            info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1, save_path=self.result_save_path)

        print(f"F1 score: {info[0]}")
        print(f"precision: {info[1]}")
        print(f"recall: {info[2]}")
        print(f"threshold: {info[-1]}\n")

    def get_save_path(self, feature_name=""):
        dir_path = self.env_config["save_path"]
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime("%m-%d-%H:%M:%S")
        datestr = self.datestr

        paths = [
            f"./pretrained/{dir_path}/best_{datestr}.pt",
            f"./results/{dir_path}/{datestr}/",
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch", type=int, default=128)
    parser.add_argument("-epoch", type=int, default=100)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-slide_win", type=int, default=15)
    parser.add_argument("-slide_stride", type=int, default=5)
    parser.add_argument("-dim", type=int, default=64)
    parser.add_argument("-save_path_pattern", type=str, default="")
    parser.add_argument("-dataset", type=str, default="openSCADA")
    parser.add_argument("-device", type=str, default="cuda")
    parser.add_argument("-random_seed", type=int, default=2024)
    parser.add_argument("-comment", type=str, default="")
    parser.add_argument("-out_layer_num", type=int, default=1)
    parser.add_argument("-out_layer_inter_dim", type=int, default=256)
    parser.add_argument("-decay", type=float, default=0)
    parser.add_argument("-val_ratio", default=0.1)
    parser.add_argument("-topk", type=int, default=20)
    parser.add_argument("-report", type=str, default="best")
    parser.add_argument("-loss_type", type=str, default="pred")
    parser.add_argument("-load_model_path", type=str, default="")
    parser.add_argument("-exp", type=str, default="")
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)

    train_config = {
        "batch": args.batch,
        "epoch": args.epoch,
        "lr": args.lr,
        "slide_win": args.slide_win,
        "dim": args.dim,
        "slide_stride": args.slide_stride,
        "comment": args.comment,
        "seed": args.random_seed,
        "out_layer_num": args.out_layer_num,
        "out_layer_inter_dim": args.out_layer_inter_dim,
        "decay": args.decay,
        "val_ratio": args.val_ratio,
        "topk": args.topk,
        "loss_type": args.loss_type,
    }

    env_config = {
        "save_path": args.save_path_pattern,
        "dataset": args.dataset,
        "report": args.report,
        "device": args.device,
        "load_model_path": args.load_model_path,
        "exp": args.exp,
    }

    main = Main(train_config, env_config, debug=False)
    main.run()
