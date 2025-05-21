import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from utils.time import *
from utils.env import *
import argparse
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
from utils.data import *
from utils.preprocess import *
from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.utils import negative_sampling

def test(model, dataloader, save_path=None, flag=None):
    loss_func = nn.MSELoss(reduction="mean")
    device = get_device()
    test_loss_list = []
    now = time.time()
    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []
    test_len = len(dataloader)
    model.eval()
    i = 0
    acu_loss = 0
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]
        with torch.no_grad():
            predicted = model(x, edge_index).float().to(device)
            loss = loss_func(predicted, y)
            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])
            if len(t_test_predicted_list) == 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)
        test_loss_list.append(loss.detach().cpu().numpy())
        acu_loss += loss.item()
        i += 1
        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))
    if save_path:
        np.save(f"{save_path}{flag}_predicted.npy", t_test_predicted_list.cpu().numpy())
        np.save(f"{save_path}{flag}_true.npy", t_test_ground_list.cpu().numpy())
    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()
    test_labels_list = t_test_labels_list.tolist()
    avg_loss = sum(test_loss_list) / test_len
    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]
