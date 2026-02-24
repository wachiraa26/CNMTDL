import os
import time
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import medmnist
from medmnist import INFO
from medmnist.evaluator import getACC, getAUC

from model import CCNN
from data_2d_chest import get_train_val_test_data_2d

torch.set_default_dtype(torch.float32)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def predict_probs(model, loader, task, device, config):
    if config["task"] == "multi-label, binary-class":
        prediction = nn.Sigmoid()
    else:
        prediction = nn.Softmax(dim=1)

    y_true, y_prob = torch.tensor([]).to(device), torch.tensor([]).to(device)
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)

            if config["task"] == "multi-label, binary-class":
                target = target.to(torch.float32).to(device)
            else:
                target = torch.squeeze(target, 1).long().to(device)

            logits = model(data)
            probs = prediction(logits)

            y_true = torch.cat((y_true, deepcopy(target)), 0)
            y_prob = torch.cat((y_prob, deepcopy(probs)), 0)

    return y_true.cpu().numpy(), y_prob.cpu().numpy()


def train_and_test_model( model,   train_loader,  val_loader,  test_loader,  criterion,  optimizer,  scheduler,  epoch,  task,  device,  config):
    print("start training")

    if config["task"] == "multi-label, binary-class":
        prediction = nn.Sigmoid()
    else:
        prediction = nn.Softmax(dim=1)

    for e in range(epoch):
        model.train()
        train_loss = 0.0
        train_true, train_pred = torch.tensor([]).to(device), torch.tensor([]).to(device)

        for data, target in train_loader:
            data = data.to(device)
            if config["task"] == "multi-label, binary-class":
                target = target.to(torch.float32).to(device)
            else:
                target = torch.squeeze(target, 1).long().to(device)

            y_pred = model(data)
            optimizer.zero_grad()
            loss = criterion(y_pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            outputs = prediction(y_pred)
            train_true = torch.cat((train_true, deepcopy(target)), 0)
            train_pred = torch.cat((train_pred, deepcopy(outputs.detach())), 0)

        train_ACC = getACC(train_true.cpu().numpy(), train_pred.cpu().numpy(), task)
        train_AUC = getAUC(train_true.cpu().numpy(), train_pred.cpu().numpy(), task)

        # -------- val --------
        model.eval()
        val_true, val_pred = torch.tensor([]).to(device), torch.tensor([]).to(device)
        val_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                if config["task"] == "multi-label, binary-class":
                    target = target.to(torch.float32).to(device)
                else:
                    target = torch.squeeze(target, 1).long().to(device)

                y_pred = model(data)
                loss_now = criterion(y_pred, target)
                val_loss += loss_now.item()

                outputs = prediction(y_pred)
                val_true = torch.cat((val_true, deepcopy(target)), 0)
                val_pred = torch.cat((val_pred, deepcopy(outputs)), 0)

        val_ACC = getACC(val_true.cpu().numpy(), val_pred.cpu().numpy(), task)
        val_AUC = getAUC(val_true.cpu().numpy(), val_pred.cpu().numpy(), task)

        print(
            "epoch:", e + 1,
            "train loss:", round(train_loss, 2),
            "acc:", round(train_ACC * 100, 2),
            "auc:", round(train_AUC * 100, 2),
            "val acc:", round(val_ACC * 100, 2),
            "auc:", round(val_AUC * 100, 2),
        )

        if e == epoch - 1:
            test_true_np, test_pred_np = predict_probs(model, test_loader, task, device, config)
            test_ACC = getACC(test_true_np, test_pred_np, task)
            test_AUC = getAUC(test_true_np, test_pred_np, task)
            print("test", "acc:", round(test_ACC * 100, 2), "auc:", round(test_AUC * 100, 2))
            return test_true_np, test_pred_np

    return None, None


def train_model_2d(dataname,img_size, batch,lr,epoch,h_channel,head,layer_num,seed,device,num_workers=4):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    info = INFO[dataname]
    config = {
        "dataset": dataname,
        "img_size": img_size,
        "batch_size": batch,
        "learning_rate": lr,
        "epochs": epoch,
        "h_channel": h_channel,
        "head": head,
        "layer_num": layer_num,
        "task": info["task"],
        "class_num": len(info["label"]),
        "in_channel": info["n_channels"] * 2 * 3,
        "seed": seed,
        "group": info["n_channels"],
    }

    train_dataset, val_dataset, test_dataset = get_train_val_test_data_2d(config)

    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=pin_memory,
    )

    model = CCNN(
        in_channel=config["in_channel"],
        h_dim=config["h_channel"],
        head=config["head"],
        class_num=config["class_num"],
        layer=config["layer_num"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"],
        pct_start=0.3,
    )

    # loss
    criterion = nn.CrossEntropyLoss()
    if config["task"] == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()

    print(config)
    print("train:", len(train_dataset), "test:", len(test_dataset), "val:", len(val_dataset))

    test_true_np, test_pred_np = train_and_test_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        task=config["task"],
        device=device,
        config=config,
    )
    return test_true_np, test_pred_np
