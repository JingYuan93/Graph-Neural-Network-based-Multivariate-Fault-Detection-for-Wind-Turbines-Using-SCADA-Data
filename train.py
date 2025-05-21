from test import *
import torch.nn.functional as F


EPS = 1e-15

def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction="mean")
    return loss

def adjust_learning_rate(optimizer, epoch, init_lr):
    lr = init_lr * (0.9 ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Updating learning rate to {}'.format(lr))

def train(
    model: torch.nn.Module = None,
    save_path="",
    config={},
    train_dataloader=None,
    val_dataloader=None,
    feature_map=[],
    test_dataloader=None,
    test_dataset=None,
    dataset_name="swat",
    train_dataset=None,
):
    seed = config["seed"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["decay"]
    )
    now = time.time()
    train_loss_list = []
    cmp_loss_list = []
    device = get_device()
    acu_loss = 0
    min_loss = 1e8
    min_f1 = 0
    min_pre = 0
    best_prec = 0
    i = 0
    epoch = config["epoch"]
    early_stop_patience = 3
    model.train()
    log_interval = 1000
    stop_improve_count = 0
    dataloader = train_dataloader
    for i_epoch in range(epoch):
        acu_loss = 0
        model.train()
        for x, y, attack_labels, edge_index in dataloader:
            _start = time.time()
            x, y, edge_index = [
                item.float().to(device) for item in [x, y, edge_index]
            ]
            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)
            loss = loss_func(out, y)
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())
            acu_loss += loss.item()
            i += 1
        if val_dataloader is not None:
            val_loss, val_result = test(model, val_dataloader, flag="val")
            print(
                "epoch ({}/{}) (Loss:{:.8f}, Val_loss:{:.8f}, ACU_loss:{:.8f})".format(
                    i_epoch + 1, epoch, acu_loss / len(dataloader), val_loss, acu_loss
                ),
                flush=True,
            )
            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1
                print(f'EarlyStopping counter: {stop_improve_count} out of {early_stop_patience}')
            if stop_improve_count >= early_stop_patience:
                print("Early stopped!")
                print(f"Best mse: {min_loss}")
                break
        else:
            if acu_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss
    return train_loss_list
