import argparse
import torch
from medmnist.evaluator import getACC, getAUC
from medmnist import INFO
from train import train_model_2d


def layers_by_size(dataname: str, data_num: dict) -> int:
    n = data_num[dataname]
    return 5 if n < 150_000 else 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="random seed")
    parser.add_argument(
        "--dataset",
        type=str,
        default="retinamnist",
        help="dataset name (e.g. retinamnist, bloodmnist, organamnist)",
    )
    args = parser.parse_args()

    seed = args.seed
    dataname = args.dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Seed:", seed)
    print("Dataset:", dataname)

    data_2d = [ "retinamnist", "dermamnist","bloodmnist", "organamnist","organcmnist", "organsmnist"]

    data_num = { "retinamnist": 1600,  "dermamnist": 10015, "bloodmnist": 17092,  "organamnist": 58830,   "organcmnist": 23583,"organsmnist": 25211}

    assert dataname in data_2d, f"{dataname} not in supported dataset list"
    assert dataname in data_num, f"{dataname} missing from data_num"

    epoch = 30
    img_size = 224
    batch = 64
    lr = 5e-3
    h_channel = 64
    head = 4

    layer_num = layers_by_size(dataname, data_num)
    task_ref = INFO[dataname]["task"]

    print(f"\nDataset={dataname}  n={data_num[dataname]}  layers={layer_num}")

    y_true_np, y_pred_np = train_model_2d(
        dataname=dataname,
        img_size=img_size,
        batch=batch,
        lr=lr,
        epoch=epoch,
        h_channel=h_channel,
        head=head,
        layer_num=layer_num,
        seed=seed,
        device=device,
    )

    if y_true_np is None:
        raise RuntimeError("Got no predictions back from train_model_2d")

    acc = getACC(y_true_np, y_pred_np, task_ref)
    auc = getAUC(y_true_np, y_pred_np, task_ref)

    print(f"\nFinal Results")
    print(f"ACC = {acc*100:.2f}")
    print(f"AUC = {auc*100:.2f}")


if __name__ == "__main__":
    main()
