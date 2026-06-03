import argparse
import torch
from medmnist.evaluator import getACC, getAUC
from medmnist import INFO
from train_3d import train_model_3d


def layers_by_size_3d(dataname: str, data_num: dict) -> int:
    n = data_num[dataname]
    return 5 if n < 3000 else 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="random seed")
    parser.add_argument(
        "--dataset",
        type=str,
        default="fracturemnist3d",
        help=(
            "dataset name (e.g. organmnist3d, nodulemnist3d, "
            "adrenalmnist3d, fracturemnist3d, vesselmnist3d, synapsemnist3d)"
        ),
    )
    args = parser.parse_args()

    seed = args.seed
    dataname = args.dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Seed:", seed)
    print("Dataset:", dataname)

    data_3d = [
        "organmnist3d",
        "nodulemnist3d",
        "adrenalmnist3d",
        "fracturemnist3d",
        "vesselmnist3d",
        "synapsemnist3d",
    ]

    data_num = {
        "organmnist3d": 1742,
        "nodulemnist3d": 1633,
        "adrenalmnist3d": 1584,
        "fracturemnist3d": 1370,
        "vesselmnist3d": 1908,
        "synapsemnist3d": 1759,
    }

    assert dataname in data_3d, f"{dataname} not in supported dataset list"
    assert dataname in data_num, f"{dataname} missing from data_num"

    epoch = 10
    img_size = 64
    batch = 8
    lr = 1e-3
    h_channel = 64
    head = 4

    layer_num = layers_by_size_3d(dataname, data_num)
    task_ref = INFO[dataname]["task"]

    print(f"\nDataset={dataname}  n={data_num[dataname]}  layers={layer_num}")

    y_true_np, y_pred_np = train_model_3d(
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
        raise RuntimeError("Got no predictions back from train_model_3d")

    acc = getACC(y_true_np, y_pred_np, task_ref)
    auc = getAUC(y_true_np, y_pred_np, task_ref)

    print(f"\nFinal Results")
    print(f"ACC = {acc*100:.2f}")
    print(f"AUC = {auc*100:.2f}")


if __name__ == "__main__":
    main()
