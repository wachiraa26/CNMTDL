import argparse
import numpy as np
import os
from medmnist.evaluator import getACC, getAUC
from medmnist import INFO
from train import train_model_2d
import torch


def get_run_id(cli_run_id: str | None) -> str:
    env_run_id = (
        os.environ.get("SLURM_ARRAY_JOB_ID")
        or os.environ.get("SLURM_JOB_ID")
        or "nojid"
    )
    return cli_run_id or env_run_id


def layers_by_size(dataname: str, data_num: dict) -> int:
    n = data_num[dataname]
    return 5 if n < 150_000 else 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="single seed for this run")
    parser.add_argument("--run_id", type=str, default=None, help="optional run id override (default: SLURM_ARRAY_JOB_ID)")
    args = parser.parse_args()

    seed = args.seed
    run_id = get_run_id(args.run_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(">>> seed:", seed)
    print(">>> run_id:", run_id)
    print(">>> SLURM_ARRAY_JOB_ID:", os.environ.get("SLURM_ARRAY_JOB_ID"))
    print(">>> SLURM_JOB_ID:", os.environ.get("SLURM_JOB_ID"))

    data_2d = ["retinamnist", "bloodmnist","organamnist", "organcmnist", "organsmnist"]

    datasets =  data_2d[0]

    data_num = { "retinamnist": 1600, "bloodmnist": 17092,"organamnist": 58830, "organcmnist": 23583, "organsmnist": 25211}

    epoch = 30
    img_size = 224
    batch = 64
    lr = 5e-3
    h_channel = 64
    head = 4

    pred_root = "./predictions"
    run_dir = os.path.join(pred_root, f"run{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    for dataname in datasets:
        assert dataname in data_2d, f"{dataname} not in data_2d list"
        assert dataname in data_num, f"{dataname} missing from data_num"

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

        print(f"\nSeed {seed}  ACC={acc*100:.2f}  AUC={auc*100:.2f}")

        save_path = os.path.join(
            run_dir,
            f"{dataname}_seed{seed}_lr{lr}_ep{epoch}_L{layer_num}.npz"
        )

        np.savez(
            save_path,
            y_true=y_true_np,
            y_pred=y_pred_np,
            task=task_ref,
            dataname=dataname,
            seed=seed,
            lr=lr,
            epoch=epoch,
            layer_num=layer_num,
            run_id=str(run_id),
        )
        print("saved predictions:", save_path)


if __name__ == "__main__":
    main()


