import os
import h5py
import numpy as np
import torch
import medmnist
from medmnist import INFO
from medmnist.dataset import MedMNIST3D


# ============================================================
# 1) Load selected Hodge decomposition components from .h5
# ============================================================
def load_decomposition_img_3d(
    dataname,
    split,
    index,
    label,
    components=("cur", "div", "har"),
    root_dir=".",
):
    """
    Load selected 3D Hodge decomposition components.

    For each selected component:
        cur/div/har has 3 directional channels in 3D.

    Therefore:
        ("cur",)              -> 3 channels
        ("div",)              -> 3 channels
        ("har",)              -> 3 channels
        ("cur", "div")        -> 6 channels
        ("cur", "div", "har") -> 9 channels
    """

    if isinstance(components, str):
        raise TypeError(
            f"components must be a tuple/list like ('div',), not a string: {components}"
        )

    # MedMNIST3D split sizes
    train_test_num = {
        "organmnist3d": [971, 610],
        "nodulemnist3d": [1158, 310],
        "adrenalmnist3d": [1188, 298],
        "fracturemnist3d": [1027, 137],
        "vesselmnist3d": [1335, 382],
        "synapsemnist3d": [1230, 352],
    }

    train_num, test_num = train_test_num[dataname]

    if split == "train":
        now_index = index
    elif split == "test":
        now_index = index + train_num
    elif split == "val":
        now_index = index + train_num + test_num
    else:
        raise ValueError(f"Unknown split: {split}")

    filename = os.path.join(root_dir, dataname, f"{now_index}.h5")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing decomposition file: {filename}")

    img_high = []

    with h5py.File(filename, "r") as file:
        label_h5 = file["label"][:]

        # safer label check
        label_h5_int = int(np.array(label_h5).reshape(-1)[0])
        label_int = int(np.array(label).reshape(-1)[0])
        assert label_h5_int == label_int, (
            f"Label mismatch at {filename}: h5={label_h5_int}, dataset={label_int}"
        )

        # 3D MedMNIST images are grayscale, so k = 0
        k = 0

        # this must match decomposition.py
        v = 1.1

        for comp in components:
            name = f"{now_index}-{v}-{k}-{comp}"

            if name not in file:
                raise KeyError(
                    f"Missing key '{name}' in {filename}. "
                    f"Available examples: {list(file.keys())[:10]}"
                )

            arr = np.round(file[name][:], 8)

            # Expected arr shape from 3D decomposition:
            # usually (3, D, H, W), one channel per x/y/z direction
            arr = np.asarray(arr)

            # If decomposition has padding/cropping need, handle common cases
            # Desired output spatial size is 64 x 64 x 64
            if arr.ndim != 4:
                raise ValueError(
                    f"Expected 4D array for {name}, got shape {arr.shape}"
                )

            # If it is larger than 64 due to padding, crop central region
            # Usually arr should already be close to 64^3.
            C, D, H, W = arr.shape

            target_size = 64
            if D >= target_size and H >= target_size and W >= target_size:
                d0 = (D - target_size) // 2
                h0 = (H - target_size) // 2
                w0 = (W - target_size) // 2
                arr = arr[
                    :,
                    d0:d0 + target_size,
                    h0:h0 + target_size,
                    w0:w0 + target_size,
                ]

            img_high.append(arr)

    img_high = np.concatenate(img_high, axis=0) * 2.0

    return torch.tensor(img_high, dtype=torch.get_default_dtype())


# ============================================================
# 2) Dataset wrapper
# ============================================================
class My_MedMNIST3D(MedMNIST3D):
    def __init__(
        self,
        *args,
        components=("cur", "div", "har"),
        decomp_root=".",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if isinstance(components, str):
            raise TypeError(
                f"components must be a tuple/list like ('div',), not a string: {components}"
            )

        self.components = components
        self.decomp_root = decomp_root

    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index].astype(int)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img2 = load_decomposition_img_3d(
            dataname=self.flag,
            split=self.split,
            index=index,
            label=self.labels[index],
            components=self.components,
            root_dir=self.decomp_root,
        )

        return img2, target


# ============================================================
# 3) Dataset classes
# ============================================================
class OrganMNIST3D(My_MedMNIST3D):
    flag = "organmnist3d"


class NoduleMNIST3D(My_MedMNIST3D):
    flag = "nodulemnist3d"


class AdrenalMNIST3D(My_MedMNIST3D):
    flag = "adrenalmnist3d"


class FractureMNIST3D(My_MedMNIST3D):
    flag = "fracturemnist3d"


class VesselMNIST3D(My_MedMNIST3D):
    flag = "vesselmnist3d"


class SynapseMNIST3D(My_MedMNIST3D):
    flag = "synapsemnist3d"


# ============================================================
# 4) Dataset selector
# ============================================================
def get_3d_data_class(typ):
    if typ == "organmnist3d":
        return OrganMNIST3D
    elif typ == "nodulemnist3d":
        return NoduleMNIST3D
    elif typ == "adrenalmnist3d":
        return AdrenalMNIST3D
    elif typ == "fracturemnist3d":
        return FractureMNIST3D
    elif typ == "vesselmnist3d":
        return VesselMNIST3D
    elif typ == "synapsemnist3d":
        return SynapseMNIST3D
    else:
        raise ValueError(f"Unknown 3D dataset: {typ}")


# ============================================================
# 5) Main loader used by train3D_ablation.py
# ============================================================
def get_train_val_test_data_3d(config):
    DataClass = get_3d_data_class(config["dataset"])

    components = config.get("components", ("cur", "div", "har"))

    if isinstance(components, str):
        raise TypeError(
            f"config['components'] must be a tuple/list like ('div',), not a string: {components}"
        )

    decomp_root = config.get("decomp_root", ".")

    train_dataset = DataClass(
        split="train",
        download=True,
        size=config["img_size"],
        components=components,
        decomp_root=decomp_root,
    )

    val_dataset = DataClass(
        split="val",
        download=True,
        size=config["img_size"],
        components=components,
        decomp_root=decomp_root,
    )

    test_dataset = DataClass(
        split="test",
        download=True,
        size=config["img_size"],
        components=components,
        decomp_root=decomp_root,
    )

    return train_dataset, val_dataset, test_dataset