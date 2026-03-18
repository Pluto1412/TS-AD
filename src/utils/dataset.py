import os
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import torch
from torch.utils.data import ConcatDataset, Dataset

def recover_patch(seqs_with_patch, patch_stride=1, seq_stride=1):
    # if isinstance(seqs_with_patch, list):
    #     seq_patches = torch.stack(seqs_with_patch, dim=0)

    # Without Avg
    def flatten_patch(seq_with_patch_ls: list | torch.Tensor, stride=1):
        return_tensor = False
        if isinstance(seq_with_patch_ls, torch.Tensor):
            seq_with_patch_ls = [seq_with_patch_ls]
            return_tensor = True

        flattened = []
        for seq_with_path in seq_with_patch_ls:
            flattened.append(torch.cat([seq_with_path[:-1, :stride].reshape(-1), seq_with_path[-1].reshape(-1)], dim=0))

        if len(flattened) == 1 and return_tensor:
            return flattened[0]
        else:
            return flattened

    if isinstance(seqs_with_patch, torch.Tensor):
        sample_num = seqs_with_patch.shape[0]

        seq_without_patch = torch.cat([seqs_with_patch[:, :-1, :patch_stride].reshape(sample_num, -1), seqs_with_patch[:, -1].reshape(sample_num, -1)], dim=1)
        return torch.cat([seq_without_patch[:-1, :seq_stride].reshape(-1), seq_without_patch[-1].reshape(-1)], dim=0)

    seq_without_patch = torch.stack(flatten_patch(seqs_with_patch, patch_stride), dim=0)
    one_dim_seq  = flatten_patch(seq_without_patch, seq_stride)

    print(one_dim_seq.shape)

    return one_dim_seq



class UTSDataset(Dataset):
    def __init__(self, seqs, win_len, seq_len, labels=None) -> None:
        super().__init__()

        assert len(seqs) > win_len + seq_len - 2

        self.seq_len = seq_len
        self.win_len = win_len
        self.labels = labels

        self.seqs = self.time_delay_embedding(seqs, win_len)

        if labels is not None:
            self.labels = self.time_delay_embedding_for_label(labels, win_len)


    def __len__(self):
        return len(self.seqs) - self.seq_len + 1
    

    def time_delay_embedding(self, seqs, win_len):
        return torch.tensor(np.array([seqs[i-win_len : i] for i in range(win_len, len(seqs)+1)]), dtype=torch.float32)
    
    def time_delay_embedding_for_label(self, labels, win_len):
        return torch.tensor(labels[win_len-1:], dtype=torch.int)
    

    def __getitem__(self, index):
        if self.labels is not None:
            return self.seqs[index : index + self.seq_len], self.labels[index + self.seq_len - 1]
        else:
            return self.seqs[index : index + self.seq_len]
        

class DenoisedUTSDataset(UTSDataset):
    def __init__(self, seqs, win_len, seq_len, labels=None) -> None:
        denoised_seqs = DenoisedUTSDataset.wavelet_denoising(seqs)
        super().__init__(denoised_seqs, win_len, seq_len, labels)

    @classmethod
    def wavelet_denoising(cls, data: np.ndarray):
        db4 = pywt.Wavelet('db4')
        coeffs = pywt.wavedec(data, db4)
        coeffs[len(coeffs)-1] *= 0
        coeffs[len(coeffs)-2] *= 0
        meta = pywt.waverec(coeffs, db4)

        if len(data) % 2 == 1:
            meta = meta[:-1]

        return meta
    

class UTSDatasetWithHistorySliding(UTSDataset):
    def __init__(self, seqs, win_len, seq_len, labels=None, seq_stride=1) -> None:
        super().__init__(seqs, win_len, seq_len, labels)

        self.seq_stride = seq_stride

    def __getitem__(self, index):
        end_index = index + self.seq_len
        start_index = end_index - 1 - (self.seq_len - 1) * self.seq_stride

        if start_index < 0:
            start_index %= self.seq_stride

        seqs = self.seqs[start_index : end_index : self.seq_stride]

        padding_len = self.seq_len - len(seqs)
        if padding_len > 0:
            padding = torch.zeros((padding_len, self.win_len))
            seqs = torch.cat((padding, seqs), dim=0)



        if self.labels is not None:
            return seqs, self.labels[end_index - 1]
        else:
            return seqs


class KAD_DisformerTrainSet(Dataset):
    def __init__(self, seqs, win_len, seq_len, labels=None, seq_stride=1) -> None:
        self.seq_len = seq_len
        self.win_len = win_len
        self.seq_stride = seq_stride

        self.context_flow = UTSDataset(seqs, win_len, seq_len, labels)
        self.history_flow = UTSDatasetWithHistorySliding(seqs, win_len, seq_len, labels, seq_stride)

    def __len__(self):
        return len(self.context_flow)


    def __getitem__(self, index):
        return self.context_flow[index], self.history_flow[index]




class KAD_DisformerTestSet(Dataset):
    def __init__(self, seqs, win_len, seq_len, labels=None, seq_stride=1) -> None:
        self.seq_len = seq_len
        self.win_len = win_len
        self.seq_stride = seq_stride

        self.context_flow = UTSDataset(seqs, win_len, seq_len, labels)
        self.history_flow = UTSDatasetWithHistorySliding(seqs, win_len, seq_len, labels, seq_stride)

    def __len__(self):
        return len(self.context_flow)


    def __getitem__(self, index):
        return self.context_flow[index], self.history_flow[index]


def normalize_series(seqs, eps: float = 1e-6):
    seqs = np.asarray(seqs, dtype=np.float32)
    mean = float(seqs.mean())
    std = float(seqs.std())
    if std < eps:
        std = 1.0
    return (seqs - mean) / std


def load_single_series_csv(csv_path, normalize: bool = False):
    df = pd.read_csv(csv_path)
    if "value" not in df.columns:
        raise ValueError(f"CSV does not contain 'value' column: {csv_path}")

    seqs = df["value"].to_numpy(dtype=np.float32)
    if normalize:
        seqs = normalize_series(seqs)
    return seqs


def discover_kpi_train_csvs(data_path):
    data_path = Path(data_path)
    if data_path.is_file():
        return [data_path]

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    direct_train_csv = data_path / "train.csv"
    if direct_train_csv.is_file():
        return [direct_train_csv]

    train_csvs = sorted(
        child / "train.csv"
        for child in data_path.iterdir()
        if child.is_dir() and (child / "train.csv").is_file()
    )
    if not train_csvs:
        raise FileNotFoundError(
            f"No train.csv found at {data_path} or its immediate KPI subdirectories."
        )
    return train_csvs


def build_pretrain_dataset(
    data_path,
    win_len,
    seq_len,
    seq_stride=1,
    normalize_per_kpi: bool = False,
):
    train_csvs = discover_kpi_train_csvs(data_path)
    datasets = []
    stats = []

    for csv_path in train_csvs:
        seqs = load_single_series_csv(csv_path, normalize=normalize_per_kpi)
        if len(seqs) <= win_len + seq_len - 2:
            continue

        dataset = KAD_DisformerTrainSet(
            seqs,
            win_len=win_len,
            seq_len=seq_len,
            seq_stride=seq_stride,
        )
        datasets.append(dataset)
        stats.append(
            {
                "source": str(csv_path),
                "points": len(seqs),
                "samples": len(dataset),
            }
        )

    if not datasets:
        raise ValueError("No usable KPI train.csv files were found for pretraining.")

    if len(datasets) == 1:
        return datasets[0], stats
    return ConcatDataset(datasets), stats
    

def train_test_split(data, train_ratio=0.8):
    train_cnt = round(train_ratio * len(data))
    train_data = data[:train_cnt]
    test_data = data[train_cnt:]
    return train_data, test_data


def load_csvs(csv_path):
    if os.path.isdir(csv_path):
        uids = [i[:-4] for i in os.listdir(csv_path) if i.endswith(".csv")]
        datasets = {uid: pd.read_csv(os.path.join(csv_path, uid+".csv")) for uid in uids}
    else:
        datasets = pd.read_csv(csv_path)

    return datasets


if __name__ == '__main__':
    import pandas as pd

    raw_ts = pd.read_csv("../../data/train.csv")[['value', 'label']].to_numpy()
    dd = DenoisedUTSDataset(raw_ts[:, 0], 20, 120, raw_ts[:, 1])
    d = UTSDataset(raw_ts[:, 0], 20, 120, raw_ts[:, 1])


    raw_ts = np.arange(1, 10)
    print(raw_ts)

    d = UTSDatasetWithHistorySliding(raw_ts, 1, 3, seq_stride=1)
    d2 = UTSDatasetWithHistorySliding(raw_ts, 1, 3, seq_stride=3)

    print(d2[0], d2[6], sep='\n')
