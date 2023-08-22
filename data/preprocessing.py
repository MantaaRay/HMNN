import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

class HeightmapDataset(Dataset):
    def __init__(self, npz_path):
        with np.load(npz_path, allow_pickle=True) as data:
            self.observations = data['observations'].tolist()

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        observation = self.observations[idx]
        features = observation['features']
        e_mean = features['e_mean']
        e_stdev = features['e_stdev']
        map_value = features['map']
        mat = features['mat']
        roughness = features['roughness']
        labels = {
            "e_mean": e_mean,
            "e_stdev": e_stdev,
            "map": map_value,
            "mat": mat,
            "roughness": roughness
        }

        target = observation['target']
        # Normalize the target to [0, 1] range
        target = target / np.max(target)

        return torch.tensor(target, dtype=torch.float32), labels

def load_data(npz_path, batch_size=32):
    dataset = HeightmapDataset(npz_path=npz_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

# Example usage
if __name__ == "__main__":
    npz_path = 'data/data.npz'
    dataloader = load_data(npz_path)

    # Iterate through the data
    for targets, labels in dataloader:
        print(targets.shape)  # Torch tensor containing the heightmaps
        print(labels)         # Dictionary containing the labels
