import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
import config

def custom_transform(target, image_size, channels, flip_prob=0.5):
    # Convert to tensor
    target_tensor = torch.from_numpy(target).float()

    # Normalize the heightmap to the range [0, 1]
    target_tensor = (target_tensor - target_tensor.min()) / (target_tensor.max() - target_tensor.min())

    # Add batch and channel dimensions (assuming specified channels)
    target_tensor = target_tensor.unsqueeze(0).repeat(channels, 1, 1).unsqueeze(0)

    # Resize
    target_resized = F.interpolate(target_tensor, size=(image_size, image_size), mode='bilinear')

    # Random Horizontal Flip
    if torch.rand(1) < flip_prob:
        target_resized = torch.flip(target_resized, [3])

    # Normalize (example: scaling to range [-1, 1])
    mean = torch.tensor([0.5 for _ in range(channels)])
    std = torch.tensor([0.5 for _ in range(channels)])
    target_normalized = (target_resized - mean[:, None, None]) / std[:, None, None]

    # Remove batch dimension
    target_normalized = target_normalized.squeeze(0)
    
    target_as_tensor = torch.tensor(target_normalized, dtype=torch.float32)

    return target_as_tensor

class HeightmapDataset(Dataset):
    def __init__(self, npz_path, transform=None, image_size=config.IMAGE_SIZE):
        with np.load(npz_path, allow_pickle=True) as data:
            self.observations = data['observations'].tolist()
            
        self.transform = transform
        self.image_size = image_size

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
        # target = target / np.max(target)
        
        # print (f"Target shape before transform: {target.shape}")
        target_transformed = custom_transform(target, image_size=self.image_size, channels=config.CHANNELS_IMG) if self.transform else target
        # print(f"Target shape after transform: {target_transformed.shape}")

        return target_transformed, labels

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