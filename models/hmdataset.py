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
    def __init__(self, npz_path, transform=None, image_size=config.IMAGE_SIZE, e_mean_range=None, e_stdev_range=None):
        with np.load(npz_path, allow_pickle=True) as data:
            observations = data['observations'].tolist()

        # Filter observations based on e_mean and e_stdev ranges
        if e_mean_range or e_stdev_range:
            filtered_observations = []
            for observation in observations:
                features = observation['features']
                e_mean = features['e_mean']
                e_stdev = features['e_stdev']
                if (e_mean_range is None or e_mean_range[0] <= e_mean <= e_mean_range[1]) and \
                   (e_stdev_range is None or e_stdev_range[0] <= e_stdev <= e_stdev_range[1]):
                    filtered_observations.append(observation)
            self.observations = filtered_observations
        else:
            self.observations = observations

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

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

if __name__ == "__main__":
    npz_path = "data/data.npz"  # Update this path to your npz file
    dataset = HeightmapDataset(npz_path, e_stdev_range=[60.0, 1000.0])
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Get a batch of 10 random images and their labels
    images, labels = next(iter(dataloader))
    
    print(f"Size of dataset: {len(dataset)}")

    # Plot the images
    fig, axs = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        # Assuming the images are single-channel, you can modify this line if they are multi-channel
        axs[i].imshow(images[i].squeeze().numpy(), cmap="gray")
        axs[i].axis('off')
    plt.show()