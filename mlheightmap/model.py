import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
from torch.utils.checkpoint import checkpoint

import pytorch_lightning as pl
from torch.utils.data import random_split
from pytorch_lightning.loggers import TensorBoardLogger


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, RandomRotation

class RandomAugmentation:
    def __init__(self):
        self.transforms = [
            RandomRotation(90),          # Rotate 90 degrees
            RandomRotation(180),         # Rotate 180 degrees
            RandomRotation(-90),         # Rotate 270 degrees
            RandomHorizontalFlip(1.0),   # Always horizontal flip
            RandomVerticalFlip(1.0)      # Always vertical flip
        ]

    def __call__(self, features_tensor, target_tensor):
        # Select a random transform
        transform = random.choice(self.transforms)

        # Apply the transformation to both features and target
        transformed_features = transform(features_tensor)
        transformed_target = transform(target_tensor)

        return transformed_features, transformed_target

class HeightmapDataset(Dataset):
    def __init__(self, npz_file_path, transform=None, num_observations=-1):
        with np.load(npz_file_path, allow_pickle=True) as data:
            observations = data['observations'].tolist()

        # If num_observations is not -1, slice the list of observations
        if num_observations != -1 and num_observations < len(observations):
            observations = observations[:num_observations]
        
        self.observations = self.normalize_observations(observations) # Normalize the observations here
        
        # print (f"Number of observations before D4 transformations: {len(self.observations)}")
        # self.observations = self.apply_d4_transformations(self.observations) # Applying D4 transformations
        # print (f"Number of observations after D4 transformations: {len(self.observations)}")
        self.observations = self.resize_targets(self.observations)  # Resizing the targets

        self.transform = transform
        
    

    def normalize_observations(self, observations):
        normalized_observations = []

        # Collect all targets if you need them for normalization (e.g., finding min and max across all targets)
        targets = [observation['target'] for observation in observations]
        target_min = np.min(targets)
        target_max = np.max(targets)
        
        # Find global min and max for each feature
        feature_mins = {key: np.inf for key in observations[0]['features']}
        feature_maxs = {key: -np.inf for key in observations[0]['features']}

        for observation in observations:
            for key, value in observation['features'].items():
                feature_mins[key] = min(feature_mins[key], np.min(value))
                feature_maxs[key] = max(feature_maxs[key], np.max(value))

        for observation in observations:
            normalized_features = {}
            for key, value in observation['features'].items():
                min_value = feature_mins[key]
                max_value = feature_maxs[key]
                normalized_features[key] = (value - min_value) / (max_value - min_value)

            # Normalizing the target using the global min and max values
            normalized_target = (observation['target'] - target_min) / (target_max - target_min)

            normalized_observation = {
                'features': normalized_features,
                'target': normalized_target
            }
            normalized_observations.append(normalized_observation)
        
        return normalized_observations
    
    def apply_d4_transformations(self, observations):
        transformed_observations = []
        for observation in observations:
            target = observation['target']
            features = observation['features']

            # Original
            transformed_observations.append(observation)

            # R (90-degree rotation clockwise)
            transformed_observations.append({
                'features': features,
                'target': np.rot90(target, -1).copy()
            })

            # R2 (180-degree rotation)
            transformed_observations.append({
                'features': features,
                'target': np.rot90(target, 2).copy()
            })

            # L (90-degree rotation counterclockwise)
            transformed_observations.append({
                'features': features,
                'target': np.rot90(target, 1).copy()
            })

            # H (Horizontal flip)
            transformed_observations.append({
                'features': features,
                'target': np.copy(np.fliplr(target))
            })

            # V (Vertical flip)
            transformed_observations.append({
                'features': features,
                'target': np.copy(np.flipud(target))
            })

            # RH (90-degree rotation + horizontal flip)
            transformed_observations.append({
                'features': features,
                'target': np.copy(np.fliplr(np.rot90(target, -1)))
            })

            # RV (90-degree rotation + vertical flip)
            transformed_observations.append({
                'features': features,
                'target': np.copy(np.flipud(np.rot90(target, -1)))
            })

        return transformed_observations

    
    def resize_targets(self, observations):
        resized_observations = []
        resize_transform = Resize((1024, 1024))

        for observation in observations:
            target = observation['target']
            target_tensor = torch.tensor(target).unsqueeze(0)  # Adding a channel dimension
            resized_target = resize_transform(target_tensor).numpy()  # Removing the channel dimension
            observation['target'] = resized_target
            resized_observations.append(observation)

        return resized_observations



    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        observation = self.observations[idx]
        features = observation['features']
        target = observation['target']

        # Convert the features into a tensor
        features_tensor = torch.tensor([features['e_mean'], features['e_stdev'], features['mat'], features['map'], features['roughness']])

        # Convert the target into a tensor
        target_tensor = torch.tensor(target)

        # Apply transformations if any are defined
        if self.transform:
            features_tensor, target_tensor = self.transform(features_tensor, target_tensor)
        
        # print(f"Features tensor shape: {features_tensor.shape}, target tensor shape: {target_tensor.shape}")

        return features_tensor, target_tensor
    
class HeightmapDataModule(pl.LightningDataModule):

    def __init__(self, npz_file_path, batch_size=32, transform=None, num_observations=-1):
        super().__init__()
        self.npz_file_path = npz_file_path
        self.batch_size = batch_size
        self.transform = transform
        self.num_observations = num_observations

    def prepare_data(self):
        # You can perform actions that are shared across nodes here (e.g., download data)
        pass

    def setup(self, stage=None):
        # Split data into training and validation sets
        dataset = HeightmapDataset(self.npz_file_path, self.transform, self.num_observations)
        train_length = int(len(dataset) * 0.8)
        val_length = len(dataset) - train_length

        self.train_dataset, self.val_dataset = random_split(dataset, [train_length, val_length])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2, pin_memory=True)

class Reshape(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), 512, 32, 32)

class Generator(nn.Module):
    def __init__(self, input_features=0, latent_dim=100, img_shape=(1, 1024, 1024)):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        # Divide the sequential model into parts
        self.part1 = nn.Sequential(
            nn.Linear(input_features + latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512 * 32 * 32),
            nn.ReLU(),
            Reshape()
        )
        self.part2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.part3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z, features=None):
        if (features is not None):
            z = torch.cat([z, features], dim=1)
        
        img = self.part1(z)
        img = self.part2(img)
        img = self.part3(img)
        img = img.view(img.size(0), *self.img_shape)
        return img
    
class Discriminator(nn.Module):
    def __init__(self, input_features=0):
        super(Discriminator, self).__init__()
        
        self.conv_part1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),  # Adding max pooling layer
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)   # Adding max pooling layer
        )
        self.conv_part2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),  # Adding max pooling layer
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)   # Adding max pooling layer
        )

        # You'll need to update the size of the input to the fully connected layer based on the reduced spatial dimensions.
        reduced_dim = 512 * 4 * 4  # This needs to be calculated based on your input size and the convolutions/pooling layers.
        self.fc = nn.Sequential(
            nn.Linear(reduced_dim + input_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, image, features=None):
        x = self.conv_part1(image)
        x = self.conv_part2(x)
        x = x.view(x.size(0), -1)
        
        if (features is not None):
            x = torch.cat([x, features], dim=1)
        
        x = self.fc(x)
        return x


class GAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=0.0002):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        # networks
        self.generator = Generator(latent_dim = self.hparams.latent_dim)
        self.discriminator = Discriminator()
        
        # random noise 
        self.validation_z = torch.randn(6, self.hparams.latent_dim)
        
    def forward(self, z):
        return self.generator(z)
    
    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)
    
    def training_step(self, batch):
        features, imgs = batch
        
        opt_g, opt_d = self.optimizers()

        
        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)
        
        # print(f"imgs shape: {imgs.shape} z shape: {z.shape}")
        
        # Train generator
        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        self.toggle_optimizer(opt_g)
        # adversarial loss is binary cross-entropy
        gan_attempt = self(z)
        # print(f"GAN attempt shape: {gan_attempt.shape}")
        g_loss = self.adversarial_loss(self.discriminator(gan_attempt), valid)
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        # Train discriminator
        # Measure discriminator's ability to classify real from generated samples
        # how well can it label as real?
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        self.toggle_optimizer(opt_d)
        # print(f"Discriminator input shape: {imgs.shape}")
        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

        # how well can it label as fake?
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        self.log_dict({"d_loss": d_loss, "g_loss": g_loss})
            

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []
    
    # def plot_imgs(self):
    #     z = self.validation_z.type_as(self.generator.fc[0].weight)
    #     sample_imgs = self(z).cpu()
        
    #     # print('epoch ', self.current_epoch)
    #     fig = plt.figure()
    #     for i in range(sample_imgs.size(0)):
    #         plt.subplot(2, 3, i+1)
    #         plt.tight_layout()
    #         plt.imshow(sample_imgs.detach()[i, 0, :, :], cmap='gray', interpolation='none')
    #         plt.title("Generated Data")
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.axis('off')
    #     plt.show()
    
    def on_train_epoch_end(self):
        z = self.validation_z.type_as(self.generator.part1[0].weight)
        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        for logger in self.loggers:
            logger.experiment.add_image("generated_images", grid, self.current_epoch)
        pass

if __name__ == "__main__":
    # Hyper-parameters
    NUMEPOCHS = 20
    BATCHSIZE = 20
    LEARNINGRATE = 0.001
    LATENTDIM = 100
    NUMOBSERVATIONS = 1000

    # torch.set_float32_matmul_precision("high")
    model = GAN(latent_dim=LATENTDIM, lr=LEARNINGRATE)

    datamodule = HeightmapDataModule(
        'mlheightmap/featuregathering/observations/observations.npz', batch_size=BATCHSIZE, num_observations=NUMOBSERVATIONS)

    logger = TensorBoardLogger('tb_logs', name='heightmap_model_v0.0.1_testinghparams')
    trainer = pl.Trainer(logger=logger, max_epochs=NUMEPOCHS, log_every_n_steps=5)
    trainer.fit(model, datamodule)
