import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    generate_examples
)
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm
import config
from hmdataset import HeightmapDataset

torch.backends.cudnn.benchmark = True

def get_loader(image_size, step):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)], 
                [0.5 for _ in range(config.CHANNELS_IMG)]
            ),
        ]
    )
    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    dataset = HeightmapDataset(npz_path=config.NPZ_PATH, transform=transform, image_size=image_size, e_mean_range=config.E_MEAN_RANGE, e_stdev_range=config.E_STDEV_RANGE)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return loader, dataset

def train_fn(
    critic, 
    gen, 
    loader,
    dataset, 
    step, 
    alpha, 
    opt_critic, 
    opt_gen,
    tensorboard_step, 
    writer, 
    scaler_gen, 
    scaler_critic
):
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]
        
        # Train Critic: max E[critic(real)] - E[critic(fake)]
        noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)
        assert not torch.isnan(noise).any(), "Noise contains NaN values"

        
        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + config.LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()
        
        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / (
            (config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
        )
        alpha = min(alpha, 1)
        
        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
            plot_to_tensorboard(
                writer, 
                loss_critic.item(), 
                loss_gen.item(),
                real.detach(), 
                fixed_fakes.detach(), 
                tensorboard_step
            )
            tensorboard_step += 1
            
        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )
        
    return tensorboard_step, alpha

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    gen = Generator(
        config.Z_DIM, 
        config.IN_CHANNELS, 
        img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)
    
    critic = Discriminator(
        config.Z_DIM, 
        config.IN_CHANNELS, 
        img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)
    
    total_gen_params = count_parameters(gen)
    print(f"Total trainable generator parameters: {total_gen_params}")

    total_critic_params = count_parameters(critic)
    print(f"Total trainable critic parameters: {total_critic_params}")
    
    opt_gen = optim.Adam(
        gen.parameters(), 
        lr=config.LEARNING_RATE, 
        betas=(0.0, 0.99)
    )
    opt_critic = optim.Adam(
        critic.parameters(), 
        lr=config.LEARNING_RATE, 
        betas=(0.0, 0.99)
    )
    
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()
    
    
    
    writer = SummaryWriter(config.LOG_PATH)
    
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE
        )
        
    gen.train()
    critic.train()
    tensorboard_step = 0
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        loader, dataset = get_loader(4 * 2 ** step, step)
        print(f"Current image size: {4 * 2 ** step}")
        
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            tensorboard_step, alpha = train_fn(
                critic, 
                gen, 
                loader,
                dataset, 
                step, 
                alpha, 
                opt_critic, 
                opt_gen,
                tensorboard_step, 
                writer, 
                scaler_gen, 
                scaler_critic
            )
            
            if config.SAVE_MODEL:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(critic, opt_critic, filename=config.CHECKPOINT_CRITIC)
        
        step += 1
        
        

if __name__ == "__main__":
    main()