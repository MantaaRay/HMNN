import torch
import pytorch_lightning as pl
from models.model import Generator
from models.critic import Critic
from torchvision.utils import make_grid

class WGAN_GP(pl.LightningModule):
    def __init__(self, z_dim=128, lambda_gp=10, learning_rate=0.0002):
        super(WGAN_GP, self).__init__()
        self.automatic_optimization = False
        self.z_dim = z_dim
        self.lambda_gp = lambda_gp
        self.learning_rate = learning_rate

        self.generator = Generator(z_dim=z_dim)
        self.critic = Critic()

    def forward(self, z, depth):
        return self.generator(z, depth)

    def gradient_penalty(self, real_data, fake_data, depth):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

        critic_interpolates = self.critic(interpolates, depth)
        gradients = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(critic_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
        return gradient_penalty

    def training_step(self, batch):
        real_data, _ = batch
        critic_optimizer, generator_optimizer = self.optimizers()
        
        depth = self.current_epoch // 4  # Example of controlling growth every 4 epochs
        
        z = torch.randn(real_data.size(0), self.z_dim, 1, 1, device=self.device)
        

        # Training critic
        self.toggle_optimizer(critic_optimizer)
        
        fake_data = self.generator(z, depth).detach()
        real_critic_output = self.critic(real_data, depth)
        fake_critic_output = self.critic(fake_data, depth)
        gp = self.gradient_penalty(real_data, fake_data, depth)
        critic_loss = fake_critic_output.mean() - real_critic_output.mean() + gp
        
        critic_optimizer.zero_grad()
        self.manual_backward(critic_loss)
        critic_optimizer.step()
        self.untoggle_optimizer(critic_optimizer)
        

        # Training generator
        self.toggle_optimizer(generator_optimizer)
        
        fake_data = self.generator(z, depth)
        fake_critic_output = self.critic(fake_data, depth)
        generator_loss = -fake_critic_output.mean()
        
        generator_optimizer.zero_grad()
        self.manual_backward(generator_loss)
        generator_optimizer.step()
        self.untoggle_optimizer(generator_optimizer)
        
        self.log_dict({"critic_loss": critic_loss, "generator_loss": generator_loss})

    def configure_optimizers(self):
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), self.learning_rate, betas=(0.5, 0.999))
        return [critic_optimizer, generator_optimizer], []
    
    def on_train_epoch_end(self):
        # Get a batch of real data
        real_data, _ = next(iter(self.train_dataloader()))
        real_data = real_data[:16] # Take the first 16 images

        # Generate fixed noise
        fixed_noise = torch.randn(16, self.z_dim, 1, 1, device=self.device)
        depth = self.current_epoch // 4 # Adjust as needed

        # Generate fake images
        with torch.no_grad():
            generated_images = self.generator(fixed_noise, depth)


        # Take the mean across the channels
        mean_image = generated_images.mean(dim=1, keepdim=True)

        # Create grids of real and fake images
        real_grid = make_grid(real_data, nrow=8, normalize=True)
        fake_grid = make_grid(mean_image, nrow=8, normalize=True)

        # Log to TensorBoard
        self.logger.experiment.add_image('real/generated_images', real_grid, global_step=self.current_epoch)
        self.logger.experiment.add_image(f'fake/generated_images_depth_{depth}', fake_grid, global_step=self.current_epoch)
