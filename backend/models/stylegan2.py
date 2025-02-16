import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
import numpy as np
import random

class AugmentPipe:
    def __init__(self, p=0.0, target_p=0.6, max_p=0.85, speed=1e-5):
        self.p = p  # Current augmentation probability
        self.target_p = target_p  # Target augmentation probability
        self.max_p = max_p  # Maximum augmentation probability
        self.speed = speed  # How quickly to adjust p
        
        # Initialize augmentation parameters
        self.xflip = 0.5
        self.rotate90 = 0.25
        self.xint = 0.1
        self.scale = 0.2
        self.rotate = 0.2
        self.aniso = 0.2
        self.xfrac = 0.2
        self.brightness = 0.2
        self.contrast = 0.2
        self.saturation = 0.2
        self.hue = 0.1
        
    def update(self, real_sign):
        # Update p based on sign of D(real)
        self.p = min(self.max_p, self.p + (real_sign * self.speed))
        return self.p
    
    def apply(self, img):
        if random.random() > self.p:
            return img
            
        # Apply random augmentations with their respective probabilities
        if random.random() < self.xflip:
            img = TF.hflip(img)
            
        if random.random() < self.rotate90:
            k = random.randint(1, 3)
            img = TF.rotate(img, 90 * k)
            
        if random.random() < self.brightness:
            factor = random.uniform(0.5, 1.5)
            img = TF.adjust_brightness(img, factor)
            
        if random.random() < self.contrast:
            factor = random.uniform(0.5, 1.5)
            img = TF.adjust_contrast(img, factor)
            
        if random.random() < self.saturation:
            factor = random.uniform(0.5, 1.5)
            img = TF.adjust_saturation(img, factor)
            
        if random.random() < self.hue:
            factor = random.uniform(-0.1, 0.1)
            img = TF.adjust_hue(img, factor)
            
        if random.random() < self.scale:
            scale = random.uniform(0.8, 1.2)
            size = img.shape[-2:]
            img = TF.resize(img, [int(size[0] * scale), int(size[1] * scale)])
            img = TF.center_crop(img, size)
            
        return img

class ImageDataset(Dataset):
    def __init__(self, image_dir, image_size=256, augment_pipe=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.augment_pipe = augment_pipe

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        if self.augment_pipe is not None:
            image = self.augment_pipe.apply(image)
            
        return image

class StyleGAN2Trainer:
    def __init__(self, image_dir, image_size=256, batch_size=32, lr=0.002, use_ada=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.use_ada = use_ada
        
        # Initialize ADA
        self.augment_pipe = AugmentPipe() if use_ada else None
        
        # Initialize dataset with ADA
        self.dataset = ImageDataset(image_dir, image_size, self.augment_pipe)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        # Initialize models (placeholder for now)
        self.generator = nn.Sequential(
            nn.Linear(512, 512 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        ).to(self.device)
        
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Training metrics
        self.g_losses = []
        self.d_losses = []
        self.ada_stats = {'p': [], 'rt': []}  # ADA statistics
        
    def compute_ada_rt(self, d_real):
        # Compute sign of real samples' discriminator output
        sign = (d_real > 0.5).float().mean().item()
        return sign
        
    def train_step(self, real_images):
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        
        # Train discriminator
        self.d_optimizer.zero_grad()
        
        label_real = torch.ones(batch_size, 1).to(self.device)
        label_fake = torch.zeros(batch_size, 1).to(self.device)
        
        # Get discriminator predictions for real images
        d_real = self.discriminator(real_images)
        d_real_loss = F.binary_cross_entropy(d_real, label_real)
        
        # Update ADA probability based on discriminator's performance on real images
        if self.use_ada:
            rt = self.compute_ada_rt(d_real)
            ada_p = self.augment_pipe.update(rt - self.augment_pipe.target_p)
            self.ada_stats['p'].append(ada_p)
            self.ada_stats['rt'].append(rt)
        
        # Generate and classify fake images
        z = torch.randn(batch_size, 512).to(self.device)
        fake_images = self.generator(z)
        d_fake = self.discriminator(fake_images.detach())
        d_fake_loss = F.binary_cross_entropy(d_fake, label_fake)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train generator
        self.g_optimizer.zero_grad()
        
        d_fake = self.discriminator(fake_images)
        g_loss = F.binary_cross_entropy(d_fake, label_real)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item(), d_loss.item()
    
    def train(self, num_epochs, callback=None):
        for epoch in range(num_epochs):
            g_losses = []
            d_losses = []
            
            for i, real_images in enumerate(self.dataloader):
                g_loss, d_loss = self.train_step(real_images)
                g_losses.append(g_loss)
                d_losses.append(d_loss)
                
                if callback:
                    metrics = {
                        'epoch': epoch + 1,
                        'batch': i + 1,
                        'g_loss': g_loss,
                        'd_loss': d_loss,
                        'progress': (epoch * len(self.dataloader) + i + 1) / (num_epochs * len(self.dataloader))
                    }
                    
                    if self.use_ada:
                        metrics.update({
                            'ada_p': self.augment_pipe.p,
                            'ada_rt': self.ada_stats['rt'][-1] if self.ada_stats['rt'] else 0
                        })
                        
                    callback(metrics)
            
            self.g_losses.append(np.mean(g_losses))
            self.d_losses.append(np.mean(d_losses))
    
    def generate_samples(self, num_samples=1):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, 512).to(self.device)
            samples = self.generator(z)
            samples = (samples + 1) / 2  # Denormalize
            samples = samples.cpu()
        self.generator.train()
        return samples
    
    def save_model(self, path):
        save_data = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'use_ada': self.use_ada
        }
        
        if self.use_ada:
            save_data.update({
                'ada_stats': self.ada_stats,
                'augment_pipe_state': {
                    'p': self.augment_pipe.p,
                    'target_p': self.augment_pipe.target_p,
                    'max_p': self.augment_pipe.max_p,
                    'speed': self.augment_pipe.speed
                }
            })
            
        torch.save(save_data, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']
        
        if checkpoint.get('use_ada', False):
            self.use_ada = True
            self.ada_stats = checkpoint['ada_stats']
            pipe_state = checkpoint['augment_pipe_state']
            self.augment_pipe = AugmentPipe(
                p=pipe_state['p'],
                target_p=pipe_state['target_p'],
                max_p=pipe_state['max_p'],
                speed=pipe_state['speed']
            )