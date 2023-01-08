from __future__ import print_function

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

manualSeed = 999
# manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = "celeba" # Directory of the dataset
workers = 2 # Number of workers for dataloader
batch_size = 128 # Batch size during training
image_size = 64 # Desired image size to resize images
nc = 3 # Number of color channels in the training images
nz = 100 # Size of the latent vector (the input of the generator)
ngf = 64 # Size of feature maps in generator
ndf = 64 # Size of feature maps in discriminator
num_epochs = 5 # Number of epochs for training
lr = 0.0002 # Learning rate hyperparameter for optimizers
beta1 = 0.5 # Beta1 hyperparameter for Adam optimizers
ngpu = 1 # Number of GPUs that will be used for training (0 for CPU mode)

# The following function initializes the weights of the neural network
# passed in a parameter randomly to mean 0 and standard deviation 0.02.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Neural Network for the generator
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is Z, the latent vector
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # Current size: (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # Current size: (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Current size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Current size: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # State size: (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Neural Network for the Discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Current size: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Current size: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Current size: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Current size: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.main(input)


if __name__ == '__main__':
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)


    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Showing examples from the dataset
    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    # plt.show()

    netG = Generator(ngpu).to(device) # The Generator neural network

    # Handling the multiple GPUs if available and desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    
    # Apply the weights_init function on netG to initialize its weights
    netG.apply(weights_init)

    # # Print the model
    # print(netG)

    netD = Discriminator(ngpu).to(device) # The Discriminator neural network

    # Handling the multiple GPUs if available and desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weight_init function on netD to initialize its weights
    netD.apply(weights_init)

    # # Print the model
    # print(netD)

    # Now, set up the loss functions and optimizers

    # Initialize the Binary Cross Entropy loss (BCELoss) function
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    real_label = 1
    fake_label = 0

    # Setup Adam optimizers that will be used for both neural networks
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Now, the training loop
    
    # To keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Training loop")
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            # (1): Update Discriminator netowrk: maximize log(D(x)) + log(1-D(G(z)))
            # To do this, we will do this in two steps. First, we construct a batch of
            # real samples (from the training data), forward pass through the
            # Discriminator, calculate the loss (log(D(x))), and calculate the gradients
            # in a backward pass.
            # Second, we construct a batch of fake samples using the Generator, forward
            # pass through the Discriminator again, calculate the loss (1-D(G(z))), and
            # calculate the gradients in a backward pass.
            # The first step works towards maximizing log(D(x)), while the second step
            # works towards maximizing (1-D(G(z))), so repeating these two steps, the
            # model is being trained to maximize log(D(x)) + log(1-D(G(z))), which is
            # our goal.

            ## Train the Descriminator with all-real batch
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through Discriminator
            output = netD(real_cpu).view(-1)
            # Now, calculate loss on the real batch. Loss here is log(D(x)).
            errD_real = criterion(output, label)
            # Calculate gradients for Discriminator in backwards pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train the Descriminator with all-fake batch
            # Generate batch of latent vectors using the Generator
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            # Forward the fake batch through Discriminator
            output = netD(fake.detach()).view(-1)
            # Now, calculate loss on the fake batch. Loss here is log(1-D(G(z))).
            errD_fake = criterion(output, label)
            # Calculate gradients for Discriminator in backwards pass
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Compute total error of Discriminator
            errD = errD_real + errD_fake

            # Update D using the optimizer
            optimizerD.step()

            # (2): Update Generator network: maximize log(D(G(z))).
            # To do this, we will run the all-fake batch through the
            # Discriminator again that we just updated. We will then
            # use the Discriminator's output to calculate the loss and
            # update the Generator.

            netG.zero_grad()
            label.fill_(real_label) # We use real_label here even though we are using
                                    # this for the generator. This is because we want
                                    # to use the "log" part of the loss function instead
                                    # of the "1-log" part since we are trying to maximize
                                    # log(D(G(z)))
            # In the first step, we updated the Discriminator. Now, perform another forward
            # pass of all-fake batch through D.
            output = netD(fake).view(-1)
            # Calculate the Generator's loss based on this output.
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G using the optimizer
            optimizerG.step()

            # Output training statistics:
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            
            # Will be plotted later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the Generator is doing by saving its output on fixed noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1


        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

        HTML(ani.to_jshtml())


        # Grab a batch of real images from the dataloader
        real_batch = next(iter(dataloader))

        # Plot the real images
        plt.figure(figsize=(15,15))
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

        # Plot the fake images from the last epoch
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1],(1,2,0)))
        plt.show()

        torch.save(netD, "models//netD.pth")
        torch.save(netG, "models//netG.pth")


