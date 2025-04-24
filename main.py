from scheduler import Scheduler
from SwinUnet import SwinUnet
from dataset import CustomDataset

import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils

from torch.utils.data import DataLoader
from torch import optim
import torch


if __name__ == "__main__":
    
    epochs = 1000
    device = torch.device("mps")
    denoise_steps = 1000

    dataset = CustomDataset("images")
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = SwinUnet(channels=3, dim=96, mlp_ratio=4, patch_size=2, window_size=4,
                    depth=[2, 2, 6, 2], nheads=[3, 6, 12, 24], use_condition=False).to(device)
    scheduler = Scheduler(model, denoise_steps)

    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    for e in range(1, epochs + 1):
        
        model.train()
        losses = []

        for i, images in enumerate(data_loader):

            img = torch.tensor(images).float().to(device)

            t = torch.randint(low=0, high=denoise_steps, size=(img.shape[0],)).to(device)

            loss = scheduler.training_losses(img, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            
        torch.save(model.state_dict(), f"swin-unet-cifar.pth")
        
        if e % 10 == 0:
            model.eval()
            with torch.no_grad():

                    gen_img = scheduler.ddim((64, 3, 64, 64), device)

                    vutils.save_image(gen_img, f"results/epoch_{e}.png", normalize=True, nrow=8)

        print("Epochs: {}/{}, Losses: {:.2f}".format(e, epochs, sum(losses)/len(losses)))