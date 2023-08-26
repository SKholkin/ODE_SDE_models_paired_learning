import argparse
from flow_mathcing import FlowMathcing
from unet import UNet
from utils import unpaint_center, save_picutes_cond_gen, gaussian_blurring
import torch
import torchvision
from tqdm import tqdm as tqdm
import wandb
from PIL import Image

def train_epoch(model, opt, loader, corrupt_fn, device):
    model.train()
    loss_storage = []
    pbar = tqdm(enumerate(iter(loader)))

    for itr, item in pbar:
        x_1, y = item
        x_0 = corrupt_fn(x_1)
        x_0, x_1 = x_0.to(device), x_1.to(device)
        t = torch.rand(x_1.shape[0]).to(device)
        loss = model.step(x_0, x_1, t)

        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_storage.append(loss.item())
        pbar.set_description(f'Loss: {torch.mean(torch.tensor(loss_storage))}')
        if wandb.run:
            wandb.log({'loss': loss.item()})


tasks_choices = ['inpainting', 'deblurring_gaussian']
if __name__ == '__main__':

    device = 'cuda:1'
    parser = argparse.ArgumentParser("Paired learning")
    parser.add_argument('--task', choices=tasks_choices, type=str, default='inpainting')
    parser.add_argument('--dataset', choices=['mnist', 'cifar', 'celeba'], type=str, default='cifar')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    if args.wandb:
        wandb.init(project='Paired/Unpaired Learning', config={'task': args.task, 'dataset': args.dataset})

    if args.dataset == 'cifar': 

        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        unet = UNet(3, base_channels=128, channel_mults=(1, 2, 4, 8))
    elif args.dataset == 'mnist':

        mnist_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                torchvision.transforms.Normalize([0.5],[0.5])
            ])
        train_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=mnist_transform, download=True)
        unet = UNet(3, base_channels=128, channel_mults=(1, 2, 4))
    elif args.dataset.lower() == 'celeba':
        celeba_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((64, 64)),
                torchvision.transforms.Normalize([0.5],[0.5])
            ])
        train_dataset = torchvision.datasets.CelebA(root="data", split='train', transform=celeba_transform, download=True)
        unet = UNet(3, base_channels=256, channel_mults=(1, 2, 4, 8))
    else:
        raise ValueError('Wrong dataset name')
    
    if args.task == 'inpainting':
        corrupt_fn = unpaint_center
    elif args.task == 'deblurring_gaussian':
        corrupt_fn = gaussian_blurring

    fm_model = FlowMathcing(unet)
    fm_model.to(device)
    opt = torch.optim.AdamW(fm_model.parameters(), lr=1e-4)

    epochs = 200
    batch_size = 64
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=0, drop_last=True)
    for epoch in range(epochs):
        print(f'Epoch {epoch} starts...')
        train_epoch(fm_model, opt, loader, corrupt_fn, device)

        torch.save(fm_model.state_dict(), f'fm_model_{args.dataset}_{args.task}.pth')
        save_pic_name = f'pics/{args.dataset}_{args.task}_{epoch}.jpeg'
        x_0_sample = next(iter(loader))[0].to(device)
        save_picutes_cond_gen(fm_model, corrupt_fn(x_0_sample), save_name=save_pic_name, gt=x_0_sample)

        if wandb.run:
            im = Image.open(save_pic_name)
            wandb.log({"Samples": wandb.Image(im)})
