import argparse
from flow_mathcing import FlowMathcing
from unet import UNet
from utils import unpaint_center, save_picutes_cond_gen
import torch
import torchvision
from tqdm import tqdm as tqdm

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
        # if itr % 100 == 0:
        #     print(f'Iter {itr} Loss {torch.mean(torch.tensor(loss_storage))}')

        pbar.set_description(f'Loss: {torch.mean(torch.tensor(loss_storage))}')
    # torch.save(fm_model.state_dict(), f'fm_model_cifar_32_ep_{epoch}.pth')


tasks_choices = ['inpainting']
if __name__ == '__main__':

    device = 'cuda:1'
    parser = argparse.ArgumentParser("Paired learning")
    parser.add_argument('--task', choices=tasks_choices, type=str, default='inpainting')
    parser.add_argument('--dataset', choices=['mnist', 'cifar'], type=str, default='cifar')

    args = parser.parse_args()

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
    else:
        raise ValueError('Wrong dataset name')
    
    if args.task == 'inpainting':
        corrupt_fn = unpaint_center

    fm_model = FlowMathcing(unet)
    fm_model.to(device)
    opt = torch.optim.AdamW(fm_model.parameters(), lr=1e-4)

    epochs = 100
    batch_size = 128
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, drop_last=True)
    for epoch in range(epochs):
        print(f'Epoch {epoch} starts...')
        train_epoch(fm_model, opt, loader, corrupt_fn, device)

        torch.save(fm_model.state_dict(), f'fm_model_{args.dataset}_{args.task}.pth')

        save_picutes_cond_gen(fm_model, corrupt_fn(next(iter(loader))[0].to(device)), save_name=f'pics/{args.dataset}_{args.task}_{epoch}.jpeg')
