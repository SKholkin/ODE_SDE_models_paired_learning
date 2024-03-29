import argparse
from flow_mathcing import FlowMathcing
from bridge_matching import BridgeMathcing, LinearBridgeNoiseScheduler
from unet import UNet
from utils import unpaint_center, save_picutes_cond_gen, gaussian_blurring, inverse_to_uint8_pic
import torch
import torchvision
from tqdm import tqdm as tqdm
import wandb
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance

def train_epoch(model, opt, loader, corrupt_fn, device, grad_acc):
    model.train()
    loss_storage = []
    pbar = tqdm(enumerate(list(iter(loader))))
    opt.zero_grad()

    t_eps = 1e-3
    for itr, item in pbar:
        x_1, y = item
        x_0 = corrupt_fn(x_1)
        x_0, x_1 = x_0.to(device), x_1.to(device)
        t = (torch.rand(x_1.shape[0]).to(device)) * (1 - 2 * t_eps) + t_eps
        loss = model.step(x_0, x_1, t)

        loss.backward()
        clipping_value = 0.1 # arbitrary value of your choosing
        torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
        if itr % grad_acc == 0:
            opt.step()
            opt.zero_grad()
            # print('step')

        loss_storage.append(loss.item())
        pbar.set_description(f'Loss: {torch.mean(torch.tensor(loss_storage))}')
        if wandb.run:
            wandb.log({'loss': loss.item()})

        if itr > 1000:
            break

def calculate_fid(test_loader, corrupt_fn, model, device):
    fid = FrechetInceptionDistance(feature=64).to(device)
    print(f'Computing FID')
    pbar = tqdm(list(enumerate(iter(test_loader))))

    for itr, item in pbar:
        x_1_gt, y = item
        x_0 = corrupt_fn(x_1_gt)
        x_0, x_1_gt = x_0.to(device), x_1_gt.to(device)
        x_1_sampled = model.sample(x_0, pbar=False)

        fid.update(inverse_to_uint8_pic(x_1_gt), real=True)
        fid.update(inverse_to_uint8_pic(x_1_sampled), real=False)
        
        if itr > 100:
            break

    ret_val = fid.compute()
    if wandb.run:
        wandb.log({'fid': ret_val})
    return ret_val
        

tasks_choices = ['inpainting', 'deblurring_gaussian']
if __name__ == '__main__':

    device = 'cuda:1'
    parser = argparse.ArgumentParser("Paired learning")
    parser.add_argument('--task', choices=tasks_choices, type=str, default='inpainting', help='The task for paired learning')
    parser.add_argument('--dataset', choices=['mnist', 'cifar', 'celeba'], type=str, default='cifar')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False, help='Turns on/off wandb logging')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--fid', action=argparse.BooleanOptionalAction, default=False, help='Turns on/off FID calculation. Be aware as it takes a lot of time')
    parser.add_argument('--model', choices=['fm', 'bm'], default=['bm'], type=str, help='Choose model fm for Flow Matching, bm for Bridge Matching')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--grad_acc', type=int, default=1, help='Number of gradient accumulation steps')

    args = parser.parse_args()

    if args.wandb:
        wandb.init(project='Paired Unpaired Learning', config={'task': args.task, 'dataset': args.dataset})

    if args.dataset == 'cifar': 
        
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
        unet = UNet(3, base_channels=128, channel_mults=(1, 2, 4, 8))
    elif args.dataset == 'mnist':

        mnist_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                torchvision.transforms.Normalize([0.5],[0.5])
            ])
        train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=mnist_transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=mnist_transform, download=True)
        unet = UNet(3, base_channels=128, channel_mults=(1, 2, 4))
    elif args.dataset.lower() == 'celeba':
        celeba_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((64, 64)),
                torchvision.transforms.Normalize([0.5],[0.5])
            ])
        train_dataset = torchvision.datasets.CelebA(root="data", split='train', transform=celeba_transform, download=True)
        test_dataset = torchvision.datasets.CelebA(root="data", split='valid', transform=celeba_transform, download=True)
        unet = UNet(3, base_channels=128, channel_mults=(1, 2, 4, 8))
    else:
        raise ValueError('Wrong dataset name')
    
    if args.task == 'inpainting':
        corrupt_fn = unpaint_center
    elif args.task == 'deblurring_gaussian':
        corrupt_fn = gaussian_blurring

    if args.model == 'fm':
        model = FlowMathcing(unet)
    else:
        noise_sch = LinearBridgeNoiseScheduler(max_beta=1.5e-4)
        model = BridgeMathcing(unet, noise_sch)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    
    epochs = 100
    batch_size = args.batch_size
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=0, drop_last=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=0, drop_last=True)
    for epoch in range(epochs):
        print(f'Epoch {epoch} starts...')
        train_epoch(model, opt, loader, corrupt_fn, device, args.grad_acc)

        save_pic_name = f'pics/{args.dataset}_{args.task}_{epoch}.jpeg'
        x_0_sample = next(iter(loader))[0].to(device)
        save_picutes_cond_gen(model, corrupt_fn(x_0_sample), save_name=save_pic_name, gt=x_0_sample)

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'{args.model}_model_{args.dataset}_{args.task}_ep_{epoch}.pth')

        if (epoch + 1) % 25 == 0 and args.fid:
            fid = calculate_fid(test_loader, corrupt_fn, model, device)
            print(f'FID: {fid}')

        # if wandb.run:
        #     im = Image.open(save_pic_name)
        #     wandb.log({"Samples": wandb.Image(im)})
