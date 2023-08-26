import torch
import matplotlib.pyplot as plt
import torchvision

def unpaint_center(images, mask_size=0.4):
    img_size = images.shape[-1]
    mask = torch.ones_like(images)
    mask[:, :, int(mask_size / 2 * img_size):img_size - int(mask_size / 2 * img_size), int(mask_size / 2 * img_size):img_size - int(mask_size / 2 * img_size)] = 0
    return torch.mul(images, mask)

def save_picutes_cond_gen(model, x_0, save_name='pics.jpeg', gt=None):
    x_1 = model.sample(x_0)
    n_pics = 8
    x_0 = x_0[:n_pics]
    x_1 = x_1[:n_pics]
    x_0, x_1 = x_0.permute([1, 0, 2, 3]), x_1.permute([1, 0, 2, 3])
    x_0 = torch.cat([x_0[:, i] for i in range(n_pics)], dim=1) * 0.5 + 0.5
    x_1 = torch.cat([x_1[:, i] for i in range(n_pics)], dim=1) * 0.5 + 0.5

    x_0, x_1 = x_0.squeeze().permute([1, 2, 0]), x_1.squeeze().permute([1, 2, 0])

    out = torch.cat([x_0, x_1], dim=1)

    if gt is not None:
        gt = gt[:n_pics]
        gt = gt.permute([1, 0, 2, 3])
        gt = torch.cat([gt[:, i] for i in range(n_pics)], dim=1) * 0.5 + 0.5
        gt = gt.squeeze().permute([1, 2, 0])
        out = torch.cat([out, gt], dim=1)

    out = out.cpu().numpy()

    plt.rcParams["figure.figsize"] = [5, 5]
    plt.imshow(out)
    plt.savefig(save_name, dpi = 1000)

def gaussian_blurring(images, max_sigma=3):
    blurring_fn = torchvision.transforms.GaussianBlur(5, sigma=(0.1, max_sigma))
    return blurring_fn(images)
    