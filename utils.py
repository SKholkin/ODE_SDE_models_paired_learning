import torch
import matplotlib.pyplot as plt

def unpaint_center(images, mask_size=0.5):
    img_size = images.shape[-1]
    mask = torch.ones_like(images)
    mask[:, :, int(mask_size / 2 * img_size):img_size - int(mask_size / 2 * img_size), int(mask_size / 2 * img_size):img_size - int(mask_size / 2 * img_size)] = 0
    return torch.mul(images, mask)

def save_picutes_cond_gen(model, x_0, save_name='pics.jpeg'):
    x_1 = model.sample(x_0)
    n_pics = 8
    x_0 = x_0[:n_pics]
    x_1 = x_1[:n_pics]
    x_0, x_1 = x_0.permute([1, 0, 2, 3]), x_1.permute([1, 0, 2, 3])
    x_0 = torch.cat([x_0[:, i] for i in range(n_pics)], dim=1) * 0.5 + 0.5
    x_1 = torch.cat([x_1[:, i] for i in range(n_pics)], dim=1) * 0.5 + 0.5
    print(x_0.shape)
    # x_0 = x_0.reshape([3, 2 * x_0.shape[2], 4 * x_0.shape[3]]) * 0.5 + 0.5
    # x_1 = x_1.reshape([3, 2 * x_1.shape[2], 4 * x_1.shape[3]]) * 0.5 + 0.5

    x_0, x_1 = x_0.squeeze().permute([1, 2, 0]), x_1.squeeze().permute([1, 2, 0])

    out = torch.cat([x_0, x_1], dim=1).cpu().numpy()

    plt.imshow(out)
    plt.savefig(save_name)
