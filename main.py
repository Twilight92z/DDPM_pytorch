import yaml
import torch
import argparse
from diffusion import Diffusion
from dataloader import CustomDataset, show_pic
from torch.utils.data import DataLoader
from train import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--nums', type=int, default=10)
    parser.add_argument('--model_path', type=str, default="./model.pth")
    parser.add_argument('--output_path', type=str, default="./test.png")
    args = parser.parse_args()
    return args

def predict_x(model, shape, T, alpha, beta, bar_beta, sigma, device, t=0, noise=None, random_sample=True):
    """
        Generate from noise to x_t
        Shape is [batch_size, channels, img_size, img_size]
    """
    if noise is None: noise = torch.randn(shape, device=device)
    process = [torch.clip(noise, -1, 1)]
    with torch.no_grad():
        for ts in range(t, T):
            real_t = T - ts - 1
            batch_t = torch.tensor([real_t] * shape[0], device=device).long()
            factor = (beta[batch_t]**2 / bar_beta[batch_t])[:, None, None, None]
            output = model(noise, batch_t)
            noise -= output * factor
            noise /= alpha[batch_t][:, None, None, None]
            if random_sample: noise += torch.randn(shape, device=device) * sigma[batch_t][:, None, None, None]
            process.append(torch.clip(noise, -1, 1))
    noise = torch.clip(noise, -1, 1)
    return noise, torch.stack(process, dim=0)

def train(path, img_size, batch_size, T, lr, device, epochs, bar_alpha, bar_beta, model_path):
    dataset = CustomDataset(path, img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    trainer = Trainer(model, lr, T, device, model_path)
    trainer.train(dataloader, epochs, bar_alpha, bar_beta)

def predict(model, shape, T, alpha, beta, bar_beta, sigma, device, noise, output_path):
    _, process = predict_x(model, shape, T, alpha, beta, bar_beta, sigma, device, noise=noise, random_sample=True)
    process = process.permute(1, 0, 3, 4, 2)
    show_pic(process, output_path)

if __name__ == "__main__":
    # 配置文件
    with open("config.yml", "r") as f: configs = yaml.safe_load(f)
    embedding_size = configs.get("embedding_size", 128)
    batch_size = configs.get("batch_size", 32)
    img_size = configs.get("img_size", 128)
    blocks = configs.get("blocks", 2)
    channels = configs.get("channels", [1, 1, 2, 2, 4, 4])
    device = torch.device(configs.get("device", "cuda:1"))
    lr = configs.get("lr", 5e-5)
    epochs = configs.get("epochs", 1000)
    path = configs.get("path", './data/CIFAR10/train')

    # 超参数选择
    T = 1000
    alpha = torch.sqrt(1 - 0.02 * torch.arange(1, T + 1) / T).to(device)
    beta = torch.sqrt(1 - alpha**2).to(device)
    bar_alpha = torch.cumprod(alpha, dim=0).to(device)
    bar_beta = torch.sqrt(1 - bar_alpha**2).to(device)
    sigma = torch.sqrt(1 - alpha**2).to(device)

    args = parse_args()
    mode = args.mode

    if mode == "train":
        model = Diffusion(T, embedding_size, channels, blocks)
        train(path, img_size, batch_size, T, lr, device, epochs, bar_alpha, bar_beta, args.model_path)
    
    elif mode == "predict":
        shape = (args.nums, 3, img_size, img_size)
        noise = torch.randn(*shape, device=device)
        
        model = Diffusion(T, embedding_size, channels, blocks)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()

        predict(model, shape, T, alpha, beta, bar_beta, sigma, device, noise, args.output_path)
