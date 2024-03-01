import torch
from tqdm import tqdm

def sample_x(x_0, t, bar_alpha, bar_beta, noise=None):
    """
        Add noise to x_0 to get x_t
        t is [batch_size]
    """
    if not noise: noise = torch.randn_like(x_0)
    shape = (x_0.shape[0], 1, 1, 1)
    return x_0 * bar_alpha[t].view(shape) + noise * bar_beta[t].view(shape), noise

def diffusion_loss(episilon, model_output):
    diff = episilon - model_output
    loss = torch.sqrt(torch.sum(diff ** 2))
    return loss

class Trainer:
    def __init__(self, model, learning_rate, T, device, model_save_path='./model.pth'):
        self.model = model
        self.model.to(device)
        self.T = T
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.model_save_path = model_save_path

    def train_step(self, x_t, t, episilon):
        output = self.model(x_t, t)
        loss = diffusion_loss(episilon, output)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train(self, dataloader, num_epochs, bar_alpha, bar_beta, noise=None):
        self.model.train()
        for epoch in range(num_epochs):
            pbar = tqdm(dataloader)
            total_loss = 0
            for i, x_0 in enumerate(pbar):
                cur_batch = x_0.shape[0]
                t = torch.randint(0, self.T, (cur_batch,), device=self.device).long()
                x_0 = x_0.to(self.device)
                x_t, episilon = sample_x(x_0, t, bar_alpha, bar_beta, noise)
                total_loss += self.train_step(x_t, t, episilon)
                pbar.set_description(f'Epoch {epoch + 1} Loss: {(total_loss / (i + 1)):.4f}')
            torch.save(self.model.state_dict(), self.model_save_path)