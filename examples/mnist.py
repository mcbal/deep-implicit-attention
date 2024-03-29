import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

from einops.layers.torch import Rearrange

from deep_implicit_attention.attention import DEQMeanFieldAttention
from deep_implicit_attention.deq import DEQFixedPoint
from deep_implicit_attention.solvers import anderson


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(model, device, train_loader, test_loader, optimizer, epoch, log_interval=10):
    model.train()

    with tqdm(train_loader, unit='it') as tqdm_loader:
        for batch_idx, (data, target) in enumerate(tqdm_loader):
            tqdm_loader.set_description(f'Epoch {epoch}')

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            preds = output.argmax(dim=1, keepdim=True)
            correct = preds.eq(target.view_as(preds)).sum().item()
            accuracy = correct / target.shape[0]
            loss.backward()
            optimizer.step()

            tqdm_loader.set_postfix(loss=loss.item(), accuracy=accuracy)


def test(model, device, test_loader, batch_idx=None):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output,
                                         target, reduction='sum').item()
            preds = output.argmax(dim=1, keepdim=True)
            correct += preds.eq(target.view_as(preds)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{})\n'.format(
            test_loss,
            correct,
            len(test_loader.dataset),
        )
    )


class MNISTNet(nn.Module):
    def __init__(self, dim=10, dim_conv=32, num_spins=16):
        super(MNISTNet, self).__init__()

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(1, dim_conv, kernel_size=3),  # -> 26 x 26
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),  # -> 12 x 12
            nn.Conv2d(dim_conv, dim_conv, kernel_size=3),  # -> 10 x 10
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),  # -> 4 x 4
            Rearrange(
                'b c h w -> b (h w) c'
            ),
            nn.Linear(dim_conv, dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.deq_atn = nn.Sequential(
            DEQFixedPoint(
                DEQMeanFieldAttention(
                    num_spins=num_spins+1,
                    dim=dim,
                    weight_sym_internal=True,
                    weight_sym_sites=False,
                    lin_response=True,
                ),
                anderson,
                solver_fwd_max_iter=40,
                solver_fwd_tol=1e-4,
                solver_bwd_max_iter=40,
                solver_bwd_tol=1e-4,
            ),
        )
        self.final = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.deq_atn(x)
        return self.final(x[:, 0, :])


def main():
    set_seeds(2666)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data.
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_data = datasets.MNIST(
        '.', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('.', train=False, transform=transform)

    train_loader = DataLoader(train_data,
                              batch_size=60,
                              num_workers=1,
                              shuffle=True)
    test_loader = DataLoader(test_data,
                             batch_size=512,
                             num_workers=1,
                             shuffle=False)
    # Model.
    model = MNISTNet().to(device)

    def get_num_params(model):
        """Count effective number of params."""
        return sum(
            [p.nelement() if '_weight' not in name
             else model.deq_atn[0].fun.count_params()
             for name, p in model.named_parameters()]
        )

    print(
        f'Initialized {model.__class__.__name__} ({get_num_params(model)} '
        f'params) on {device}.'
    )

    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    test(model, device, test_loader)
    for epoch in range(1, 20 + 1):
        train(model, device, train_loader, test_loader, optimizer, epoch)
        test(model, device, test_loader)


if __name__ == '__main__':
    main()
