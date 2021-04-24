from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output
from deep_implicit_attention.deq import DEQFixedPoint
from deep_implicit_attention.modules import (
    FeedForward,
    GeneralizedIsingGaussianAdaTAP,
)
from deep_implicit_attention.solvers import anderson
import numpy as np

from einops.layers.torch import Rearrange


class Net(nn.Module):
    def __init__(self, image_size, patch_size, dim=10, channels=1):
        super(Net, self).__init__()

        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        # assert pool in {
        #     "cls",
        #     "mean",
        # }, "pool type must be either cls (cls token) or mean (mean pooling)"

        num_spins, dim = (num_patches + 1, dim)

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            nn.Linear(patch_dim, dim),
            FeedForward(dim, dim, dropout=0.1),
        )
        # self.conv1 = nn.Conv2d(1, dim, 3, 1)
        # self.conv2 = nn.Conv2d(dim, dim, 3, 1)

        self.final = nn.Linear(dim, 10)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.norm = nn.LayerNorm(dim)
        self.deq_atn = nn.Sequential(
            DEQFixedPoint(
                GeneralizedIsingGaussianAdaTAP(
                    num_spins=num_spins,
                    dim=dim,
                    weight_init_std=1.0
                    / np.sqrt(num_spins * dim ** 2),  # np.sqrt(num_spins * dim),
                    weight_symmetric=True,
                    lin_response=True,
                ),
                anderson,
                solver_fwd_max_iter=30,
                solver_fwd_tol=1e-4,
                solver_bwd_max_iter=30,
                solver_bwd_tol=1e-4,
            ),
            # FeedForward(dim, dim, dropout=0.1),
            # DEQFixedPoint(
            #     IsingGaussianAdaTAP(
            #         num_spins=num_spins,
            #         dim=dim,
            #         weights_init_std=0.5
            #         / np.sqrt(num_spins),  # np.sqrt(num_spins * dim),
            #         weights_symmetric=True,
            #         solver_max_iter=5,
            #         solver_tol=1e-3,
            #         lin_response_correction=True,
            #     ),
            #     anderson,
            #     solver_fwd_max_iter=15,
            #     solver_fwd_tol=1e-4,
            #     solver_bwd_max_iter=15,
            #     solver_bwd_tol=1e-4,
            # ),
            # FeedForward(dim, dim, dropout=0.1),
            # DEQFixedPoint(
            #     IsingGaussianAdaTAP(
            #         num_spins=num_spins,
            #         dim=dim,
            #         weights_init_std=0.5
            #         / np.sqrt(num_spins),  # np.sqrt(num_spins * dim),
            #         weights_symmetric=True,
            #         solver_max_iter=5,
            #         solver_tol=1e-3,
            #         lin_response_correction=True,
            #     ),
            #     anderson,
            #     solver_fwd_max_iter=15,
            #     solver_fwd_tol=1e-4,
            #     solver_bwd_max_iter=15,
            #     solver_bwd_tol=1e-4,
            # ),
            # FeedForward(dim, dim, dropout=0.1),
        )

    def forward(self, x):
        # print(self.deq_attn.fun.weights)
        # print(x.shape)
        # x = self.conv1(x)
        # # x = F.relu(x)
        # # x = self.conv2(x)
        # # x = F.relu(x)
        # x = F.avg_pool2d(x, 4)
        # # print(x.shape)
        # x = x.permute([0, 2, 3, 1]).reshape(x.shape[0], -1, 16)
        # print(torch.zeros_like(x)[:, 0, :].shape)
        # print(x.shape)
        x = self.to_patch_embedding(x)
        # print(x.shape)
        # comp_dim = 65  # - x.shape[1]
        cls_token = self.cls_token.repeat(x.size(0), 1, 1)
        # comp_tokens = torch.zeros((1, comp_dim, 16)).repeat(x.size(0), 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        # print(x.shape, comp_tokens.shape, cls_token.shape)
        # x = x / np.sqrt(x.shape[-1])
        # print(x)
        # print(x.shape)
        # print(x.shape)
        x = self.deq_atn(x)
        # print(x.shape)
        # x = torch.mean(x[:, :, :], dim=1)
        # print(x.shape)
        output = F.log_softmax(self.final(x[:, 0, :]), dim=-1)
        # print(output)
        return output


def train(args, model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(output)
        loss = F.nll_loss(output, target)
        loss.backward()
        # print(model.deq_atn[0].fun._weights.grad, model.deq_atn[1].net[0].weight.grad)
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    # scheduler.get_last_lr()[0],
                )
            )
            if args.dry_run:
                break
        # if batch_idx % (10 * log_interval) == 0:
        #     import matplotlib.pyplot as plt

        #     plt.imshow(model.deq_atn[0].fun._weight.clone().detach().numpy())
        #     plt.colorbar()
        #     plt.show()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net(28, 7).to(device)
    print("params", sum([p.nelement() for p in model.parameters()]))
    print("params", [k for k, v in model.named_parameters()])
    optimizer = optim.Adam(model.parameters(), lr=args.lr,)

    # scheduler = StepLR(optimizer, step_size=1.0, gamma=0.1)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, args.log_interval)
        test(model, device, test_loader)
        # scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
