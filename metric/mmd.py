import torch
import torch.nn as nn


class MMDLoss(nn.Module):
    def __init__(self):
        """
        Maximum Mean Discrepancy Loss
        """
        super(MMDLoss, self).__init__()
        self.eps = 1e-8

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        # TODO: implement mmd loss
        delta = f1 - f2
        # delta: [batch_size, num_features]
        loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
        return loss


if __name__ == '__main__':
    data1 = torch.rand(10, 100)
    data2 = torch.randn(10, 100)
    # data: [10, 100]
    criterion = MMDLoss()
    loss = criterion(data1, data2)
    print(loss.item())
