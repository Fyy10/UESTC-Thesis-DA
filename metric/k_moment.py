import torch
import torch.nn as nn


class KMomentLoss(nn.Module):
    def __init__(self, k: int=4):
        """
        k moment distance, where `k` represents the highest order of moment.
        """
        super(KMomentLoss, self).__init__()
        self.eps = 1e-8
        self.k = k

    def euclidean_dist(self, d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        # d: [num_features,]
        return (((d1 - d2) ** 2).sum() + self.eps).sqrt()


    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        # f1: [N, num_features]
        # f2: [N, num_features]
        loss = 0.0
        for order in range(1, self.k + 1):
            f1_k = (f1 ** order).mean(dim=0)
            f2_k = (f2 ** order).mean(dim=0)
            # f1_k: [num_features,]
            # f2_k: [num_features,]
            loss += self.euclidean_dist(f1_k, f2_k)
        return loss


if __name__ == '__main__':
    data1 = torch.randn(10, 100)
    data2 = torch.rand(10, 100)
    # data: [10, 100]
    criterion = KMomentLoss(k=4)
    loss = criterion(data1, data2)
    print(loss.item())
