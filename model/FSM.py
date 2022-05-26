import torch.nn as nn
import torch


# Feature Select via Sorting
class FeatureSelect(nn.Module):
    def __init__(self, in_dim=84, ratio=0.5):
        """
        Feature Select via Sorting

        Args:
            in_dim: the number of dimensions of raw features
            ratio: the portion of selected features
        """
        super(FeatureSelect, self).__init__()
        self.in_dim = in_dim
        self.select_dim = int(in_dim * ratio)

    def forward(self, x):
        """
        Args:
            x: feature discrepancy of shape [batch_size, in_dim]
        Returns:
            v: selecting vector of shape [batch_size, in_dim]
        """
        # x: [batch_size, in_dim]
        idx = torch.argsort(x, dim=1)
        # idx: [batch_size, in_dim]
        idx[idx < self.select_dim] = 1
        idx[idx >= self.select_dim] = 0
        v = idx
        # v: [batch_size, in_dim]
        return v


# Feature Selection Module
class FSM(nn.Module):
    def __init__(self, in_dim=84, ratio=0.5):
        """
        `Feature Selection Module`

        Args:
            in_dim: the number of dimensions of raw features
            ratio: the portion of selected features
        """
        super(FSM, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = int(in_dim * ratio)
        self.fc1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.in_dim)

    def forward(self, x):
        """
        Args:
            x: feature discrepancy of shape [batch_size, in_dim]
        Returns:
            v: selecting vector of shape [batch_size, in_dim]
        """
        v = self.fc1(x)
        v = torch.relu(v)
        v = self.fc2(v)
        v = torch.relu(v)
        return v


def test_fs():
    fs = FeatureSelect(in_dim=100, ratio=0.5)
    print(fs)
    distance = torch.randn(10, 100)
    out = fs(distance)
    print('out:', out.size())


def test_fsm():
    fsm = FSM(in_dim=100, ratio=0.5)
    print(fsm)
    distance = torch.randn(10, 100)
    out = fsm(distance)
    print('out:', out.size())


# test model
if __name__ == '__main__':
    # test_fs()
    test_fsm()
