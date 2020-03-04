import torch
from torch import nn
from randdata import numbers, labels
import torch.utils.data as Data
import tqdm


class MF(nn.Module):
    def __init__(self):
        super().__init__()
        self.sq = nn.Sequential(
            nn.Linear(8, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 68),
            nn.BatchNorm1d(68),
            nn.ReLU(),
            nn.Linear(68, 8),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x / 100
        x -= 0.5
        return self.sq(x)


if __name__ == '__main__':
    net = MF()
    net.load_state_dict(torch.load('net.pkl'))
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.008)
    ds = Data.dataset.TensorDataset(torch.tensor(numbers, dtype=torch.float), torch.tensor(labels, dtype=torch.int64))
    loader = Data.DataLoader(ds, 500, True, num_workers=4,drop_last=True)

    test = [75, 34, 16, 4, 48, 65, 84, 12]
    test = torch.tensor(test, dtype=torch.float).unsqueeze(0)
    for e in tqdm.tqdm(range(30)):
        for step, (bx, by) in enumerate(loader):
            opt.zero_grad()
            _y = net(bx)
            los = loss_fn(_y, by)
            los.backward()
            opt.step()
            if step % 100 == 0:
                print(los)
                net.eval()
                test = net(test)
                print(torch.argmax(test))
                net.train()

    net.eval()
    test = net(test)
    print(torch.argmax(test))
    torch.save(net.state_dict(),'net.pkl')