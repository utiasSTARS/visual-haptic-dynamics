import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,1)
    def forward(self, x):
        x = self.fc1(x)
        return x

class NetWrap():
    def __init__(self, net):
        self.net = net

def test_network_pass_by_ref():
    net = Net()
    net_wrapped = NetWrap(net)

    for param in net.parameters():
        print("before", param.data)
        param.data = param.data + 2.0

    for param in net.parameters():
        print("after", param.data)

    for param in net_wrapped.net.parameters():
        print("object holding net after", param.data)
    
if __name__ == "__main__":
    test_network_pass_by_ref()