import torch
import torch.nn as nn
import torch.nn.functional as F


# define model
class attentionNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_1 = nn.Linear(17, 85)
        self.act_1 = nn.ReLU()
        
        self.hidden_2 = nn.Linear(85, 425)
        self.act_2 = nn.ReLU()
        
        self.hidden_3 = nn.Linear(425, 85)
        self.act_3 = nn.ReLU()
        
        self.hidden_4 = nn.Linear(85,17)
        self.act_4 = nn.ReLU()
        
        self.out = nn.Linear(17, 1)
        self.act_out = nn.Sigmoid()
        
    def forward(self, x):
        x = self.act_1(self.hidden_1(x))
        x = self.act_2(self.hidden_2(x))
        x = self.act_3(self.hidden_3(x))
        x = self.act_4(self.hidden_4(x))
        x = self.act_out(self.out(x))
        
        return x
    
    
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()