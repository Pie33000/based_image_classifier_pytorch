import torch

class ClassifierNet(nn.Module):
    
    def __init__(self):
        super(ClassifierNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(18, 36, 5)
        
        self.fc1 = nn.Linear(36*11*11, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 4)
        
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 36*11*11)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc5(F.relu(self.fc4(x)))
        return x

def train():
    model = ClassifierNet()
    return model
