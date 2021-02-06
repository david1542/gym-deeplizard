import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, image_width, image_height):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features=image_width * image_height * 3, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)
    
    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.out(t))
        return t
