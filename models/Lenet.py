import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from time import strftime

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model_name='LeNet'
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def load(self,path):
        '''
        加载指定路径的模型
        :param path: 路径
        :return:
        '''
        self.load_state_dict(torch.load(path))

    def save(self,name=None,epoch=-1):
        if name is None:
            prefix = './ckps/' + self.model_name + '_'
            name = strftime(prefix + '%m_%d_%H_%M_ep{}.pth'.format(epoch))
        torch.save(self.state_dict(), name)
        print('model saved at {}'.format(name))