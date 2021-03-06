import torch
import torch.nn as nn
import torch.nn.functional as F
from time import strftime

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

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


class VGG11(nn.Module):
    def __init__(self, vgg_name='VGG11'):
        super(VGG11, self).__init__()
        self.model_name = 'VGG11'
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

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




class VGG13(nn.Module):
    def __init__(self, vgg_name='VGG13'):
        super(VGG13, self).__init__()
        self.model_name = 'VGG13'
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

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



class VGG16(nn.Module):
    def __init__(self, vgg_name='VGG16'):
        super(VGG16, self).__init__()
        self.model_name = 'VGG16'
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

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



class VGG19(nn.Module):
    def __init__(self, vgg_name='VGG19'):
        super(VGG19, self).__init__()
        self.model_name = 'VGG19'
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

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