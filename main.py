import os
import torchvision as tv
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
import torch
import numpy as np
from time import strftime
import models
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image
import random
from sklearn.metrics import accuracy_score

from models.Lenet import LeNet
from models.MobileNet import MobileNet
from models.Resnet import ResNet18,ResNet34,ResNet50,ResNet101,ResNet152
# from models.Mobilenetv2 import MobileNetV2
# from models.Shufflenet import ShuffleNet
# from models.Shufflenetv2 import ShuffleNetV2
from models.SqueezeNet import SqueezeNet
from models.VGG import VGG,VGG11,VGG13,VGG16,VGG19
from models.Darknet import Darknet53
from models.Inception import InceptionV4

from attack.PixelAttack import attack_all
from attack.Visualize import plot_image, plot_model, save_image
from differential_evolution import differential_evolution
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage


DOWNLOAD_CIFAR10=False 
show=ToPILImage() 


class Config:
 	def __init__(self):

	    #self.model_path = None 
	    self.model_path = 'ckps/SqueezeNet_01_01_07_52_ep-1.pth'
	    self.model = 'SqueezeNet'
	    self.use_gpu = torch.cuda.is_available() 
	    '''
	    MobileNet,ShuffleNetV2,ShuffleNet,MobileNetV2,SqueezeNet
	    VGG11,VGG13,VGG16,VGG19,
	    ResNet18,ResNet34,ResNet50,ResNet101,ResNet152
	    '''
	    self.attack_num = 2000  
	    self.train_epoch = 60 
	    self.batch_size = 100 
	    self.print_freq = 500    
  

####Train###########################################################################################

def accuracy(y_pred, y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true,y_pred_cls)


def train():

    global DOWNLOAD_CIFAR10
    opt=Config()

    model = getattr(models.SqueezeNet,opt.model)() #后面的括号是模型实例化，否则无法to.device('cuda')

    if opt.model_path:
        model.load(opt.model_path)
    model.to(torch.device("cuda"))


    criterion = nn.CrossEntropyLoss() 
    #optimizer = Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    ##optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
    metric_func = accuracy


    if not (os.path.exists('./data/cifar/')) or not os.listdir('./data/cifar/'):
        DOWNLOAD_CIFAR10=True

    transform = tv.transforms.Compose([
        #tv.transforms.Resize([299,299], interpolation=2), ##models.Inception If model=Inception then need this instruction
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(), 
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_data = tv.datasets.CIFAR10(
        root='./data/cifar/',
        train=True,
        transform=transform,
        download=DOWNLOAD_CIFAR10
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4
                              )

    loss_list=[]
    acc_list=[]

    for epoch in range(opt.train_epoch):
        for ii,(data,label) in tqdm(enumerate(train_loader)):
            input = data.to(torch.device("cuda"))
            target = label.to(torch.device("cuda"))

            optimizer.zero_grad()
            predictions = model(input)
            loss = criterion(predictions,target)
            metric = metric_func(predictions.data.cpu(),target.data.cpu())
            loss.backward()
            optimizer.step()

            if (ii+1)%opt.print_freq ==0:
                print('loss:%.2f'%loss.cpu().data.numpy())

            loss_list.append(loss)
            acc_list.append(metric)

    model.save()

    x = range(0,int(opt.train_epoch*50000/opt.batch_size))
    y = loss_list
    plt.cla()
    plt.title('Train loss vs. epoch', fontsize=20)
    plt.plot(x, y, '.-')
    plt.xlabel('train_epoch*50000/batch_size', fontsize=20)
    plt.ylabel('Train_Loss', fontsize=20)
    plt.grid()
    plt.savefig("./Trainimg/Train_Loss.png")
    plt.show()

    y1 = acc_list
    plt.cla()
    plt.title('Train Acc vs. epoch', fontsize=20)
    plt.plot(x, y1, '.-')
    plt.xlabel('train_epoch*50000/batch_size', fontsize=20)
    plt.ylabel('Train_Acc', fontsize=20)
    plt.grid()
    plt.savefig("./Trainimg/Train_Acc.png")
    plt.show()


@torch.no_grad()
def test_acc():
    opt=Config()
    global DOWNLOAD_CIFAR10

    if not (os.path.exists('./data/cifar/')) or not os.listdir('./data/cifar/'):
        DOWNLOAD_CIFAR10=True

    model = getattr(models.SqueezeNet, opt.model)()
    if opt.model_path:
        model.load(opt.model_path)
    model.to(torch.device("cuda"))

    transform = tv.transforms.Compose([
           tv.transforms.ToTensor(),
           tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
        ])

    test_data = tv.datasets.CIFAR10(
        root='./data/cifar/',
        train=False,
        transform=transform,
        download=DOWNLOAD_CIFAR10
    )

    test_loader = DataLoader(test_data,batch_size=1000,shuffle=True,num_workers=1)

    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    dataiter = iter(test_loader)
    test_x, test_y = next(dataiter)
    test_x = test_x.to(torch.device("cuda"))

    test_score = model(test_x)
    accuracy = np.mean((torch.argmax(test_score.to('cpu'),1)==test_y).numpy())
    print('test accuracy:%.2f' % accuracy)
    return accuracy


def predict():
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    opt=Config()
    model = getattr(models.SqueezeNet,opt.model)() #后面的括号是模型实例化，否则无法to.device('cuda')
    if opt.model_path:
        model.load(opt.model_path)
    model.to(torch.device("cuda"))
    model.eval()
    img = Image.open('saveImage/bird-o1.png').convert('RGB')
    img_2=tv.transforms.functional.resize(img, [32,32], interpolation=2)

    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    img2 = transform(img_2)
    with torch.no_grad():
        input= Variable(img2.unsqueeze(0)).cuda()
    confidence = F.softmax(model(input), dim=1).data.cpu().numpy()[0]
    predicted_class_num = np.argmax(confidence)
    predicted_class = class_names[predicted_class_num]
    print(predicted_class)


def predict_all():
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    opt=Config()

    model = getattr(models.SqueezeNet,opt.model)()
    if opt.model_path:
        model.load(opt.model_path)
    model.to(torch.device("cuda"))
    model.eval()
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    acc_runtime = 0
    fileList = []
    files = os.listdir("./saveImage")
    for f in files:
        if(os.path.isfile('./saveImage/' + f)):
            filepath, tmpfilename = os.path.split('./saveImage/' + f)
            shotname, extension = os.path.splitext(tmpfilename)
            fileList.append('./saveImage/' + f)
            shotnamelen = len(shotname)
            usite = shotname.find('_')
            sshotname=shotname[0:usite]

            hsite = sshotname.find('-')
            ori_class = sshotname[0:hsite]
            predic_class = sshotname[hsite+1:len(sshotname)]
            
            img2 = Image.open('./saveImage/' + f).convert('RGB')
            img2 = transform(img2)
            with torch.no_grad():
                    input= Variable(img2.unsqueeze(0)).cuda()
            confidence = F.softmax(model(input), dim=1).data.cpu().numpy()[0]
            predicted_class_num = np.argmax(confidence)
            predicted_class_runtime = class_names[predicted_class_num]

            if (predicted_class_runtime == predic_class):
                acc_runtime += 1
    
    fileListlen= len(fileList)
    print('Accuracy-Runtime is %.4f' %(acc_runtime/fileListlen))

####Attack###########################################################################################

def attack_model_pixel():

    accuracy = test_acc()
    opt=Config()
    global DOWNLOAD_CIFAR10
    if not (os.path.exists('./data/cifar/')) or not os.listdir('./data/cifar/'):
        DOWNLOAD_CIFAR10=True

    model = getattr(models.SqueezeNet, opt.model)()
    if opt.model_path:
        model.load(opt.model_path)
    model.to(torch.device("cuda")).eval()

    transform = tv.transforms.Compose([
           tv.transforms.ToTensor(),
           tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
       ])

    test_data = tv.datasets.CIFAR10(
        root='./data/cifar/',
        train=False,
        transform=transform,
        download=DOWNLOAD_CIFAR10
    )

    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    
    success_rate = attack_all(class_names, model, test_loader, pixels=1, targeted=False, maxiter=75, popsize=400, verbose=False, device=torch.device("cuda"), sample=opt.attack_num) ##pixels=(1, 3, 5)
    string = 'model name:{} | accuracy:{} | success rate:{}| time: {}\n'.format(opt.model,accuracy,success_rate, strftime('%m_%d_%H_%M_%S'))
    open('log.txt','a').write(string)


if __name__ == '__main__':
    #train()
    #test_acc()
    #predict()
    attack_model_pixel()
    predict_all()
