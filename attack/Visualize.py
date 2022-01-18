import os
import cv2
import torchvision as tv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt
import pickle
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from PIL import Image
import string



def save_image(img, attack_image, label, predicted_class, oriorattac):
    if oriorattac:
        labelori= label + '-'+ 'ori'
        fnum, maxnum, fwrt = file_num(filename=labelori)
        if (fwrt):
            filenameo = './saveImage/' + label + '-'+ 'ori' + '_' + str(1) + '.png'
        else:
            maxnum = maxnum +1
            filenameo = './saveImage/' + label + '-'+ 'ori' + '_' + str(maxnum) + '.png'
        img = img.permute(1, 2, 0)
        image_data = (np.asarray(img))*255
        cv2.imwrite(filenameo, image_data)
    else:
        labelpre= label + '-'+ predicted_class
        fnum, maxnum, fwrt = file_num(filename=labelpre)
        if (fwrt):
            filenamea = './saveImage/' + label + '-'+ predicted_class + '_' + str(1) + '.png'
        else:
            maxnum = maxnum +1
            filenamea = './saveImage/' + label + '-'+ predicted_class + '_' + str(maxnum) + '.png'

        attack_image = attack_image.permute(1, 2, 0)
        attack_data = (np.asarray(attack_image))*255
        cv2.imwrite(filenamea, attack_data)


def file_num(filename):
    fileList = []
    numList = []
    fsum = 0
    maxnum = 0
    files = os.listdir("./saveImage")
    for f in files:
        if(os.path.isfile('./saveImage/' + f)):
            filepath, tmpfilename = os.path.split('./saveImage/' + f)
            shotname, extension = os.path.splitext(tmpfilename)
            if (str.rfind(shotname,filename)!=-1):
                fileList.append(shotname)
                fsum +=1
            elif(str.rfind(shotname,filename)==-1):
                continue

    if (fsum ==0):
        fwrt=True

    else:
        fwrt=False
        for i in range(len(fileList)):
            hsite = fileList[i].find('_')
            numstr = fileList[i][hsite+1:len(fileList[i])]
            numList.append(int(numstr))
        maxnum = max(numList)

    return fsum, maxnum, fwrt
    


def plot_image(images, label_true=None, class_names=None, label_pred=None):
    images = images/2+0.5     # unnormalize
    npimg = images.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    if label_true is not None and class_names is not None:
        labels_true_name = class_names[label_true]
        if label_pred is None:
            xlabel = "True: " + labels_true_name
        else:
            labels_pred_name = class_names[label_pred]
            xlabel = "True: " + labels_true_name + "\nPredicted: " + labels_pred_name
        plt.xlabel(xlabel)
    plt.xticks([])  
    plt.yticks([])
    plt.show()  



def plot_model(model_details):

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(range(1, len(model_details.history['acc']) + 1), model_details.history['acc'])
    axs[0].plot(range(1, len(model_details.history['val_acc']) + 1), model_details.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_details.history['acc']) + 1), len(model_details.history['acc']) / 10)
    axs[0].legend(['train', 'val'], loc='best')

    axs[1].plot(range(1, len(model_details.history['loss']) + 1), model_details.history['loss'])
    axs[1].plot(range(1, len(model_details.history['val_loss']) + 1), model_details.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_details.history['loss']) + 1), len(model_details.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


def visualize_attack(df, class_names):
    _, (x_test, _) = cifar10.load_data()

    results = df[df.success].sample(9)

    z = zip(results.perturbation, x_test[results.image])
    images = np.array([perturb_image(p, img)[0]
                       for p, img in z])

    labels_true = np.array(results.true)
    labels_pred = np.array(results.predicted)
    titles = np.array(results.model)

    # Plot the first 9 images.
    plot_images(images=images,
                labels_true=labels_true,
                class_names=class_names,
                labels_pred=labels_pred,
                titles=titles)


def attack_stats(df, models, network_stats):
    stats = []
    for model in models:
        val_accuracy = np.array(network_stats[network_stats.name == model.name].accuracy)[0]
        m_result = df[df.model == model.name]
        pixels = list(set(m_result.pixels))

        for pixel in pixels:
            p_result = m_result[m_result.pixels == pixel]
            success_rate = len(p_result[p_result.success]) / len(p_result)
            stats.append([model.name, val_accuracy, pixel, success_rate])

    return pd.DataFrame(stats, columns=['model', 'accuracy', 'pixels', 'attack_success_rate'])


def evaluate_models(models, x_test, y_test):
    correct_imgs = []
    network_stats = []
    for model in models:
        print('Evaluating', model.name)

        predictions = model.predict(x_test)

        correct = [[model.name, i, label, np.max(pred), pred]
                   for i, (label, pred)
                   in enumerate(zip(y_test[:, 0], predictions))
                   if label == np.argmax(pred)]
        accuracy = len(correct) / len(x_test)

        correct_imgs += correct
        network_stats += [[model.name, accuracy, model.count_params()]]
    return network_stats, correct_imgs


def load_results():
    with open('networks/results/untargeted_results.pkl', 'rb') as file:
        untargeted = pickle.load(file)
    with open('networks/results/targeted_results.pkl', 'rb') as file:
        targeted = pickle.load(file)
    return untargeted, targeted


def checkpoint(results, targeted=False):
    filename = 'targeted' if targeted else 'untargeted'

    with open('networks/results/' + filename + '_results.pkl', 'wb') as file:
        pickle.dump(results, file)


def download_from_url(url, dst):
    r = requests.get(url, stream=True)
    with open(dst, 'wb') as f:
        for data in tqdm(r.iter_content(), unit='B', unit_scale=True):
            f.write(data)
