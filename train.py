from __future__ import print_function
from __future__ import division
from Model import *
from CXRDataset import CXRDataset
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import numpy as np
import time
import os

data_dir = "/scratch/liaoi/images_all_0.9_224"
# data_path = {'train': "./Train_Label.csv", 'test': "Test_Label.csv"}
data_path = {'train': "train_all.csv", 'test': "test_all.csv"}
save_dir = "./savedModels"


def loadData(batch_size):
    trans = transforms.Compose([
                                # transforms.Grayscale(),
                                # transforms.Resize(224), # 224, 299
                                # transforms.RandomCrop(224),
                                # transforms.CenterCrop(224), #bad
                                # transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
    image_datasets = {x: CXRDataset(data_path[x], data_dir, transform=trans) for x in ['train', 'test']}
    dataloders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                  for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    print('Training data: {}\nTest data: {}'.format(dataset_sizes['train'], dataset_sizes['test']))

    class_names = image_datasets['train'].classes
    return dataloders, dataset_sizes, class_names


def weighted_BCELoss(output, target, weights=None):
    output = output.clamp(min=1e-5, max=1 - 1e-5)
    if weights is not None:
        assert len(weights) == 2

        loss = -weights[0] * (target * torch.log(output)) - weights[1] * ((1 - target) * torch.log(1 - output))
    else:
        loss = -target * torch.log(output) - (1 - target) * torch.log(1 - output)

    return torch.sum(loss)


def train_model(model, optimizer, num_epochs=10, batch_size=2):
    batch_size = 16
    since = time.time()
    dataloders, dataset_sizes, class_names = loadData(batch_size)
    best_model_wts = model.state_dict()
    best_auc = []
    best_auc_ave = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            outputList = []
            labelList = []
            logLoss = 0
            # Iterate over data.
            for idx, data in enumerate(dataloders[phase]):
                # get the inputs
                inputs = data['image']
                labels = data['label']

                # calculate weight for loss
                P = 0
                N = 0
                for label in labels:
                    for v in label:
                        if int(v) == 1:
                            P += 1
                        else:
                            N += 1
                if P != 0 and N != 0:
                    BP = (P + N) / P
                    BN = (P + N) / N
                    weights = [BP, BN]
                    weights = torch.FloatTensor(weights).cuda()
                else:
                    weights = None

                weights = None
                # wrap them in Variable
                inputs = inputs.cuda()
                labels = labels.cuda()

                if phase == 'train':
                    inputs, labels = Variable(inputs, volatile=False), Variable(labels, volatile=False)
                else:
                    inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                out_data = outputs.data

                # if isinstance(outputs, tuple):
                #     loss = sum((criterion(o,labels) for o in outputs))
                # else:
                #     loss = criterion(outputs, labels)

                loss = weighted_BCELoss(outputs, labels, weights=weights)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                labels = labels.data.cpu().numpy()
                out_data = out_data.cpu().numpy()
                for i in range(out_data.shape[0]):
                    outputList.append(out_data[i].tolist())
                    labelList.append(labels[i].tolist())

                logLoss += loss.data[0]
                if idx % 100 == 0 and idx != 0:
                    try:
                        iterAuc = roc_auc_score(np.array(labelList[-100 * batch_size:]),
                                                np.array(outputList[-100 * batch_size:]))
                    except:
                        iterAuc = -1
                    print('{} {:.2f}% Loss: {:.4f} AUC: {:.4f}'.format(phase, 100 * idx / len(dataloders[phase]),
                                                                       logLoss / (100 * batch_size), iterAuc))
                    logLoss = 0

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_auc_ave = roc_auc_score(np.array(labelList), np.array(outputList))
            epoch_auc = roc_auc_score(np.array(labelList), np.array(outputList), average=None)

            print('{} Loss: {:.4f} AUC: {:.4f}'.format(
                phase, epoch_loss, epoch_auc_ave, epoch_auc))
            print()
            for i, c in enumerate(class_names):
                print('{}: {:.4f} '.format(c, epoch_auc[i]))
            print()

            # deep copy the model
            if phase == 'test' and epoch_auc_ave > best_auc_ave:
                best_auc = epoch_auc
                best_auc_ave = epoch_auc_ave
                best_model_wts = model.state_dict()
                # saveInfo(model)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val AUC: {:4f}'.format(best_auc_ave))
    print()
    for i, c in enumerate(class_names):
        print('{}: {:.4f} '.format(c, epoch_auc[i]))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def saveInfo(model):
    # save model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, "resnet50.pth"))


if __name__ == '__main__':
    model = ResNet50_mod()
    model.cuda()
    optimizer = optim.Adam([
        {'params': model.model_ft.parameters()},
        {'params': model.prediction.parameters()},

        # {'params': model.conv1.parameters()},
        # {'params': model.conv2.parameters()},
        # {'params': model.conv3.parameters()},
        # {'params': model.conv4.parameters()},
        
        # # {'params': model.conv5.parameters()}, #
        # # {'params': model.conv6.parameters()}, #
        # # {'params': model.conv7.parameters()}, #
        # # {'params': model.conv8.parameters()}, #

        # {'params': model.fc1.parameters()},
        # # {'params': model.fc_mid.parameters()}, #
        # {'params': model.fc2.parameters()},      
        ], lr=1e-5)

    model = train_model(model, optimizer, num_epochs=9)
