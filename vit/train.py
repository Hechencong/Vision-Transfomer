import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms
import vit


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#对训练集数据的预处理
trans_train = transforms.Compose(
    [transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

#对验证集数据进行预处理
trans_valid = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])


def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct/total


def train(net, train_data, valid_data, num_epochs, optimizer, criterion):

    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        for im, label in train_data:
            im = im.to(device)
            label = label.to(device)
            # forward
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_acc(output, label)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                im = im.to(device)
                label = label.to(device)
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
                epoch_str = (
                        "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                        % (epoch, train_loss / len(train_data),
                           train_acc / len(train_data), valid_loss / len(valid_data),
                           valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                    (epoch, train_loss / len(train_data),
                    train_acc / len(train_data)))

        print(epoch_str)


def main():

    trainset = torchvision.datasets.CIFAR10(root="../Dataset/CIFAR10", train=True, download=False, transform=trans_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    validset = torchvision.datasets.CIFAR10(root="../Dataset/CIFAR10", train=False,  download=False, transform=trans_valid)
    validloader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True, num_workers=2)

    # dataiter = iter(trainloader)
    # images, labels = dataiter.__next__()

    model = vit.ViT(image_size=224, patch_size=16, num_classes=10,
                    d_in=768, n_layers=2, n_heads=2, mlp_dim=128)

    model = model.to(device=device)
    # weights_dict = torch.load('vit_base_patch16_224.pth', map_location=device)
    # model.load_state_dict(weights_dict, strict=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3, momentum=0.9)

    train(model, trainloader, validloader, 400, optimizer, criterion)


if __name__ == '__main__':
    main()


# Epoch 99. Train Loss: 1.211110, Train Acc: 0.567375, Valid Loss: 0.965371, Valid Acc: 0.663416,