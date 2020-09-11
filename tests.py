from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import Adam
import time
import matplotlib.pyplot as plt

from losses import margin_loss, reconstruction_loss
from models.capsnet import CapsNet
from models.funcs import put_mask


def test_cifar10():
    CUDA = torch.cuda.is_available()
    net = CapsNet(8 * 8 * 32, [3, 32, 32])
    if CUDA:
        net.cuda()
    print (net)
    print ("# parameters: ", sum(param.numel() for param in net.parameters()))

    transform = transforms.Compose(
        [transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False)
    optimizer = Adam(net.parameters())

    n_epochs = 30
    print_every = 200 if CUDA else 2
    display_every = 10

    for epoch in range(n_epochs):
        train_acc = 0.
        time_start = time.time()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            masked_inputs = put_mask(inputs)
            labels_one_hot = torch.eye(10).index_select(dim=0, index=labels)
            if CUDA:
                masked_inputs, inputs, labels_one_hot, labels = masked_inputs.cuda(), inputs.cuda(), labels_one_hot.cuda(), labels.cuda()
            optimizer.zero_grad()
            class_probs, recons = net(masked_inputs, labels)
            acc = torch.mean((labels == torch.max(class_probs, -1)[1]).double())
            train_acc += acc.data.item()
            loss = (margin_loss(class_probs, labels_one_hot) + 0.0005 * reconstruction_loss(recons, inputs))
            loss.backward()
            optimizer.step()
            if (i+1) % print_every == 0:
                print('[epoch {}/{}, batch {}] train_loss: {:.5f}, train_acc: {:.5f}'.format(epoch + 1, n_epochs, i + 1, loss.data.item(), acc.data.item()))
        test_acc = 0.
        for j, data in enumerate(testloader, 0):
            inputs, labels = data
            img = inputs[0]
            print(img.shape)
            masked_inputs = put_mask(inputs)
            labels_one_hot = torch.eye(10).index_select(dim=0, index=labels)
            if CUDA:
                masked_inputs, inputs, labels_one_hot, labels = masked_inputs.cuda(), inputs.cuda(), labels_one_hot.cuda(), labels.cuda()
            class_probs, recons = net(masked_inputs)
            if (j+1) % display_every == 0:
                display(inputs[0], masked_inputs[0], recons[0])
            acc = torch.mean((labels == torch.max(class_probs, -1)[1]).double())
            test_acc += acc.data.item()
        print('[epoch {}/{} done in {:.2f}s] train_acc: {:.5f} test_acc: {:.5f}'.format(epoch + 1, n_epochs, (time.time() - time_start), train_acc/(i + 1), test_acc/(j + 1)))


def test_fashion_mnist():
    CUDA = torch.cuda.is_available()
    net = CapsNet(6 * 6 * 32)
    if CUDA:
        net.cuda()
    print (net)
    print ("# parameters: ", sum(param.numel() for param in net.parameters()))

    transform = transforms.Compose(
        [transforms.ToTensor()])

    trainset = torchvision.datasets.FashionMNIST(root='./data/fashion', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True)

    testset = torchvision.datasets.FashionMNIST(root='./data/fashion', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False)
    optimizer = Adam(net.parameters())

    n_epochs = 30
    print_every = 200 if CUDA else 2

    for epoch in range(n_epochs):
        train_acc = 0.
        time_start = time.time()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            labels_one_hot = torch.eye(10).index_select(dim=0, index=labels)
            inputs, labels_one_hot, labels = Variable(inputs), Variable(labels_one_hot), Variable(labels)
            if CUDA:
                inputs, labels_one_hot, labels = inputs.cuda(), labels_one_hot.cuda(), labels.cuda()
            optimizer.zero_grad()
            class_probs, recons = net(inputs, labels)
            acc = torch.mean((labels == torch.max(class_probs, -1)[1]).double())
            train_acc += acc.data.item()
            loss = (margin_loss(class_probs, labels_one_hot) + 0.0005 * reconstruction_loss(recons, inputs))
            loss.backward()
            optimizer.step()
            if (i+1) % print_every == 0:
                print('[epoch {}/{}, batch {}] train_loss: {:.5f}, train_acc: {:.5f}'.format(epoch + 1, n_epochs, i + 1, loss.data.item(), acc.data.item()))
        test_acc = 0.
        for j, data in enumerate(testloader, 0):
            inputs, labels = data
            labels_one_hot = torch.eye(10).index_select(dim=0, index=labels)
            inputs, labels_one_hot, labels = Variable(inputs), Variable(labels_one_hot), Variable(labels)
            if CUDA:
                inputs, labels_one_hot, labels = inputs.cuda(), labels_one_hot.cuda(), labels.cuda()
            class_probs, recons = net(inputs)
            acc = torch.mean((labels == torch.max(class_probs, -1)[1]).double())
            test_acc += acc.data.item()
        print('[epoch {}/{} done in {:.2f}s] train_acc: {:.5f} test_acc: {:.5f}'.format(epoch + 1, n_epochs, (time.time() - time_start), train_acc/(i + 1), test_acc/(j + 1)))


def test_mnist():
    CUDA = torch.cuda.is_available()
    net = CapsNet(6 * 6 * 32)
    if CUDA:
        net.cuda()
    print (net)
    print ("# parameters: ", sum(param.numel() for param in net.parameters()))

    transform = transforms.Compose(
        [transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data/mnist', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False)
    optimizer = Adam(net.parameters())

    n_epochs = 30
    print_every = 200 if CUDA else 2
    display_every = 100

    for epoch in range(n_epochs):
        train_acc = 0.
        time_start = time.time()
        # for i, data in enumerate(trainloader, 0):
        #     inputs, labels = data
        #     masked_inputs = put_mask(inputs)
        #     labels_one_hot = torch.eye(10).index_select(dim=0, index=labels)
        #     if CUDA:
        #         inputs, masked_inputs, labels_one_hot, labels = inputs.cuda(), masked_inputs.cuda(), labels_one_hot.cuda(), labels.cuda()
        #     optimizer.zero_grad()
        #     class_probs, recons = net(masked_inputs, labels)
        #     acc = torch.mean((labels == torch.max(class_probs, -1)[1]).double())
        #     train_acc += acc.data.item()
        #     loss = (margin_loss(class_probs, labels_one_hot) + 0.0005 * reconstruction_loss(recons, inputs))
        #     loss.backward()
        #     optimizer.step()
        #     if (i+1) % print_every == 0:
        #         print('[epoch {}/{}, batch {}] train_loss: {:.5f}, train_acc: {:.5f}'.format(epoch + 1, n_epochs, i + 1, loss.data.item(), acc.data.item()))
        test_acc = 0.
        for j, data in enumerate(testloader, 0):
            inputs, labels = data
            masked_inputs = put_mask(inputs)
            labels_one_hot = torch.eye(10).index_select(dim=0, index=labels)
            if CUDA:
                inputs, masked_inputs, labels_one_hot, labels = inputs.cuda(), masked_inputs.cuda(), labels_one_hot.cuda(), labels.cuda()
            class_probs, recons = net(masked_inputs)
            if (j) % display_every == 0:
                display(inputs[0].cpu(), masked_inputs[0].cpu(), recons[0].cpu().detach())
            acc = torch.mean((labels == torch.max(class_probs, -1)[1]).double())
            test_acc += acc.data.item()
        print('[epoch {}/{} done in {:.2f}s] train_acc: {:.5f} test_acc: {:.5f}'.format(epoch + 1, n_epochs, (time.time() - time_start), train_acc/(i + 1), test_acc/(j + 1)))


def display(img, masked_img, recons_img):
    imgs = [img, masked_img, recons_img]
    fig = plt.figure(figsize=(10, 10))
    for i in range(3):
        fig.add_subplot(1, 3, i+1)
        if imgs[i].shape[0] == 1:
            plt.imshow(imgs[i][0])
        else:
            plt.imshow(imgs[i])
    plt.show()

