import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import torch


class NetworkData():
    def __init__(self, acc, name):
        self.acc = acc
        self.name = name

    def get_name(self):
        return self.name

    def get_acc(self):
        return self.acc

majorLocator = MultipleLocator(20)
minorLocator = MultipleLocator(5)
fig = 0

data = []
for file in listdir('./checkpoints/'):
    if 'ckpt' in file:
        continue
    print('Loading nn from ' + file)
    network = torch.load('./checkpoints/{}'.format(file))
    data.append(NetworkData(network['acc_arr'], file.replace(".t7", "")))

for network in data:
    epochs = np.linspace(0, len(network.get_acc()), len(network.get_acc()))
    acc = network.get_acc()
    plt.figure(fig)
    fig += 1
    plt.xlabel('Epoch')
    plt.xlabel('Accuracy (%)')
    plt.title('Accuracy per epoch for ' + network.get_name())
    plt.gca().yaxis.set_major_locator(majorLocator)
    plt.gca().yaxis.set_minor_locator(minorLocator)
    plt.plot(epochs, acc)
    plt.savefig("./plots/" + network.get_name() + ".png")
    # plt.show()


plt.figure(fig)
fig += 1
for network in data:
    if 'best' in network.get_name():
        continue
    epochs = np.linspace(0, len(network.get_acc()), len(network.get_acc()))
    acc = network.get_acc()
    plt.plot(epochs, acc, label=network.get_name())
plt.xlabel('Epoch')
plt.xlabel('Accuracy (%)')
plt.gca().yaxis.set_major_locator(majorLocator)
plt.gca().yaxis.set_minor_locator(minorLocator)
plt.title('Accuracy per epoch for all networks')
plt.legend(loc='lower right')
plt.savefig("./plots/all.png")
# plt.show()


plt.figure(fig)
fig += 1
for network in data:
    if 'best' not in network.get_name():
        continue
    epochs = np.linspace(0, len(network.get_acc()), len(network.get_acc()))
    acc = network.get_acc()
    plt.plot(epochs, acc, label=network.get_name())
plt.xlabel('Epoch')
plt.xlabel('Accuracy (%)')
plt.gca().yaxis.set_major_locator(majorLocator)
plt.gca().yaxis.set_minor_locator(minorLocator)
plt.title('Accuracy per epoch for all networks')
plt.legend(loc='lower right')
plt.savefig("./plots/all.best.png")
# plt.show()


plt.figure(fig)
fig += 1
for network in data:
    if 'adam' not in network.get_name():
        continue
    epochs = np.linspace(0, len(network.get_acc()), len(network.get_acc()))
    acc = network.get_acc()
    plt.plot(epochs, acc, label=network.get_name())
plt.xlabel('Epoch')
plt.xlabel('Accuracy (%)')
plt.gca().yaxis.set_major_locator(majorLocator)
plt.gca().yaxis.set_minor_locator(minorLocator)
plt.title('Accuracy per epoch for all adams')
plt.legend(loc='lower right')
plt.savefig("./plots/all.adam.png")
# plt.show()
