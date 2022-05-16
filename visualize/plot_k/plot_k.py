import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

colors = ['r', 'g', 'b', 'c']
run_to_label = {
    0: '$k=0$',
    1: '$k=4$',
    2: '$k=1$',
    3: '$k=4$ only'
}

fig, ax = plt.subplots(nrows=1, ncols=2, num='k_moment')
# fig.suptitle('k_moment', fontsize=20)

# source
for i in range(4):
    filename='mnist_to_mnistm_source_' + str(i) + '.csv'
    data = pd.read_csv(filename)
    arr = data['Value'].to_numpy()
    # plt.subplot(1, 2, 1)
    # plt.title('Source')
    # plt.xlabel('Step')
    # plt.ylabel('Accuracy')
    # plt.plot(arr, c=colors[i], label=run_to_label[i])
    # plt.legend()
    ax[0].set_title('Source')
    ax[0].set_xlabel('Step')
    ax[0].set_ylabel('Accuracy')
    ax[0].plot(arr, c=colors[i], label=run_to_label[i])
    ax[0].legend()

# target
for i in range(4):
    filename='mnist_to_mnistm_target_' + str(i) + '.csv'
    data = pd.read_csv(filename)
    arr = data['Value'].to_numpy()
    # plt.subplot(1, 2, 2)
    # plt.title('Target')
    # plt.xlabel('Step')
    # plt.ylabel('Accuracy')
    # plt.plot(arr, c=colors[i], label=run_to_label[i])
    # plt.legend()
    ax[1].set_title('Target')
    ax[1].set_xlabel('Step')
    ax[1].set_ylabel('Accuracy')
    ax[1].plot(arr, c=colors[i], label=run_to_label[i])
    ax[1].legend()

plt.show()
