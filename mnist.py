# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 18:04:16 2019

@author: Funato
"""

import pickle
import numpy as np

def load_mnist():
    save_file = 'mnist.pkl'

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
        
    train_label = np.array(dataset['train_label'])
    train_img = np.array(dataset['train_img'])
    test_label = np.array(dataset['test_label'])
    test_img = np.array(dataset['test_img'])
    train_label = np.identity(10)[train_label]
    
    
    train_img = train_img.astype(np.float32)
    train_img /= 255.0
    
    test_img = test_img.astype(np.float32)
    test_img /= 255.0
        
    return(train_img, train_label, test_img, test_label)


train_img, train_label, test_img, test_label = load_mnist()

"""
data_size = train_img.shape[0]
batch_size = 5
print(data_size)

select = np.random.choice(data_size, batch_size)
test_img = np.array(train_img[select])
test_label = np.array(train_label[select])
print(test_label)



    

import matplotlib.pyplot as plt

for i in range(5):
    img = test_img[i].reshape((28, 28))

    plt.imshow(img)
    plt.gray()
    plt.show()
    """
    