# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 18:13:01 2019

@author: Funato
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:48:14 2019

@author: Funato Kazuyuki
"""

#このプログラムは自由に使用、改変していただけますが、それにより生じた、またそれを利用したことにより生じたいかなる損害についても責任を負いません。
import numpy as np
import mnist
import matplotlib.pyplot as plt

train_img, train_label, test_img, test_label = mnist.load_mnist()

#　学習データの準備
train_img = np.asarray(train_img)
train_label = np.asarray(train_label)
test_img = np.asarray(test_img)
test_label = np.asarray(test_label)


learning_rate = 0.1    #学習係数の設定
number = 10000    #学習回数の設定
batch_size = 100    #
data_size = train_img.shape[0]
t_data_size = test_img.shape[0]
test_count = 1000   #テスト回数

#　ネットワークの構造の設定　全体は3層構造
input_size = train_img.shape[1] #25入力    
hidden_size = 100 #中間層は20
output_size = train_label.shape[1] #5出力


test_acc_list = []
miss_list = []


def sigmoid(x): #シグモイド関数の定義
    return 1 / (1 + np.exp(-x))

def sigmoid_dash(x): #シグモイド関数の微分の定義
    return sigmoid(x) * (1 - sigmoid(x))
    
def softmax(x): #ソフトマックス関数の定義
    x = x.T
    x_max = np.max(x, axis = 0)
    exp_a = np.exp(x - x_max)
    sum_exp_a = np.sum(exp_a, axis = 0)
    return (exp_a / sum_exp_a).T


class Two_layer_network:
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input = input_size
        self.hidden = hidden_size
        self.output = output_size
        self.W1 = 0.01 * np.random.randn(hidden_size, input_size)    #荷重を??から??の範囲で乱数により初期化
        self.W2 = 0.01 * np.random.randn(output_size, hidden_size)
        self.b1 = np.zeros(hidden_size)    #しきい値をすべて0で初期化
        self.b2 = np.zeros(output_size)
        
        self.learning_rate= 0.01
        
        self.act1 = sigmoid
        self.act2 = softmax
        
        self.X1 = 0
        self.Z1 = 0
        self.X2 = 0
        self.Z2 = 0
        
    def forward(self, x):
        X1 = (self.W1 @ x.T).T + self.b1    
        Z1 = self.act1(X1)            #２層目の出力の計算
    #print(Z1.shape)
        X2 = (np.dot(self.W2, Z1.T)).T + self.b2     #出力の計算
        Z2 = self.act2(X2)
        
        self.X1 = X1
        self.X2 = X2
        self.Z1 = Z1
        self.Z2 = Z2
        
        return Z2
        

    
    def backprop(self, x, t):
        Z1 = self.Z1
        Z2 = self.Z2
        X1 = self.X1
        
        delta_out = (Z2- t) / Z2.shape[0]  #誤差の計算
    
        delta_W2 = (delta_out.T @ Z1)  #２層目の荷重の修正量の計算
        delta_b2 = np.sum(delta_out.T, axis = 1)    #２層目のしきい値の修正量
    
    
        delta_hidden = np.dot(delta_out, self.W2) * sigmoid_dash(X1)    #誤差の逆伝搬
    
        delta_W1 = delta_hidden.T @ x   #１層目の荷重の修正量の計算
        delta_b1 = np.sum(delta_hidden.T, axis = 1)    #１層目のしきい値の修正量
        
        self.W1 -= self.learning_rate * delta_W1 
        self.W2 -= self.learning_rate * delta_W2
        self.b1 -= self.learning_rate * delta_b1
        self.b2 -= self.learning_rate * delta_b2
  
    
        



"""
#　荷重としきい値の修正量を格納する変数　すべて0で初期化
delta_W1 = cp.zeros((hidden, input))    
delta_W2 = cp.zeros((output, hidden))
delta_b1 = cp.zeros(hidden)
delta_b2 = cp.zeros(output)
"""
network = Two_layer_network(input_size, hidden_size, output_size)


for i in range(number + 1):
    
    batch_mask = np.random.choice(data_size, batch_size)
    data = train_img[batch_mask]
    teach = train_label[batch_mask]
    
    network.forward(data)
    network.backprop(data, teach)
        
    #　荷重としきい値の更新
    
    #print(i)
    
    

    
    if i % 100 == 0:
        
        batch_mask = np.random.choice(t_data_size, test_count)
        t_data = test_img[batch_mask]
        t_teach = test_label[batch_mask]
        
        out = network.forward(t_data)

        y = np.argmax(out, axis = 1)
        
        acc = np.sum(y == t_teach) / test_count * 100
        test_acc_list.append(acc)
        print(str(i) + " : " + str(acc) + "%")
        
        if number == i:
            miss_list = [y != t_teach]
            miss_img = t_data[miss_list]
            print("test")
            for img in miss_img:
    
                img = img.reshape((28, 28))

                plt.imshow(img)
                plt.gray()
                plt.show()
    
    
#　結果の表示

x_len = np.arange(len(test_acc_list))
plt.plot(x_len, test_acc_list)
plt.show()



        
        
"""
    print(Z2)
    accuracy = -1* np.dot(teach[i], np.log(Z2))    #??精度??
    #print(accuracy)
    """
