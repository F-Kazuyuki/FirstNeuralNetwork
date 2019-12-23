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


def sigmoid(x): #シグモイド関数の定義
    return 1 / (1 + np.exp(-x))

def sigmoid_dash(x): #シグモイド関数の微分の定義
    return sigmoid(x) * (1 - sigmoid(x))
    
def softmax(x): #ソフトマックス関数の定義
    x_max = np.max(x)
    exp_a = np.exp(x - x_max)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a

train_img, train_label, test_img, test_label = mnist.load_mnist()

#　学習データの準備


learning_rate = 0.1    #学習係数の設定
number = 10000    #学習回数の設定
batch_size = 100    #
data_size = train_img.shape[0]

#　ネットワークの構造の設定　全体は3層構造
input = train_img.shape[1] #25入力    
hidden = 100 #中間層は20
output = train_label.shape[1] #5出力

#　荷重としきい値の修正量を格納する変数　すべて0で初期化
delta_W1 = np.zeros((hidden, input))    
delta_W2 = np.zeros((output, hidden))
delta_b1 = np.zeros(hidden)
delta_b2 = np.zeros(output)

W1 = 0.1 * np.random.randn(hidden, input)    #荷重を??から??の範囲で乱数により初期化
W2 = 0.1 * np.random.randn(output, hidden)
b1 = np.zeros(hidden)    #しきい値をすべて0で初期化
b2 = np.zeros(output)

for i in range(number):
    
    batch_mask = np.random.choice(data_size, batch_size)
    data = train_img[batch_mask]
    teach = train_label[batch_mask]
    
    for j in range(batch_size):
        
        #　順方向の計算
        X1 = np.dot(W1, data[j]) + b1    
        Z1 = sigmoid(X1)            #２層目の出力の計算
    
        X2 = np.dot(W2, Z1) + b2     #出力の計算
        Z2 = softmax(X2)
        
        #　逆方向の計算
        delta_out = (Z2- teach[j])   #誤差の計算
        
        delta_W2 += np.array([Z1 * tmp for tmp in delta_out])    #２層目の荷重の修正量の計算
        delta_b2 += delta_out    #２層目のしきい値の修正量

        delta_hidden = np.dot(W2.T, delta_out) * sigmoid_dash(X1)    #誤差の逆伝搬
        
        delta_W1 += np.array([data[j] * tmp for tmp in delta_hidden])    #１層目の荷重の修正量の計算
        delta_b1 += delta_hidden    #１層目のしきい値の修正量
        
    #　荷重としきい値の更新
    W1 -= learning_rate * delta_W1    
    W2 -= learning_rate * delta_W2
    b1 -= learning_rate * delta_b1
    b2 -= learning_rate * delta_b2

    #　修正量をすべて0にリセット       
    delta_W1 = np.zeros((hidden, input))    
    delta_W2 = np.zeros((output, hidden))
    delta_b1 = np.zeros(hidden)
    delta_b2 = np.zeros(output)
    
   # print(i)
    
#　結果の表示
print("\n")
t_data_size = test_img.shape[0]
test_count = 1000   #テスト回数
count = 0
batch_mask = np.random.choice(t_data_size, test_count)
t_data = test_img[batch_mask]
t_teach = test_label[batch_mask]
for i in range(test_count):
    X1 = np.dot(W1, t_data[i]) + b1
    Z1 = sigmoid(X1)

    X2 = np.dot(W2, Z1) + b2
    Z2 = softmax(X2)
    
    y = np.argmax(Z2)
    t = t_teach[i]
    if(y == t):
        count += 1
        
acc = count / test_count * 100
print(str(acc) + "%")
        
        
"""
    print(Z2)
    accuracy = -1* np.dot(teach[i], np.log(Z2))    #??精度??
    #print(accuracy)
    """
