import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
average_epochs = []
def sigmoid(x):
    f = 1/(1+np.exp(-x))
    return f

#layernode存储了每层节点的个数,每层的节点数记为d，q，l
class inputLayer:
    def __init__(self, layernode):
        self.x = []
        self.weight_Vdq = []
        self.weight_Vdq.extend(2*np.random.random((layernode[0], layernode[1][0]))-1)
        self.weight_Vdq = np.mat(self.weight_Vdq)
        self.alpha = []

class hiddenLayer:
    def __init__(self, layernode, n):
        self.weight_Wql = []
        self.bias_gamma = []
        if n == len(layernode[1]) - 1:
            self.weight_Wql.extend(2 * np.random.random((layernode[1][n], layernode[2])) - 1)
        else:
            self.weight_Wql.extend(2 * np.random.random((layernode[1][n], layernode[1][n+1])) - 1)
        self.bias_gamma.extend(2*np.random.random(layernode[1][n])-1)
        self.weight_Wql = np.mat(self.weight_Wql)
        self.bias_gamma = np.mat(self.bias_gamma)
        self.b = []
        self.beta = []

class outputLayer:
    def __init__(self, layernode):
        self.bias_theta = []
        self.bias_theta.extend(2*np.random.random(layernode[2])-1)
        self.bias_theta = np.mat(self.bias_theta)
        self.y = []

# 输入测试集，标签，学习率，连续两次误差， 隐藏层数目及每层节点数（输入列表），循环次数，
def fit(X, Y, learningrate=0.2,Permitted_error = 0.001,hiddenNumber=[5], epochs=1000):
        inputNumber = X.shape[1]
        ouputNumber = Y.shape[1]            #看Y是几维的
        layernode = [inputNumber, hiddenNumber, ouputNumber]
        inputx = inputLayer(layernode)
        hiddenb = []
        for n in range(len(hiddennumber)):
            hiddenb.append(hiddenLayer(layernode, n))
        outputy = outputLayer(layernode)
        dj_all = []
        Per_e_count = 0     #记录连续两次小于误差
        for epoch in range(epochs):

            dj = []
            for k in range(X.shape[0]):
                inputx.x = X[k]             #选第k个测试集
                inputx.alpha = inputx.x * inputx.weight_Vdq         #矩阵的shape为(1,10)
                for n in range(len(hiddennumber)):
                    if n == 0:
                        mat_b = inputx.alpha - hiddenb[0].bias_gamma        #如果n==0，代表为第一层隐层，其输入来自inputx
                    else:                                                   #如果n!=0,计算隐层间的正向传递
                        mat_b = hiddenb[n-1].beta - hiddenb[n].bias_gamma
                    hiddenb[n].b = np.mat([sigmoid(mat_b[0, q]) for q in range(mat_b.shape[1])])
                    hiddenb[n].beta = hiddenb[n].b * hiddenb[n].weight_Wql

                mat_y = hiddenb[len(hiddennumber)-1].beta - outputy.bias_theta
                outputy.y = sigmoid(mat_y)

                Ek = 0     #误差计算

                for l in range(outputy.y.shape[1]-1):
                    Ek += (outputy.y[0,l] - Y[k,l]) * (outputy.y[0,l] - Y[k,l])
                Ek = 1/2*Ek

                eta = learningrate
                gl = np.multiply(np.multiply((1 - outputy.y),outputy.y),(Y[k] - outputy.y))             #计算gj
                outputy.bias_theta += -eta*gl
                eq0 = gl                #eq0代表上一层的梯度，用于衔接隐藏层
                hln = list(range(len(hiddennumber)))
                hln.reverse()
                for n in hln:
                    if n == len(hiddennumber) - 1:                    #表示从输出层到最后一层隐层的权值计算
                        hiddenb[n].weight_Wql += eta * np.multiply(gl, hiddenb[n].b.T)
                    else:
                        hiddenb[n].weight_Wql += eta * np.multiply(eq0, hiddenb[n].b.T)   #未完成
                    eq1 = np.multiply(np.multiply(hiddenb[n].b, (1 - hiddenb[n].b)), eq0 * hiddenb[n].weight_Wql.T)             #计算eh
                    if n == 0:
                        inputx.weight_Vdq += eta * np.multiply(eq1, inputx.x.T)
                        hiddenb[0].bias_gamma += -eta * eq1
                    else:
                        hiddenb[n].bias_gamma += -eta * eq1
                    eq0 = eq1
                dj.append(Ek)
            dj_sub = 0
            for num in dj:
                dj_sub += num
            dj_all.append(dj_sub / len(dj))
            if (len(dj_all)!=1 and abs(dj_all[-2] - dj_all[-1]) < Permitted_error):
                Per_e_count += 1
            if(Per_e_count == 2): break
        print("epochs :", len(dj_all))
        average_epochs.append(len(dj_all))
        plt.figure()
        plt.plot(list(range(len(dj_all))), dj_all)
        # plt.show()
        return inputx,hiddenb,outputy


def predict(X_pre, inputx, hiddenb,outputy):  #分别为输入测试集，输入层，隐藏层，输出层
    x = X_pre
    alpha = x * inputx.weight_Vdq
    for n in range(len(hiddenb)):       #隐藏层的循环计算
        bq = sigmoid(alpha - hiddenb[n].bias_gamma)
        beta = bq * hiddenb[n].weight_Wql
        alpha = beta            #作为下一次计算的输入
    y = []
    for l in range(beta.shape[1]):
        y.append(round(sigmoid(beta[0,l] - outputy.bias_theta[0,l])))
    return y

def getData():
    data = []
    with open("data.txt") as f:
        for line in f.readlines():
            data.append([int(num) for num in line.strip('\n').split()])
    data = np.mat(data).astype(float)

    # data[:, 2:] = MinMaxScaler().fit_transform(data[:, 2:])
            #以下操作为了保证测试集和训练集中各种类别的比例一致
    classrate = 0.75      #选取训练集样本比例为75%
    data_label0 = np.zeros((1, data.shape[1])); data_label1 = np.zeros((1, data.shape[1]));data_label2 = np.zeros((1, data.shape[1]))
    for row in range(data.shape[0]):
        if data[row, 1] == 0: data_label0 = np.vstack((data_label0, data[row]))
        if data[row, 1] == 1: data_label1 = np.vstack((data_label1, data[row]))
        if data[row, 1] == 2: data_label2 = np.vstack((data_label2, data[row]))
    data_label0=np.delete(data_label0, 1, 0); data_label1=np.delete(data_label1, 1, 0); data_label2=np.delete(data_label2, 1, 0)    #删除第一行的0向量
    number_train0 = int(data_label0.shape[0] * classrate)         #计算用于训练的样本数
    number_train1 = int(data_label1.shape[0] * classrate)
    number_train2 = int(data_label2.shape[0] * classrate)
    np.random.shuffle(data_label0);np.random.shuffle(data_label1);np.random.shuffle(data_label2)
    fit_Mat = data_label0[0:number_train0, :]
    fit_Mat = np.vstack((fit_Mat, data_label1[0:number_train1, :], data_label2[0:number_train2, :]))
    pred_Mat = data_label0[number_train0:, :]
    pred_Mat = np.vstack((pred_Mat, data_label1[number_train1:, :],data_label2[number_train2:, :]))
    fit_Mat[:, 2:] = MinMaxScaler().fit_transform(fit_Mat[:, 2:])
    pred_Mat[:, 2:] = MinMaxScaler().fit_transform(pred_Mat[:, 2:])
    np.random.shuffle(fit_Mat); np.random.shuffle(pred_Mat)
    X = fit_Mat[:, 2:5]
    Y = fit_Mat[:, 1].T

    pre_X = pred_Mat[:, 2:5]
    pre_Y = pred_Mat[:, 1].T
    pre_Y = [pre_Y[0, i] for i in range(pre_Y.shape[1])]

    Y_list = [Y[0, num] for num in range(Y.shape[1])]
    Y_setlist = sorted(list(set(Y_list)))
    Y_onehot = np.zeros((len(Y_list), len(Y_setlist)))      #将Y进行onehot编码
    for i in range(len(Y_list)):
        if Y_list[i] == Y_setlist[0]:
            Y_onehot[i, 0] = 1
        elif Y_list[i] == Y_setlist[1]:
            Y_onehot[i, 1] = 1
        else:
            Y_onehot[i, 2] = 1
    return X, Y_onehot, pre_X, pre_Y

def onehot2num(Y_onehot):
    Y_num = []
    if Y_onehot[0] == 1:
        Y_num.append(0)
    elif Y_onehot[1] == 1:
        Y_num.append(1)
    else:
        Y_num.append(2)
    return Y_num

if __name__ == '__main__':
    acc = []
    for i in range(5):      #设置测试次数
        X, Y, pre_X, true_Y = getData()
        hiddennumber = [10]           #在这里输入每层的节点数
                                 # 输入测试集，标签，学习率，连续两次误差， 隐藏层数目及每层节点数（输入列表），循环次数，
        inputx, hiddenb, outputy = fit(X,     Y,  0.01,   1e-7,          hiddennumber,           5000)
        pre_right = 0
        pre_wrong = 0
        preout_list = []
        for index in range(pre_X.shape[0]):
            preout = predict(pre_X[index], inputx, hiddenb, outputy)
            pre_Y = onehot2num(preout)
            preout_list.extend(pre_Y)
        # acc.append(pre_right/(pre_right + pre_wrong))
        target_names = ['class 0', 'class 1', 'class 2']
        print(classification_report(true_Y, preout_list, target_names=target_names))
        print("average_epochs:", average_epochs / len(average_epochs))
