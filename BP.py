import numpy as np
from sklearn.metrics import classification_report

def sigmoid(x):
    f = 1/(1+np.exp(-x))
    return f

def sigmoid_deri(f):
    f_deri = f*(1-f)
    return f_deri

#layernode存储了每层节点的个数,每层的节点数记为d，q，l
class inputLayer:
    def __init__(self, layernode):
        self.x = []
        self.weight_Vdq = []
        self.weight_Vdq.extend(2*np.random.random((layernode[0], layernode[1]))-1)
        self.weight_Vdq = np.mat(self.weight_Vdq)
        self.alpha = []

class hiddenLayer:
    def __init__(self, layernode):
        self.weight_Wql = []
        self.bias_gamma = []
        self.weight_Wql.extend(2*np.random.random((layernode[1], layernode[2]))-1)
        self.bias_gamma.extend(2*np.random.random(layernode[1])-1)
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

def fit(X, Y, learningrate=0.2, hiddenNumber=2, rounds=1000):
        inputNumber = X.shape[1]
        ouputNumber = Y.shape[1]            #看Y是几维的
        layernode = [inputNumber, hiddenNumber, ouputNumber]
        inputx = inputLayer(layernode)
        hiddenb = hiddenLayer(layernode)
        outputy = outputLayer(layernode)
        for round in range(rounds):
            for k in range(X.shape[0]):
                inputx.x = X[k]
                inputx.alpha = inputx.x * inputx.weight_Vdq         #(1,10)
                mat_b = inputx.alpha - hiddenb.bias_gamma
                hiddenb.b = np.zeros(mat_b.shape)
                for q in range(mat_b.shape[1]):
                    hiddenb.b[0, q] = (sigmoid(mat_b[0, q]))

                hiddenb.b = np.mat(hiddenb.b)
                hiddenb.beta = hiddenb.b * hiddenb.weight_Wql
                mat_y = hiddenb.beta - outputy.bias_theta
                outputy.y = np.zeros(mat_y.shape)
                for l in range(mat_y.shape[1]):
                    outputy.y[0,l] = (sigmoid(mat_y[0,l]))
                outputy.y = np.mat(outputy.y)


                Ek = 0
                for l in range(outputy.y.shape[1]-1):
                    Ek += (outputy.y[0,l] - Y[k,l]) * (outputy.y[0,l] - Y[k,l])
                Ek = 1/2*Ek             #????算出来干什么
                print(Ek)

                eta = learningrate
                gl = []             #计算gj
                for l in range(outputy.y.shape[1]):
                    gl.append(outputy.y[0,l]*(1 - outputy.y[0, l])*(Y[k, l] - outputy.y[0,l]))     #gl为列表

                eq = []             #计算eh
                for q in range(hiddenb.b.shape[1]):
                    WhjGj = 0
                    for l in range(len(gl)):
                        WhjGj += hiddenb.weight_Wql[q,l] * gl[l]
                    eq.append(hiddenb.b[0, q] * (1 - hiddenb.b[0, q]) * WhjGj)              #eq为列表

                for l in range(outputy.y.shape[1]):
                    outputy.bias_theta[0, l] += -eta*gl[l]

                for q in range(hiddenb.weight_Wql.shape[0]):
                    for l in range(outputy.y.shape[1]):
                        hiddenb.weight_Wql[q,l] += eta*gl[l]*hiddenb.b[0, q]

                for d in range(inputx.x.shape[1]):
                    for q in range(hiddenb.b.shape[1]):
                        inputx.weight_Vdq[d,q] += eta*eq[q]*inputx.x[0,d]

                for q in range(hiddenb.b.shape[1]):
                    hiddenb.bias_gamma += -eta*eq[q]

            print("round:%d\n"%round)
        return inputx,hiddenb,outputy


def predict(X_pre, inputx, hiddenb,outputy):
    x = X_pre
    alpha = x * inputx.weight_Vdq
    bq = []
    for q in range(alpha.shape[1]):
        bq.append(sigmoid(alpha[0,q] - hiddenb.bias_gamma[0,q]))
    bq = np.mat(bq)
    beta = bq * hiddenb.weight_Wql
    y = []
    for l in range(beta.shape[1]):
        y.append(round(sigmoid(beta[0,l] - outputy.bias_theta[0,l])))
    return y

def getData():
    data = []
    with open("data.txt") as f:
        for line in f.readlines():
            data.append([int(num) for num in line.strip('\n').split()])
    data = np.mat(data)

    number_train = int(data[-1, 0] * 0.8)
    np.random.shuffle(data)
    fit_Mat = data[0:number_train, :]
    pred_Mat = data[number_train:, :]
    # print(fit_Mat)
    # print(pred_Mat)
    X = fit_Mat[:, 2:5]
    Y = fit_Mat[:, 1].T

    pre_X = pred_Mat[:, 2:5]
    pre_Y = pred_Mat[:, 1].T

    Y_list = [Y[0, num] for num in range(Y.shape[1])]
    Y_setlist = sorted(list(set(Y_list)))
    Y_onehot = np.zeros((len(Y_list), len(Y_setlist)))
    for i in range(len(Y_list)):
        if Y_list[i] == Y_setlist[0]:
            Y_onehot[i, 0] = 1
        elif Y_list[i] == Y_setlist[1]:
            Y_onehot[i, 1] = 1
        else:
            Y_onehot[i, 2] = 1

    pre_Y_list = [pre_Y[0, num] for num in range(pre_Y.shape[1])]
    pre_Y_setlist = sorted(list(set(pre_Y_list)))
    pre_Y_onehot = np.zeros((len(pre_Y_list), len(pre_Y_setlist)))
    for i in range(len(pre_Y_list)):
        if pre_Y_list[i] == pre_Y_setlist[0]:
            pre_Y_onehot[i, 0] = 1
        elif pre_Y_list[i] == pre_Y_setlist[1]:
            pre_Y_onehot[i, 1] = 1
        else:
            pre_Y_onehot[i, 2] = 1
    return X, Y_onehot, pre_X, pre_Y_onehot,pre_Y

def onehot2num(Y_onehot):
    Y_num = []
    # for i in range(len(Y_onehot)):
    if Y_onehot[0] == 0:
        Y_num.append(0)
    elif Y_onehot[1] == 1:
        Y_num.append(1)
    else:
        Y_num.append(2)
    return Y_num


if __name__ == '__main__':
    acc = []
    for i in range(1):
        X, Y, pre_X, true_Y_onehot, true_Y = getData()
        print(Y)
        inputx, hiddenb, outputy = fit(X, Y, 0.01, 10, 10)
        pre_right = 0
        pre_wrong = 0
        preout_list = []
        for index in range(pre_X.shape[0]):
            preout = predict(pre_X[index], inputx, hiddenb, outputy)
            # if preout == [i for i in pre_Y[index]]:
            #     prejudge = 1
            #     pre_right += 1
            # else:
            #     prejudge = 0
            #     pre_wrong += 1
            # print("predict X:",pre_X[index],"value: ", preout , "judge:", prejudge)
            pre_Y = onehot2num(preout)
            preout_list.extend(pre_Y)
        # acc.append(pre_right/(pre_right + pre_wrong))
        print(classification_report([true_Y[0,i] for i in range(true_Y.shape[1])], preout_list))
    # sum = 0
    # for i in acc:
    #     sum += i
    # sum = sum/len(acc)
    # print("Acuuracy:", sum)