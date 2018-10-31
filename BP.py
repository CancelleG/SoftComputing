import numpy as np

def sigmoid(x):
    f = 1/(1+np.exp(x))
    return f(x)

def sigmoid_deri(f):
    f_deri = f*(1-f)
    return f_deri

#layernode存储了每层节点的个数,每层的节点数记为d，q，l
class inputLayer:
    def __init__(self, layernode):
        self.weight_Vdh = []
        self.weight_Vdh.append(2*np.random.random((layernode[0], layernode[1]))-1)

class hiddenLayer:
    def __init__(self, layernode):
        self.weight_Wql = []
        self.bias_gamma = []
        self.weight_Wql.append(2*np.random.random((layernode[1], layernode[2]))-1)
        self.bias_gamma.append(2*np.random.random(layernode[1])-1)
        self.alpha = []

class outputLayer:
    def __init__(self, layernode):
        self.bias_theta = []
        self.bias_theta.append(2*np.random.random(layernode[3])-1)
        self.beta = []

def train(X, Y, learningrate=0.1, rounds=1000):
        inputNumber = X.shape[1]
        ouputNumber = Y.shape[1]
        hiddenNumber = 10       #the number of hiddenlayer node
        layernode = [inputNumber,hiddenNumber, ouputNumber]
        inputx = inputLayer(layernode)
        hiddenb = hiddenLayer(layernode)
        outputy = outputLayer(layernode)
        for round in range(rounds):
            



if __name__ == '__main__':
    layernode = [2, 2, 2]
    input = inputLayer(layernode)
    # print(input.weight_Vdh)

