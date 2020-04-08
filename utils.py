import sys, os
sys.path.append(os.pardir)
import numpy as np

def create_weight_tensor(lst, std=0.01):
    """
    lst[0]を入力数,
    lst[i]を隠れ層第i層目のパーセプトロンの個数とすると
    Weight_public_nnクラスのコンストラクタに渡せるWをランダムに生成する。
    """
    num = np.size(lst)-1;

    z = []
    for i in range(0,num):
        z.append(std * np.random.randn(lst[i]+1, lst[i+1]))
    
    return z
    

def create_weights_tensor(lst):
    return create_weight_tensor(lst)


class Sigmoid:
    def __init__(self, x):
        self.x = x
        self.result = 1 / (1+np.exp(-x))
        
    def get(self):
        return self.result

    def backout(self, back_vector):
        return back_vector[0,1:] * (1-self.result) * self.result

    
    
class Relu:
    def __init__(self, x):
        self.x = x
        if x > 0:
            self.result = x
        else:
            self.result = 0

    def get(self):
        return self.result

    def backout(self, back_vector):
        if self.x > 0:
            return back_vector
        else:
            return 0

class Softmax_lasthiddenlayeronly:
    def __init__(self, x):
        self.x = x
        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x)
        self.result = exp_x / sum_exp_x

    def get(self):
        return self.result

    def backout(self, t_vector):
        return self.result - t_vector

class Identity_lasthiddenlayeronly:
    def __init__(self, x):
        self.x = x
        self.result = x

    def get(self):
        return self.result

    def backout(self, t_vector):
        return self.result - t_vector        
        
    


class Triangle:
    """
    三角形の三辺をランダムに生成する。
    また、それが鋭角三角形か鈍角三角形かのラベルもつける。
    直角三角形は鈍角三角形とみなす
    """
    def __init__(self):
        self.data = None
    
    def _generate_once(self):
        sides = (10*np.random.rand(3)).astype(np.int8);
        sorted_sides = np.sort(sides)[::-1]
        if (sorted_sides[0] ** 2 >= sorted_sides[1] ** 2 + sorted_sides[2] ** 2):
            sides = np.append(sides, 0)
            sides = np.append(sides, 1)
            return sides
        else:
            sides = np.append(sides, 1)
            sides = np.append(sides, 0)
            return sides

    def generate(self, cnt):
        if (self.data is None):
            self.data = self._generate_once()
        else:
            self.data = np.vstack((self.data, self._generate_once()))
            
        for i in range(1, cnt):
            self.data = np.vstack((self.data, self._generate_once()))
        return self

    def reset(self):
        self.data = None
        return self

    def get(self):
        return self.data

    def save(self, path):
        np.savetxt(path, self.data, fmt='%d')
        return self        
    
