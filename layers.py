import sys, os
sys.path.append(os.pardir)
import numpy as np
import utils as act
from abc import ABCMeta, abstractmethod #抽象クラス https://qiita.com/kaneshin/items/269bc5f156d86f8a91c4


class Layer:
    def __init__(self, input_vector, W_matrix):
        self.input_vector = input_vector
        self.W_matrix = W_matrix
        self.activate_function = None
        self.learning_rate = 0.1
    
    def output(self):
        x = np.hstack([1,self.input_vector])
        """#detail
        print('層への入力ベクトル')
        print(x)
        print('層の重み行列')
        print(self.W_matrix)
        print('それらの積')
        print(np.dot(x, self.W_matrix))
        """#detail

        self.activate_function = act.Sigmoid(np.dot(x, self.W_matrix))
       #self.activate_function = act.Relu(np.dot(x, self.W_matrix))
        """#detail
        print('それらのシグモイド')
        print(self.activate_function.get())
        """#detail
        return self.activate_function.get()   

    def get_new_W_matrix(self):
        """
        機能は単にself.W_matrixの返却に過ぎない。
        更新(backout呼び出し)前にこのメソッドを呼べば更新前の重み配列が得られるので注意。
        名前にnewとあるのは、呼ぶときにわかりやすくするためである。
        """
        return self.W_matrix

    def backout(self, back_vector):
        
        activate_backout = np.array([self.activate_function.backout(back_vector)])
        
        old_W_matrix = self.W_matrix
        x_T = np.array([np.hstack([1,self.input_vector])]).T
        dW_matrix = np.dot(x_T, activate_backout)
        dW_matrix = np.squeeze(dW_matrix)
        self.W_matrix -= self.learning_rate * dW_matrix
        return np.dot(activate_backout, old_W_matrix.T)
        

class Input_layer(Layer):
    pass

        
class Hidden_layer(Layer):
    pass


class Last_hidden_layer_abstract(Hidden_layer):
    def __init__(self, input_vector, W_matrix):
        super().__init__(input_vector, W_matrix)
    #    self.input_vector = input_vector
    #    self.activate_function = None
    #    self.W_matrix = W_matrix
    #    self.learning_rate = sup.learning_rate


    def output(self):
        return None


#    def backout(self, t_vector):
#        return self.activate_function.backout(t_vector)

class Last_hidden_layer_classification(Last_hidden_layer_abstract):

    def output(self):
        x = np.hstack([1,self.input_vector])
        self.activate_function = act.Softmax_lasthiddenlayeronly(np.dot(x, self.W_matrix))

        return self.activate_function.get()
    
    
class Last_hidden_layer_regression(Last_hidden_layer_abstract):

    def output(self):    
        x = np.hstack([1,self.input_vector])
        self.activate_function = act.Identity_lasthiddenlayeronly(np.dot(x, self.W_matrix))

        return self.activate_function.get()


class Output_layer(Layer):
    
    def output(self):    
        return self.input_vector

    def backout(self, t_vector):
        return t_vector
    
