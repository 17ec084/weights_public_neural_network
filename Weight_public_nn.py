import sys, os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
import layers

class Weight_public_nn:
    """
    Weight_public_nnは重みを外部から操作可能なニューラルネットワークである。
    学習時は
    for train_datum in train_data:
        nn.learn(train_datum)
        (重みを外部から操作する処理)
    とする。
    
    """
    def __init__(self, W):
        """
        Wは各層の重み行列(ndarray)の配列(array)
        pythonではフィールドも外部からアクセス可である。
        """
        self.W = W;
        self._input_layer = None
        """
        入力層はbackout()を持つが、伝える相手はいない。
        即ち入力層のbackout()の返却値は何の役にも立たない。
        ただし入力層の重み配列を更新する意味は持つ。
        """
        self._hidden_layers = None
        self._output_layer = None
        """
        重みや活性化関数は手前の層のものなので、
        出力層は重みも活性化関数ももたない。入力をそのまま出力するだけ。
        """

    def predict(self, input_vector):
        """
        入力ベクトルから、出力ベクトルを計算で求める。
        """
        self._hidden_layers = [None]
        """#detail
        print('入力')
        print(input_vector)
        print('入力層の重み行列')
        print(self.W[0])
        """#detail
        self._input_layer = layers.Input_layer(input_vector, self.W[0])
        input_vector = self._input_layer.output()
        """#detail
        print('入力層の出力')
        print(input_vector)
        """#detail
        hidden_cnt = len(self.W)-1 #隠れ層の個数
        
        for i in range(1, hidden_cnt):
            """#detail
            print('第'+str(i)+'隠れ層の入力')
            print(input_vector)
            """#detail
            self._hidden_layers = np.append(self._hidden_layers, layers.Hidden_layer(input_vector, self.W[i]))
            input_vector = self._hidden_layers[i].output()
            """#detail
            print('第'+str(i)+'隠れ層の重み行列')
            print(self.W[i])
            print('第'+str(i)+'隠れ層の出力')
            print(input_vector)
            """#detail

        """#detail
        print('第'+str(hidden_cnt)+'(最終)隠れ層の入力')
        print(input_vector)
        """#detail
        self._hidden_layers = np.append(self._hidden_layers, layers.Last_hidden_layer_classification(input_vector, self.W[hidden_cnt]))
        input_vector = self._hidden_layers[hidden_cnt].output()
        """#detail
        print('第'+str(hidden_cnt)+'隠れ層の重み行列')
        print(self.W[hidden_cnt])
        print('第'+str(hidden_cnt)+'隠れ層の出力')
        print(input_vector)
        """#detail
        self._output_layer = layers.Output_layer(input_vector, None)
        """#detail
        print('出力層の入力')
        print(input_vector)
        """#detail
        output_vector = self._output_layer.output()
        """#detail
        print('出力')
        print(output_vector)
        """#detail
        return output_vector

    def learn(self, train_datum):
        """
        訓練データ1組を受け取り、学習を行う(つまりパラメータを更新する)
        """
        hidden_cnt = len(self.W)-1 #隠れ層の個数
        input_vector = train_datum[0:(np.shape(self.W[0])[0])-1]
        t_vector = train_datum[(np.shape(self.W[0])[0])-1:]

        self.predict(input_vector)
        """#detail
        print('学習を開始します。')
        print('ラベル')
        print(t_vector)
        """#detail
        back_vector = self._output_layer.backout(t_vector);
        """#detail
        print('出力層の誤差逆出力')
        print(back_vector)
        """#detail

        for i in range(hidden_cnt, 0, -1):
            back_vector = self._hidden_layers[i].backout(back_vector) #重みもbackoutが更新
            """#detail
            print('第'+str(i)+'層の誤差逆出力')
            print(back_vector)
            """#detail
            self.W[i] = self._hidden_layers[i].get_new_W_matrix()
        back_vector = self._input_layer.backout(back_vector)
        """#detail
        print('入力層の誤差逆出力')
        print(back_vector)
        """#detail
        self.W[0] = self._input_layer.get_new_W_matrix()

        
