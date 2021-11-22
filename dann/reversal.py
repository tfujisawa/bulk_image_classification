import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

#Gradient reversal layer for DANN (Ganin et al. 201x)
@tf.custom_gradient
def grad_reverse(x):
    result = x
    def custom_grad(dy):
        return -dy
    return result, custom_grad

class ReversalLayer(Layer):
    def __init__(self):
        super(ReversalLayer, self).__init__()
    def call(self, x):
        return grad_reverse(x)

#Gradient reversal layer with a lambda parameter
@tf.custom_gradient
def grad_reverse_lambd(x, lambd):
    result = x
    def custom_grad_l(dy):
        return  lambd * -dy, 0.
    return result, custom_grad_l

class ReversalLayerLambd(Layer):
    def __init__(self, lambd):
        super(ReversalLayerLambd, self).__init__()
        self.lambd = lambd

    def call(self, x):
        return grad_reverse_lambd(x, self.lambd)

#Gradient reversal layer with a variable lamabda parameter
class VariableLambda(Callback):
    def __init__(self, model_lambd, max_epoch):
        super().__init__()
        self.model_lambd = model_lambd
        # self.current_epoch = current_epoch
        self.max_epoch = max_epoch
        self.gamma = 5

        self.p = 1/self.max_epoch

        self.p_hist = [self.p]
        self.l_hist = [0.]
    def on_epoch_begin(self, epoch, logs = None):
        # K.set_value(self.current_epoch, epoch)
        # self.current_epoch.append(epoch)
        self.p = epoch/self.max_epoch
        self.lambd = 2/(1+np.exp(-self.gamma*self.p)) - 1
        self.p_hist.append(self.p)
        self.l_hist.append(self.lambd)
        # print (self.model_lambd)
        K.set_value(self.model_lambd, self.lambd)

class VariableLearningRate:
    def __init__(self, lr=0.01, max_epoch=200):
        self.lr = lr
        self.max_epoch = max_epoch
        self.alpha = 10
        self.beta = 0.75
    def __call__(self, epoch, lr):
        self.p = epoch/self.max_epoch
        return self.lr/(1+self.alpha*self.p)**self.beta

class TrainingHistory(Callback):
    def __init__(self, metric, outfile="./log.txt"):
        super().__init__()
        self.metric = metric
        self.outfile = outfile
        self.record = []

    def write(self):
        print ("\t".join(self.metric))
        for rec in self.record:
            print ("\t".join([str(m) for m in rec]))

    def write_file(self):
        with open(self.outfile, "w") as f:
            f.write("epoch\t"+"\t".join(self.metric)+"\n")
            for i, rec in enumerate(self.record):
                f.write(str(i) +"\t" +"\t".join([str(m) for m in rec])+"\n")

    def on_epoch_end(self, epoch, logs = None):
        self.record.append([logs[m] for m in self.metric])

class OutputObserver(Callback):
    def __init__(self, target_layers, test_x, test_y, step=10, outfile="./out.txt"):
        super().__init__()
        self.target_layers = target_layers
        self.test_x = test_x
        self.test_y = test_y
        self.outfile = outfile
        self.step = step
        self.record = []

    def on_epoch_end(self, epoch, logs = None):
        if epoch%self.step == 0:
            x = self.model.get_layer(self.target_layers[0])(self.test_x)
            for layr in self.target_layers[1:]:
                x = self.model.get_layer(layr)(x)
            self.record.append(x.numpy())

    def write(self):
        for i, rec in enumerate(self.record):
            for x, y in zip(rec, self.test_y):
                print (f"{i*10}," + f"{y}," + ",".join(map(str, x)))

    def write_file(self):
        with open(self.outfile, "w") as f:
            for i, rec in enumerate(self.record):
                for x, y in zip(rec, self.test_y):
                    f.write(f"{i*10}," + f"{y}," + ",".join(map(str, x))+"\n")

