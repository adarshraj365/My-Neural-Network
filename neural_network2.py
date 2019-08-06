import numpy as np
import matplotlib.pyplot as plt
np.random.seed(100)

class layer:
    # class to represent a layer (hidden or output) in our neural network
    def __init__(self,n_input,n_neuron,activation):
        """ input size  represent input layer or a previous layer
        n_neuron  is number of neuron in this layer """
        self.weight=  np.random.rand(n_input,n_neuron)
        self.activation = activation
        self.bias =  np.random.rand(n_neuron)

## Calculate the activation function of a given layer
    def activate(self,x):
        r = np.dot(x,self.weight) + self.bias
        self.last_activation =  self._apply_activation(r)
        return self.last_activation

    def _apply_activation(self,r):
        if self.activation is None :
            return  r
        if self.activation == "sigmoid" :
            return  1/(1+np.exp(-r))  ## this will return the activation of the given input like sigmoid and tanh functions ...
    def activation_derevative(self,r):
        if self.activation =='sigmoid':
            return r*(1-r)
class neural_network:
    def __init__(self):
        self._layers = []
    def add_layer(self,layer):
        self._layers.append(layer)
    def feed_forward(self,x):
        for layers in self._layers :
            x = layers.activate(x) ## x is input vactor
        return x
    def predict(self,x):
        ff = self.feed_forward(x)
        if ff.ndim==1:
            return np.argmax(ff)
        return np.argmax(ff , axis=1)
    ## now we apply backpropagation algorithm to get learn the algorithm to predict on uknown inputs ....
    def backpropagation(self,x,y,learning_rate):
        output = self.feed_forward(x)
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            if layer == self._layers[-1]:
                layer.error = y-output
                layer.delta = layer.error * layer.activation_derevative(layer.last_activation)
            else :
                next_layer = self._layers[i+1]
                layer.error= np.dot(next_layer.weight , next_layer.error)
                layer.delta = layer.error * layer.activation_derevative(layer.last_activation)
        for i in range(len(self._layers)):
            layer = self._layers[i]
            ## now we have to use diffferent input (x) for different layers
            input_to_use = np.atleast_2d(x if i==0 else self._layers[i-1].last_activation)
            layer.weight  += layer.delta * input_to_use.T *learning_rate
    def learn(self,x,y,learning_rate,max_epochs):
        mses = [] # mse represents for mean square error
        for i in range(max_epochs):
            for j in range(len(x)):
                self.backpropagation(x[j],y[j],learning_rate)
            if i % 10 ==0:
                mse = np.mean(np.square(y-self.feed_forward(x)))
                mses.append(mse)
                print('Epoch: #%s, MSE:%f' %(i , float(mse)))
        return  mses
## create a network .......
nn = neural_network()
nn.add_layer(layer(2,3,'sigmoid'))
nn.add_layer(layer(3,3,'sigmoid'))
nn.add_layer(layer(3,2,'sigmoid'))
x = [[1,1],[0,0],[1,0],[0,1]]
y=[[1],[0],[1],[1]]
errors = nn.learn(x,y,0.2,5000)
plt.plot(errors)
plt.title('Change in MSE')
plt.xlabel('Epoch (every 10th)')
plt.ylabel('MSE')
plt.show()