import numpy as np
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))#sigmoid
def swish(z):
    return z*sigmoid(z)#swish
def d_swish(z):
    return sigmoid(z)+z*sigmoid(z)*(1-sigmoid(z))#derivative swish
def l_relu(z, alpha=0.01):#leaky relu
    return np.where(z > 0, z, alpha * z)
def dl_relu(z, alpha=0.01):#derivative leaky relu
    return np.where(z > 0, 1, alpha)
def softmax(z):#soft max final layer activation
    exp_z = np.exp(z-np.max(z))
    return exp_z / np.sum(exp_z,keepdims=True)
def cross_entropy_loss(y_p, y_t):#cross entropy function
    return np.sum(np.nan_to_num(-y_t*np.log(y_p)))
def soft_grad_scaling(g):#implementing soft gradient scaling for doubling small gradients to prevent small gradient vanishing
    return g + g*np.exp(-np.abs(g))/100
class Network:#ups-SWISH activation function ,HE gaussian weight initialization, L2 regularization, momentum based sgd, cross entropy removes saturation in sigmoid, softmax models output as probability, soft gradient scaling increases small gradients
    def __init__(self, layers):
        self.n_layers = len(layers)
        self.layers = layers        
        self.biases = [np.zeros((x, 1)) for x in layers[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(2/x) for x, y in zip(layers[:-1], layers[1:])]#better weight initialization - HE gaussian weight initialization divides by no of input neurons to make standard deviation of weight sum small
        self.velo_b = [np.zeros(b.shape) for b in self.biases]#empty place holders for velocity of gradient descent
        self.velo_w = [np.zeros(w.shape) for w in self.weights]#velocity build up if gradient travels in same direction for long ie same sign
    def feedforward(self, a):
        for l, (w, b) in enumerate(zip(self.weights, self.biases)):#unique activation functions for output - softmax helps model probability in output
            if l + 2 != self.n_layers:
                a = swish(np.dot(w, a) + b)
            else:
                a = softmax(np.dot(w, a) + b)
        return a
    def sgd(self,train_data,eta,lmbda,mew,mini_batch_size,epochs,test_data=False):
        n=len(train_data)
        for i in range(epochs):
            np.random.shuffle(train_data)
            mini_batches=[train_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta,lmbda,mew,n)
            if(test_data):
                print(f"epoch {i+1} cost {np.mean([cross_entropy_loss(self.feedforward(x),y) for x,y in train_data[0:10001]]):.2f} accu {self.evaluate(test_data):.2f}")
            else:
                print(f"epoch {i+1} cost {np.mean([cross_entropy_loss(self.feedforward(x),y) for x,y in train_data]):.2f}")
    def update_mini_batch(self, mini_batch, eta, lmbda, mew, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.velo_b = [mew*vb-(eta/len(mini_batch))*nb for vb,nb in zip(self.velo_b,nabla_b)]#velocity in b increases constanty if gradient in same direction else decrease
        self.velo_w = [mew*vw-(eta/len(mini_batch))*nw for vw,w,nw in zip(self.velo_w,self.weights,nabla_w)]#velocity in w 
        self.biases = [b + vb for b, vb in zip(self.biases, self.velo_b)]#regularization on weights - L2
        self.weights = [(1-(eta*lmbda)/n)*w + vw for w, vw in zip(self.weights, self.velo_w)]#L2 achived by multiplying weight by (1-(eta*lam)/n) which skrinks w
    def backprop(self, x, y):
        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]
        a = x
        A = [a]
        Z = []
        for l,(w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, a) + b
            Z.append(z)
            if l+2!= self.n_layers:
                a = swish(z)
            else:
                a = softmax(z)
            A.append(a)
        delta = A[-1] - y #better cost function removes sigmoid function preventing saturation of d_sigmoid - cross entropy loss
        delta = soft_grad_scaling(delta)
        delta_nabla_b[-1] = delta
        delta_nabla_w[-1] = np.dot(delta,A[-2].transpose())
        for l in range(2, self.n_layers):
            delta = np.dot(self.weights[-l+1].transpose(),delta) * d_swish(Z[-l])
            delta = soft_grad_scaling(delta)
            delta_nabla_b[-l] = delta
            delta_nabla_w[-l] = np.dot(delta, A[-l-1].transpose())
        return delta_nabla_b, delta_nabla_w
    def evaluate(self, test_data):
        return np.sum([int(np.argmax(self.feedforward(x))==np.argmax(y)) for x,y in test_data])/len(test_data)*100