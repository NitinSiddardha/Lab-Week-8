import numpy as np
def sigmoid(z):
    return np.where(z >= 0, 1.0 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
def l_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)
def dl_relu(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)
def swish(z):
    return z*sigmoid(z)
def d_swish(z):
    sig_z=sigmoid(z)
    return sig_z+(z*sig_z)*(1-sig_z)
def softmax(z):
    exp_z=np.exp(z-np.max(z,axis=0,keepdims=True))
    return exp_z/np.sum(exp_z,axis=0,keepdims=True)
def cross_entropy(a,y):
    return np.sum(np.nan_to_num(-y*np.log(a)),keepdims=True)
def soft_gradient_scale(g):
    return g + g*np.exp(-np.abs(g))/100
def layer_norm(z):
    return (z-np.mean(z))/(np.std(z)+1e-5)
class Network:
    def __init__(self,layers):
        self.layers=layers
        self.n_layers=len(layers)
        self.biases=[np.zeros((x,1)) for x in self.layers[1:]]
        self.weights=[np.random.randn(y,x)*np.sqrt(2/x) for x,y in zip(self.layers[:-1],self.layers[1:])]
        self.velo_b=[np.zeros(b.shape) for b in self.biases]
        self.velo_w=[np.zeros(w.shape) for w in self.weights]
    def feedforward(self,a):
        for l,(w,b) in enumerate(zip(self.weights,self.biases)):
            z = np.dot(w, a) + b
            #z = layer_norm(z)
            if l+2!=self.n_layers:
                a=swish(z)
            else:
                a=softmax(z)
        return a
    def sgd(self,train_data,eta,lmbda,mew,mini_batch_size,epochs,test_data=False):
        n=len(train_data)
        for i in range(epochs):
            np.random.shuffle(train_data)
            mini_batches=[train_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta,lmbda,mew,n)
            if(test_data):
                print(f"epoch {i+1} cost {np.mean([cross_entropy(self.feedforward(x),y) for x,y in train_data[0:10001]]):.2f} accu {self.evaluate(test_data):.2f}")
            else:
                print(f"epoch {i+1} cost {np.mean([cross_entropy(self.feedforward(x),y) for x,y in train_data]):.2f}")
    def update_mini_batch(self,mini_batch,eta,lmbda,mew,n):
        delta_b=[np.zeros(b.shape) for b in self.biases]
        delta_w=[np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w=self.backprop(x,y)
            delta_b=[db+dnb for db,dnb in zip(delta_b,delta_nabla_b)]
            delta_w=[dw+dnw for dw,dnw in zip(delta_w,delta_nabla_w)]
        self.velo_b=[mew*vb-(eta/len(mini_batch))*db for vb,db in zip(self.velo_b,delta_b)]
        self.velo_w=[mew*vw-(eta/len(mini_batch))*dw for vw,dw in zip(self.velo_w,delta_w)]
        self.biases=[b+vb for b,vb in zip(self.biases,self.velo_b)]
        self.weights=[w*(1-(eta*lmbda/n))+vw for w,vw in zip(self.weights,self.velo_w)]
    def backprop(self,x,y):
        delta_nabla_b=[np.zeros(b.shape) for b in self.biases]
        delta_nabla_w=[np.zeros(w.shape) for w in self.weights]
        a=x
        A=[x]
        Z=[]
        for l,(w,b) in enumerate(zip(self.weights,self.biases)):
            z=np.dot(w,a)+b
            #z=layer_norm(z)
            Z.append(z)
            if l+2!=self.n_layers:
                a=swish(z)
            else:
                a=softmax(z)
            A.append(a)
        delta=A[-1]-y
        delta_nabla_b[-1]=delta
        #print(f"layer -1 \n{delta}")
        delta_nabla_w[-1]=np.dot(delta,A[-2].transpose())
        for l in range(2,self.n_layers):
            delta=np.dot(self.weights[-l+1].transpose(),delta)*d_swish(Z[-l])
            #print(f"layer {-l} before\n{delta}")
            #delta = soft_gradient_scale(delta)
            #print(f"layer {-l} after\n{delta}")
            delta_nabla_b[-l]=delta
            delta_nabla_w[-l]=np.dot(delta,A[-l-1].transpose())
        return delta_nabla_b,delta_nabla_w
    def evaluate(self,train_data):
        return np.sum([int(np.argmax(self.feedforward(x))==np.argmax(y)) for x,y in train_data])/len(train_data)*100


        

