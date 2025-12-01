import numpy as np 
def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))#sigmoid
def d_sigmoid(z):
        return sigmoid(z)*(1-sigmoid(z))#derivative sigmoid
def relu(z):
          return np.maximum(0,z)#relu
def d_relu(z):
        return np.where(z>0,1,0)#derivative relu
def leaky_relu(z, alpha=0.01):#leaky relu
    return np.where(z > 0, z, alpha * z)
def d_leaky_relu(z, alpha=0.01):#derivative of leaky relu
    return np.where(z > 0, 1, alpha)
class Network():
        def __init__(self,layers):
                self.layers=layers#layers in matrix form
                self.n_layers=len(layers)#no of layers
                self.biases=[np.random.randn(x,1) for x in self.layers[1:]]#no biases for starting/input layer
                self.weights=[np.random.randn(y,x) for x,y in zip(self.layers[:-1],self.layers[1:])]#weights matrix is - for each hidden layer neuron y we have x weights corresponding to x input neurons
        def feedforward(self,a):
                for l,(w,b) in enumerate(zip(self.weights,self.biases)):#enumerate gives tuples with startin num 0 to get weights,biases corresponding layer num
                        z=np.dot(w,a)+b#weighted sum is the input to activation function of layer
                        if l+2!=self.n_layers:#if last layer condition not satified
                                a=relu(z)#application of activation
                        else:
                                a=sigmoid(z)#since starting num 0 and 1st layer not counted 2 added to l
                return a
        def sgd(self, training_data, epochs, mini_batch_size, eta,test_data=None):
                n = len(training_data)#len of training data
                for i in range(epochs):#for each epoch
                        np.random.shuffle(training_data)#shuffle training data
                        mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]#training data is divided into mini batches each of size mini_batch_size
                        for mini_batch in mini_batches:#for each mini_batch update weights,biases 
                                self.update_mini_batch(mini_batch, eta)#eta is learning rate
                if test_data:
                        print(f"Epoch {i}: {self.evaluate(test_data)}")#evaluate model on test data
                else:
                        print (f"Epoch {i} complete")
        def update_mini_batch(self,mini_batch,eta):
                delta_b=[np.zeros(b.shape) for b in self.biases]#empty biases matrix holding sum of biases for each minibatch
                delta_w=[np.zeros(w.shape) for w in self.weights]#empty weights matrix holding sum of weights for each minibatch
                for x,y in mini_batch:#x is input, y is expected value
                        delta_nabla_b,delta_nabal_w=self.backprop(x,y)#derivative of biases and weights for one mini batch
                        delta_b=[db+dnb for db,dnb in zip(delta_b,delta_nabla_b)]#adding bias derivatives to place holder
                        delta_w=[dw+dnw for dw,dnw in zip(delta_w,delta_nabal_w)]#adding weight derivatives to place holder
                self.biases=[b-(eta/len(mini_batch))*db for b,db in zip(self.biases,delta_b)]#bias update rule - from each bias subtract learn_rate*avg(derivative)
                self.weights=[w-(eta/len(mini_batch))*dw for w,dw in zip(self.weights,delta_w)]#weight update rule - from each weight subtract learn_rate*avg(derivative)
        def backprop(self,x,y):
                delta_nabla_b=[np.zeros(b) for b in self.biases]#bias derivative place holder
                delta_nabla_w=[np.zeros(w) for w in self.weights]#weight derivative place holder
                a=x#input to network/input layer activation
                A=[a]#list storing all activations of layers
                Z=[]#list storing weighted sums of layers
                for l,(w,b) in enumerate(zip(self.weights,self.biases)):#for each layer of network
                        z=np.dot(w,a)+b#calculate the weighted sum
                        Z.append(z)#and add it to list
                        if l+2!=self.n_layers:#calculate the activation of each layer
                                a=relu(z) 
                        else:
                                a=sigmoid(z)
                        A.append(a)#and add it to list
                delta= (A[-1]-y)*d_sigmoid(Z[-1]) #delta is the derivative of cost function with respect to weighted sum of corresponding layer so dc/da*da/dz
                delta_nabla_b[-1]=delta#derivative of cost function with rescect to bias so dc/da*da/dz*dz/db dz/db is 1
                delta_nabla_w[-1]=np.dot(delta,A[-2].transpose())#derivative of cost function with rescect to weight so dc/da*da/dz*dz/dw dz/dw is activation of prev layer
                for l in range(2,self.n_layers):#start from reverse since last layer already calculated and 1st layer not calculated, interatively multiply derivative 
                        delta=np.dot(delta,self.weights[-l+1].transpose())*d_relu(Z[-l])#delta is the derivative of cost function with respect to weighted sum of corresponding layer so dc/dzL*dzL/dal*dal/dzl: dc/dzl is prev layer delta,dzL/dal is weight of next layer, dal/dzl is weighted sum of current layer 
                        delta_nabla_b[-l]=delta#derivative of cost function with respect to current bias so dc/dzL*dzL/dzl*dzl/db dzl/db is 1
                        delta_nabla_w[-l]=np.dot(delta,A[-l-1].transpose())#derivative of cost function with respect to current weight so dc/dzL*dzL/dzl*dzl/dw dzl/dw is activation of prev layer
                return delta_nabla_b,delta_nabla_w#return the derivatives for a mini batch
        def evaluate(self, test_data):#accuracy on test data
                test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]#argmax returns index of maximum element store indeces as tuples in a list
                return sum([int(x == y) for (x, y) in test_results])/len(test_data)#return percentage of matching indeces assuming one hot encoded