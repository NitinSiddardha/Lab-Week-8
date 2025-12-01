def cost_derivative(self, output_activations, y):
    return (output_activations - y)#cost function returns diffrence of predicted values and actual value
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))#sigmoid function is an s shaped activation function that gives betwene 0 and 1 ie, close to 1 for +ve and close to zero for -ve - 1/(1+e^-x)
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))#the derivative of the sigmoid function wrt to input z
class Network():
    def __init__(self, sizes):
        self.num_layers = len(sizes)#len of sizes is the no of layers in the network
        self.sizes = sizes#sizes is a list that contains the number of neurons in each layer
        self.biases = [np.random.randn(x, 1) for x inss sizes[1:]] #initialising a list of biases coresponding to each layer of neurons no of elements in each bias vecotr should be no of neurons in the layer, biases start from hidden layer as input layer does not have biases
        self.weights = [np.random.randn(y, x) for y, x in zip(sizes[1:], sizes[:-1])]#initialising a list of weights coresponding to each layers of neurons no of vectors in weights matrix should be equal to hidden neurons and no of elements in each vector should be no of input neurons,therefore no of vectors start from hidden and no of elements end before output neurons
    def feedforward(self, a):
        for b, w in zip(self.biases,self.weights):#since we iterate betwene all the biases and weights and keep using the input vector a we send this a through all the network layers
            a = sigmoid(np.dot(w, a) + b)#Equation that does a forward pass betwene two layers dot product of weights matrix and input plus biase vector passes to sigmoid function
        return a
    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)#only if test_data is submitted testing is done
        n = len(training_data)
        for j in range(epochs):#no of training loops
            np.random.shuffle(training_data)#shuffling training data
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]#a list of batches each of size batch_size, if the last batch has fewer than batch_size elements it will make the last samller batch not rising an error
            for mini_batch in mini_batches:#take eatch batch in mini_batches each containing len batch_size
                self.update_mini_batch(mini_batch, eta)#pass each minibatch and learning rate to update the weights and biases of object
            if test_data:#if test data is available
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")#we send the test data to evaluate function which tests model after each gd update
            else:
                print(f"Epoch {j} complete")#else only print the number of epoch
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]#a list of arrays with zeroes in size of each of the bias vector ie, a copy of the self.biases filled with zeroes
        nabla_w = [np.zeros(w.shape) for w in self.weights]#a list of matrices with zeores in shape of each weight matrix ie, a copy of the self.weights filled with zeroes
        for x, y in mini_batch:#x is the 784 len greyscale values input vector and y is the 10 len one hot encoded output vector with 1 for correct digit and 0 for wrong
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)#delta_b and delta_w are vectors and matrices that contain derivatives of biases and weights wrt to input x
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]#we add delta_b to nabla_b to accumilate the gradient wrt to each input x in the batch size if we directly subtract delta_b from biases it is like performin gradient descent for batch size 1
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]#we add delta_w to nabla_w to accumilate the gradient wrt to each input x in the batch size if we directly subtract delta_w from weights it is like performin gradient descent for batch size 1
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]#we are performing update of biases vectors by subtracting learning rate*nabla_b from biases, we take average of nabla_b by dividing with len(batch_size) 
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]#we are performing update of weights matrices by subtracting learning rate*nabla_w from weight, we take average of nabla_w by dividing with len(batch_size) 
    def backprop(self,x,y):#the most import calulation is delta or error of each layer it is defined in every layer as the derivative of cost function wrt the weighted sum z. it is important because in is a step that is used to further calculate derivative of cost function  wrt to bias or weight or in inner layers wrt inner layers weighted sum, we backpropogate through the network by connecting the output activation of previous layer to weighted sum z of next layer by derivative of d_z(l)/d_a(l-1) = weights(l) 
        nabla_b = [np.zeros(b.shape) for b in self.biases]#a list of biases shape vectors that contain zeroes
        nabla_w = [np.zeros(w.shape) for w in self.weights]#a list of weights shape matrices that contain zeroes
        activation = x#the input of the network x is taken as first activation as it is the output of the first layer ie,input layer
        activations = [x]#a list that contains the activation vector of each layer (the first vector is output of the first layer(x)) 
        zs = []#a list of vectors that contain the weighted input z vectors of each layer before they are sent through the activation function they are used for calculating derivatives (error of each layer)
        for b,w in zip(self.biases, self.weights):#looping through the biases and weights of each layer
            z = np.dot(w,activation)+b#calculating the weighted input z for eachlayer
            zs.append(z)#appending the input to a list that contains the weighted sum to neurons in each layer
            activation = sigmoid(z)#sending the input of each layer into an activation function (sigmoid) also note that activation ie,input activation of next layer is now the output activation of old layer 
            activations.append(activation)#appending the activation of each layer to a list that contains the activations of each layer
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])#the error delta is the derivative of cost function wrt to z/weightedSum we can use chain rule so d_costfun/d_activation = derivative of 1/2*(activation-y)^2 multiplied by d_activation/d_z_weighted_sum where activation is sigmoid(z)
        nabla_b[-1]=delta#nabla_b is the derivative of each layer with respect to each biases, we set last layer derivative of biases equal to delta because derivative of d_z/d_b weighted sum wrt to any biases is one and multiplying delta by 1 is delta
        nabla_w[-1]=np.dot(delta, activations[-2].transpose())#nabla_w is the derivative of cost function wrt to each weight we multiply it by a matrix of last layer outputs because the derivative of weighted sum wrt to any weight d_z/d_w is previous layer activations ie, delta*activation[-2].transpose the reason we transpose because to reshape the vectors which have been multiplied to matrices in forward passes and get size of current layer and during backward pass need to get size of previous layer by transpose
        for l in range(2,self.num_layers):
            z=zs[-l]#take the weighted sum of last layer ie, output layer
            sp=sigmoid_prime(z)#calculate the derivative of activation of last layer wrt input z
            delta=np.dot(self.weights[-l+1].transpose(),delta) * sp#delta of hidden layer is the derivative of cost function wrt to the weighted sum z of inner layer ie d_c/d_z(L-n) it is calculated by chain rule we multiply the forward layer error by derivative of forward layer weighted sum wrt activations of previous layer which is current layer weights.transpose
            nabla_b[-l] = delta#the derivative of hidden layer obtained by multiplication of the delta by derivative of current layer weighted sum z wrt bias which is 1 since bias is only added in the weighted input z ie,equal to delta, stored at appropriate layer of nabla_b
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())#the derivative of hidden layer obtained by multiplying the delta with the derivative of current layer weighted sum z wrt weights which is the activations.transpose of previous layer we transpose because to chaing the size of the vector as we go back
        return (nabla_b, nabla_w)#nabla_b,nabla_w are now the gradient of cost function with respect to all biases and weights in the network ie, d_C/d_b and d_C/d_w
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]# np.argmax() returns the index of maximum value in an array 
        return sum(int(x == y) for (x, y) in test_results)#returning the no of times fedfroward x is equal to y