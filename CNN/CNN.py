import numpy as np
def relu(x):
    return np.maximum(0, x)
def d_relu(x):
    return np.where(x > 0, 1, 0)
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
class SimpleCNN:
    def __init__(self, input_shape, filter_size, num_filters, pool_size, output_size):
        """
        input_shape: (height, width) of input image    
        filter_size: Size of convolution filter (assumed square)
        num_filters: Number of convolutional filters
        pool_size: Size of max pooling filter (assumed square)
        output_size: Number of output classes
        """
        self.input_shape = input_shape  
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.pool_size = pool_size
        self.output_size = output_size
        # Initialize convolution filters and biases
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size**2)
        self.conv_biases = np.zeros((num_filters, 1))
        # Compute dimensions of convolution output (valid convolution, no padding)
        self.conv_out_h = input_shape[0] - filter_size + 1
        self.conv_out_w = input_shape[1] - filter_size + 1
        # Compute dimensions of pooling output (non-overlapping windows)
        self.pool_out_h = self.conv_out_h // pool_size
        self.pool_out_w = self.conv_out_w // pool_size
        # Fully connected layer weights and biases.
        fc_input_size = self.pool_out_h * self.pool_out_w * num_filters
        self.fc_weights = np.random.randn(output_size, fc_input_size) / fc_input_size
        self.fc_biases = np.zeros((output_size, 1))
        # Pre-allocate arrays for forward pass outputs (assuming fixed input dimensions)
        self.conv_out = np.zeros((self.conv_out_h, self.conv_out_w, num_filters))
        self.conv_output = np.zeros_like(self.conv_out)
        self.pool_out = np.zeros((self.pool_out_h, self.pool_out_w, num_filters))
        self.pool_mask = np.zeros((self.conv_out_h, self.conv_out_w, num_filters))
        self.flat = np.zeros((fc_input_size, 1))
        # Placeholders for catching during forward pass
        self.last_input = None
        self.last_totals = None
        self.out = None
    def forward(self, x):
        """
        Forward pass through the CNN:
         1. Convolution + Bias addition.
         2. ReLU activation.
         3. Max Pooling.
         4. Flatten.
         5. Fully Connected Layer + Softmax.
        """
        
        self.last_input = x.copy()
        # Convolution layer
        for k in range(self.num_filters):        
            for i in range(self.conv_out_h):
                for j in range(self.conv_out_w):
                    # Extract region from input and perform convolution
                    region = x[i:i+self.filter_size, j:j+self.filter_size]
                    self.conv_out[i, j, k] = np.sum(region * self.filters[k]) + self.conv_biases[k]
        # Apply ReLU activation (store in pre-allocated array)
        self.conv_output = relu(self.conv_out)
        # Max Pooling (non-overlapping windows)
        # Reset pooling output and mask (in case they hold previous values)
        self.pool_out.fill(0)
        self.pool_mask.fill(0)
        for k in range(self.num_filters):
            for i in range(self.pool_out_h):
                for j in range(self.pool_out_w):
                    region = self.conv_output[i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size, k]
                    max_val = np.max(region)
                    self.pool_out[i, j, k] = max_val
                    # Create mask for the pooling region
                    mask = (region == max_val)
                    self.pool_mask[i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size, k] = mask
        # Flatten pooled output for the fully connected layer
        self.flat = self.pool_out.flatten().reshape(-1, 1)
        # Fully connected layer and softmax
        self.last_totals = np.dot(self.fc_weights, self.flat) + self.fc_biases
        self.out = softmax(self.last_totals)
        return self.out
    def backprop(self, x, y):
        """
        Compute the loss and gradients for a single training example.
        Returns:
          loss,
          grad_fc_weights, grad_fc_biases,
          grad_filters, grad_conv_biases.
        """
        # Forward pass
        output = self.forward(x)
        loss = -np.log(output[np.argmax(y)] + 1e-7)
        # Gradient of loss with respect to output layer totals (after softmax)
        d_out = output - y  # Shape: (output_size, 1)
        # Gradients for the fully connected layer
        grad_fc_weights = np.dot(d_out, self.flat.T)  # Shape: (output_size, fc_input_size)
        grad_fc_biases = d_out  # Shape: (output_size, 1)
        # Backpropagate through fully connected layer
        d_flat = np.dot(self.fc_weights.T, d_out)  # Shape: (fc_input_size, 1)
        d_pool = d_flat.reshape(self.pool_out_h, self.pool_out_w, self.num_filters)
        # Backpropagation through max pooling
        d_conv = np.zeros_like(self.conv_output)
        for k in range(self.num_filters):
            for i in range(self.pool_out_h):
                for j in range(self.pool_out_w):
                    r_start = i * self.pool_size
                    c_start = j * self.pool_size
                    # Distribute gradient to the position of the max value in the pooling window
                    d_conv[r_start:r_start+self.pool_size, c_start:c_start+self.pool_size, k] += (
                        d_pool[i, j, k] * self.pool_mask[r_start:r_start+self.pool_size, c_start:c_start+self.pool_size, k]
                    )
        # Backpropagate through ReLU activation
        d_conv *= d_relu(self.conv_output)
        # Gradients for the convolution layer
        grad_filters = np.zeros_like(self.filters)
        grad_conv_biases = np.zeros_like(self.conv_biases)
        for k in range(self.num_filters):
            for i in range(self.conv_out_h):
                for j in range(self.conv_out_w):
                    region = x[i:i+self.filter_size, j:j+self.filter_size]
                    grad_filters[k] += d_conv[i, j, k] * region
                    grad_conv_biases[k] += d_conv[i, j, k]
        return loss, grad_fc_weights, grad_fc_biases, grad_filters, grad_conv_biases
    def train(self, training_data, epochs, lr, mini_batch_size):
        """
        Train the CNN using mini-batch gradient descent.
        training_data: list of tuples (x, y) where x is an input image and y is the one-hot label.
        """
        n = len(training_data)
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            total_loss = 0
            for mini_batch in mini_batches:
                # Initialize accumulators for gradients for the mini-batch
                accum_grad_fc_weights = np.zeros_like(self.fc_weights)
                accum_grad_fc_biases = np.zeros_like(self.fc_biases)
                accum_grad_filters = np.zeros_like(self.filters)
                accum_grad_conv_biases = np.zeros_like(self.conv_biases)
                batch_loss = 0
                for x, y in mini_batch:
                    loss, grad_fc_w, grad_fc_b, grad_filters, grad_conv_b = self.backprop(x, y)
                    batch_loss += loss
                    accum_grad_fc_weights += grad_fc_w
                    accum_grad_fc_biases += grad_fc_b
                    accum_grad_filters += grad_filters
                    accum_grad_conv_biases += grad_conv_b
                # Update parameters using the average gradients from the mini-batch
                batch_size = len(mini_batch)
                self.fc_weights -= lr * (accum_grad_fc_weights / batch_size)
                self.fc_biases -= lr * (accum_grad_fc_biases / batch_size)
                self.filters -= lr * (accum_grad_filters / batch_size)
                self.conv_biases -= lr * (accum_grad_conv_biases / batch_size)
                total_loss += batch_loss
            avg_loss = total_loss / n
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    def evaluate(self, test_data):
        """
        Evaluate the CNN on test data.
        test_data: list of tuples (x, y) where x is an input image and y is the one-hot label.
        Returns the accuracy as a float.
        """
        num_correct = 0
        for x, y in test_data:
            output = self.forward(x)
            if np.argmax(output) == np.argmax(y):
                num_correct += 1
        return num_correct / len(test_data)
# Example usage:
if __name__ == "__main__":
    # Create a dummy dataset: 10 images of shape (10, 10) and 2 classes.
    training_data = []
    for _ in range(10):
        x = np.random.rand(10, 10)
        label = np.zeros((2, 1))
        label[np.random.randint(0, 2)] = 1
        training_data.append((x, label))
    # For simplicity, use the training data as test data.
    test_data = training_data.copy()
    # Create a SimpleCNN instance.
    cnn = SimpleCNN(input_shape=(10, 10), filter_size=3, num_filters=2, pool_size=2, output_size=2)
    # Train the network using mini-batch gradient descent.
    cnn.train(training_data, epochs=5, lr=0.01, mini_batch_size=2)
    # Evaluate the network.
    accuracy = cnn.evaluate(test_data)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")