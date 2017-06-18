"""
A simple neural net to learn tensorflow
"""
import tensorflow as tf
import random

def secretFunction(x):
    """
    A nonsense function that swaps pixels around
    
    x is a 2x2 array
    """
    return [[x[1][1], x[0][0]], [x[0][1], x[1][0]]]
    
def generateTrainingData():
    """
    Returns tuple (X, Y) where X list of random inputs and Y is list of secretFunction(X)
    """
    X = [[[float(random.randint(1, 100)), float(random.randint(1, 100))],
         [float(random.randint(1, 100)), float(random.randint(1, 100))]] for i in range(100)]
    Y = [secretFunction(x) for x in X]
    return (X, Y)
    
class SecretLearner(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [2,2])
        self.y = tf.placeholder(tf.float32, [2,2])
        
        self.hiddenLayer = tf.layers.dense(self.x, 10, tf.nn.relu)
        self.output = tf.layers.dense(self.hiddenLayer, 2)
        
        self.loss = tf.losses.mean_squared_error(self.y, self.output)
        self.optimizer = tf.train.GradientDescentOptimizer(0.5)
        self.training_op = self.optimizer.minimize(self.loss)
        
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
    
    def learn(self, training_data):
        X, Y = training_data
        for _ in range(100):
            for i in range(100):
                x, y = X[i], Y[i]
                _, prediction, loss = self.session.run(
                    [self.training_op, self.output, self.loss],
                    {self.x: x, self.y: y})
                if i == 0:
                    print(x, y, prediction, loss)
    
    def testKnowledge(self, x, y):
        prediction, loss = self.session.run([self.output, self.loss],
                                            {self.x: x, self.y: y})
        return (prediction, loss)

def main():
    # Train:
    training_data = generateTrainingData()
    learner = SecretLearner()
    learner.learn(training_data)
    # Test:
    """
    test_data = generateTrainingData()
    X, Y = test_data
    losses = []
    for i in range(100):
        x, y = X[i], Y[i]
        prediction, loss = learner.testKnowledge(x, y)
        losses.append(loss)
        print("x: {}, y: {}, prediction: {}, loss: {}".format(
            x, y, prediction, loss))
    print("Avg loss: {}".format(sum(losses) / len(losses)))
    """
    
if __name__ == "__main__":
    main()