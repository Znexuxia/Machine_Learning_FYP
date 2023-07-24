#-------------------------------------------------------------------------------
# Name:        Neural Network
# Purpose:
#
# Author:      Shahazureen Ikwan
#
# Created:     02/08/2019
# Copyright:   (c) ILLEGEAR LAGUNA 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

# Import and important python package
import numpy as np

#Creating Neural network Class
class NeuralNetwork:
    def init(self,x,y):
        self.input=x
        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)
        self.y=y
        self.output =np.zeros(y,shape)

    #Create feedfoward using sigmoid activation function
    def feedfoward(self):
        self.layer1 = sigmoid(np.dot(self.input,self.weights1))
        self.output = sigmoid(np.dot(self.layer1,self.weights2))


    #Create backpropagation
    def backprop(self):
        #application of the chain rule to find derivative of the loss function
        #with respect to Weights2 and Weights1

        d_weights2 = np.dot(self.layer1.T,(2*(self.y-self.output)*
                     sigmoid_derivative(self.output)))

        d_weights1 = np.dot(self.input.T,(np.dot(2*(self.y-self.output)
                     *sigmoid_derivative(self.output),self.weights2.T)
                     *sigmoid_derivative(self.layer1)))

        #Update the weights with the derivative slope of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
