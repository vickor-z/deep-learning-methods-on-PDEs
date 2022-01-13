# CLASS DEFINITIONS FOR NEURAL NETWORKS USED IN DEEP Galerkin METHOD

#%% import needed packages
import tensorflow as tf
import numpy as np

#%% Fully connected (dense) layer - modification of Keras layer class
   
class DenseLayer(tf.keras.layers.Layer):
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, output_dim, input_dim, transformation=None):
        '''
        Args:
            input_dim:       dimensionality of input data
            output_dim:      number of outputs for dense layer
            transformation:  activation function used inside the layer; using
                             None is equivalent to the identity map 
        
        Returns: customized Keras (fully connected) layer object 
        '''        
        
        # create an instance of a Layer object (call initialize function of superclass of DenseLayer)
        super(DenseLayer,self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        ### define dense layer parameters (use Xavier initialization)
        # w vectors (weighting vectors for output of previous layer)
        self.W = self.add_variable("W", shape=[self.input_dim, self.output_dim],
                                   initializer = tf.contrib.layers.xavier_initializer())
        
        # bias vectors
        self.b = self.add_variable("b", shape=[1, self.output_dim])
        
        if transformation == "tanh":
            self.transformation = tf.tanh
        elif transformation == "relu":
            self.transformation = tf.nn.relu
        else:
            self.transformation = transformation
    
    
    # main function to be called 
    def call(self,X):
        '''Compute output of a dense layer for a given input X 

        Args:                        
            X: input to layer            
        '''
        
        # compute dense layer output
        S = tf.add(tf.matmul(X, self.W), self.b)
                
        if self.transformation:
            S = self.transformation(S)
        
        return S
    
#%% Reslayer - residual network
class ResLayer(tf.keras.layers.Layer):
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, output_dim, input_dim, transformation='tanh'):
        '''
        Args:
            input_dim:       dimensionality of input data
            output_dim:      number of outputs for dense layer
            transformation:  activation function used inside the layer; using
                             None is equivalent to the identity map 
        
        Returns: customized Keras (fully connected) layer object 
        '''        
        
        # create an instance of a Layer object (call initialize function of superclass of DenseLayer)
        super(ResLayer,self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        ### define dense layer parameters (use Xavier initialization)
        # w vectors (weighting vectors for output of previous layer)
        self.W1 = self.add_variable("W1", shape=[self.input_dim, self.output_dim],
                                   initializer = tf.contrib.layers.xavier_initializer())
        self.W2 = self.add_variable("W2", shape=[self.output_dim, self.input_dim],
                                   initializer = tf.contrib.layers.xavier_initializer())
        
        # bias vectors
        self.b1 = self.add_variable("b1", shape=[1, self.output_dim])
        self.b2 = self.add_variable("b2", shape=[1, self.input_dim])
        
        self.transformation = tf.nn.elu
    
    
    # main function to be called 
    def call(self,X):
        '''Compute output of a dense layer for a given input X 

        Args:                        
            X: input to layer            
        '''
        
        # compute dense layer output
        S = tf.add(tf.matmul(X, self.W1), self.b1)
        S = self.transformation(S)
        S = tf.add(tf.matmul(S, self.W2), self.b2)
        S = self.transformation(S)
        S = tf.add(2*S, X**5)
        S = self.transformation(S)
                
        return S
    
#%% Neural network architecture used in DGM - modification of Keras Model class

class SingleNet(tf.keras.Model):
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, layer_width, n_layers, input_dim, final_trans=None,active = "tanh"):
        '''
        Args:
            layer_width: 
            n_layers:    number of intermediate layers
            input_dim:   spaital dimension of input data (EXCLUDES time dimension)
            final_trans: transformation used in final layer
        
        Returns: customized Keras model object representing DGM neural network
        '''  
        
        # create an instance of a Model object (call initialize function of superclass of DGMNet)
        super(SingleNet,self).__init__()
        
        # define initial layer as fully connected 
        # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
        self.initial_layer = DenseLayer(layer_width, input_dim+1, transformation = active)
        
        # define final layer as fully connected with a single output (function value)
        self.final_layer = DenseLayer(1, layer_width, transformation = final_trans)
    
    
    # main function to be called  
    def call(self,t,x):
        '''            
        Args:
            t: sampled time inputs 
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs (t,x)                
        '''     
        
        # define input vector as time-space pairs
        X = tf.concat([t,x],1)
        
        # call initial layer
        S = self.initial_layer.call(X)
        # call final layers
        result = self.final_layer.call(S)
        
        return result    
#%% Neural network architecture used in DGM - modification of Keras Model class



class DeepNet(tf.keras.Model):
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, layer_width, n_layers, input_dim, final_trans=None,active = "tanh"):
        '''
        Args:
            layer_width: 
            n_layers:    number of intermediate layers
            input_dim:   spaital dimension of input data (EXCLUDES time dimension)
            final_trans: transformation used in final layer
        
        Returns: customized Keras model object representing DGM neural network
        '''  
        
        # create an instance of a Model object (call initialize function of superclass of DGMNet)
        super(DeepNet,self).__init__()
        
        # define initial layer as fully connected 
        # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
        self.initial_layer = DenseLayer(layer_width, input_dim+1, transformation = active)
        
        # define intermediate layers
        self.n_layers = n_layers
        self.DenseLayerList = []
                
        for _ in range(self.n_layers):
            self.DenseLayerList.append(DenseLayer(layer_width, layer_width, transformation = active))

        # define final layer as fully connected with a single output (function value)
        self.final_layer = DenseLayer(1, layer_width, transformation = final_trans)
    
    
    # main function to be called  
    def call(self,t,x):
        '''            
        Args:
            t: sampled time inputs 
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs (t,x)                
        '''     
        
        # define input vector as time-space pairs
        X = tf.concat([t,x],1)
        
        # call initial layer
        S = self.initial_layer.call(X)
        
        # call intermediate layers
        for i in range(self.n_layers):
            S = self.DenseLayerList[i].call(S)
        
        # call final layers
        result = self.final_layer.call(S)
        
        return result

#%%
class ResNet(tf.keras.Model):
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, layer_width, n_layers, input_dim, final_trans="relu"):
        '''
        Args:
            layer_width: 
            n_layers:    number of intermediate layers
            input_dim:   spaital dimension of input data (EXCLUDES time dimension)
            final_trans: transformation used in final layer
        
        Returns: customized Keras model object representing DGM neural network
        '''  
        
        # create an instance of a Model object (call initialize function of superclass of DGMNet)
        super(ResNet,self).__init__()
        
        # define initial layer as fully connected 
        # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
        self.initial_layer = DenseLayer(layer_width, input_dim+1, transformation = "relu")
        
        # define intermediate layers
        self.n_layers = n_layers
        self.ResLayerList = []
                
        for _ in range(self.n_layers):
            self.ResLayerList.append(ResLayer(layer_width, layer_width))
        
        # define final layer as fully connected with a single output (function value)
        self.final_layer = DenseLayer(1, layer_width, transformation = final_trans)
    
    
    # main function to be called  
    def call(self,t,x):
        '''            
        Args:
            t: sampled time inputs 
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs (t,x)                
        '''     
        
        # define input vector as time-space pairs
        X = tf.concat([t,x],1)
        
        # call initial layer
        S = self.initial_layer.call(X)
        
        # call intermediate layers
        for i in range(self.n_layers):
            S = self.ResLayerList[i].call(S)
        
        # call final layers
        result = self.final_layer.call(S)
        
        return result

#%%
class DeepNet_HeatND_exact(tf.keras.Model):
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, layer_width, n_layers, input_dim, final_trans = None, ic = tf.nn.relu):
        '''
        Args:
            layer_width: 
            n_layers:    number of intermediate layers
            input_dim:   spaital dimension of input data (EXCLUDES time dimension)
            final_trans: transformation used in final layer
        
        Returns: customized Keras model object representing DGM neural network
        '''  
        # create an instance of a Model object (call initialize function of superclass of DGMNet)
        super(DeepNet_HeatND_exact,self).__init__()
        k = 1
        self.ic = ic
        # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
        self.initial_layer = DenseLayer(4*layer_width, input_dim+1, transformation = "tanh")
        
        
        # define intermediate layers
        self.n_layers = n_layers
        self.DenseLayerListA = []
        self.DenseLayerListB = []
        self.DenseLayerListA.append(DenseLayer(layer_width, 4*layer_width, transformation = "tanh"))
        self.DenseLayerListB.append(DenseLayer(2*layer_width, 4*layer_width, transformation = "tanh"))
        
        for _ in range(self.n_layers):
            self.DenseLayerListA.append(DenseLayer(layer_width, layer_width, transformation = "tanh"))
            self.DenseLayerListB.append(DenseLayer(2*layer_width, 2*layer_width, transformation = "tanh"))
        # define final layer as fully connected with a single output (function value)
        self.final_layer_A = DenseLayer(1, layer_width, transformation = final_trans)
        self.final_layer_B = DenseLayer(input_dim, 2*layer_width)
    
    # main function to be called  
    def call(self,t,x):
        '''            
        Args:
            t: sampled time inputs 
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs (t,x)                
        '''     
        
        # define input vector as time-space pairs
        # Add the force

        X = tf.concat([t,x],1)
        
        # call initial layer
        S = self.initial_layer.call(X)
        SA = self.DenseLayerListA[0].call(S)
        SB = self.DenseLayerListB[0].call(S)
        
        # call intermediate layers
        for i in range(self.n_layers):
            SA = self.DenseLayerListA[i+1].call(SA)
            SB = self.DenseLayerListB[i+1].call(SB)
        # call final layers
        A = self.final_layer_A.call(SA)
        B = self.final_layer_B.call(SB)
        
        result = tf.concat([A,B],1)
        return result

