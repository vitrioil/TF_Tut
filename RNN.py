import sys
sys.path.append("C:/Users/HiteshOza/Documents/Keras_Tut")

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time,os
import numpy as np
import psutil
import cv2
import colorama
from imdb_LSTM import load_data
colorama.init()
class ansi:
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    END = "\033[0m"
    

class RNN:
    def __init__(self,Xs,Ys,batch_size,epoch,lr,
                 folder_name = "",
                 search_timestamp = "1530634669",
                 model_path = "C:/Users/HiteshOza/Documents/TF_Tut/CNN/MNIST/1530634669/model.ckpt-2750000"
                 ,training_session = True):
        "shape = total_examples,time_step,feature"
        self.X_train,self.X_test = Xs
        self.Y_train,self.Y_test = Ys
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.training_session = training_session
        self.search_timestamp = search_timestamp
        self.timestamp =  str(int(time.time())) if self.training_session else self.search_timestamp 
        self.folder_name = folder_name
        self.logdir="C:/Users/HiteshOza/Documents/TF_Tut/CNN/"+self.folder_name+"/"+self.timestamp
        self.model_path = model_path if not training_session else model_path[:41]+self.folder_name+"/"+self.timestamp+"/model.ckpt-"+str(epoch*self.X_train.shape[0])        
        self.state_size = 100
        self.num_class = self.X_train.shape[2]
        self.time_steps = self.X_train.shape[1]
    def check_usage(self):
        return (95 < psutil.virtual_memory().percent)
        
    def convert_to_one_hot(Y, C):        
        Y = np.eye(C)[Y.reshape(-1)].T
        return Y
    
    def simpleRNN(self):
        cell =tf.contrib.rnn.BasicRNNCell(self.state_size)
        init_state = tf.zeros([self.batch-])
        rnn_outputs, final_state = tf.contrib.rnn.static_rnn(
                cell,self.rnn_input,initial_state = init_state)
        
        return rnn_outputs,final_state
    
    def train(self,func = simpleRNN):
        
        tf.reset_default_graph()
        
        X_t = tf.placeholder(tf.float64,[None,*self.X_train.shape[1:]])
        Y_t = tf.placeholder(tf.float64,[None,*self.Y_train.shape[1:]])

        self.rnn_inputs = tf.unstack(X_t,axis=1)
        outputs,final = self.func()
        
        with tf.variable_scope("Softmax_Vars"):
            W = tf.get_variable("W",[self.state_size,self.num_class])
            b = tf.get_variable("b",[self.num_class],intializer = tf.constant_initializer(0.0))
        
        logits = [tf.matmul(rnn_output,W)+b for rnn_output in rnn_outputs]
        predictions = [tf.nn.softmax(logit) for logit in logits]
        
        Y_split = tf.unstack(Y_t,num = self.time_steps,axis = 1)
        
        losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = label,logit = logit) for label,logit in zip(Y_split,logits)
        ]
        
        loss = tf.reduce_mean(losses)
        optimizer = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(loss)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for epoch in range(self.epoch):
                
        
        
if __name__ == "__main__":
    a = load_data()
    print(a[0].shape,a[1].shape)