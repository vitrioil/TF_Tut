import os
import cv2
import numpy as np
import pandas as pd
from CNN import CNN
import tensorflow as tf
class HEC(CNN):

    def __init__(self,Xs,Ys,batch_size,epoch,lr,
                 folder_name = "",
                 search_timestamp = "1530634669",
                 model_path = "C:/Users/HiteshOza/Documents/TF_Tut/CNN/MNIST/1530634669/model.ckpt-2750000"
                 ,training_session = True):
                 CNN.__init__(self,Xs,Ys,batch_size,epoch,lr,folder_name, search_timestamp,model_path ,training_session)
                 
                 self.assign_model()
    
    def assign_model(self):
        self.model = self.newModel
                 
    def newModel(self,x_t,training = True):
        with tf.variable_scope("Conv1"):
            z = self.conv2d(x_t, 16, 5, 2,activation = tf.nn.relu)
            z = self.max_pool(z, 2, 2)
        with tf.variable_scope("Conv2"):
            z = self.conv2d(z, 32, 5, 2,activation = tf.nn.relu)
            z = self.max_pool(z, 2, 2)
        with tf.variable_scope("Conv3"):
            z = self.conv2d(z, 64, 3, 2,activation = tf.nn.relu)
            z = self.max_pool(z, 2, 2)
        with tf.variable_scope("Flatten1"):
            z = tf.layers.flatten(z)
        with tf.variable_scope("Dense1"):
            z = tf.layers.dense(z,units=128,activation=tf.nn.relu)
           # z = tf.layers.dropout(z,rate=0.2,training=training)
        with tf.variable_scope("Dense2"):
            z = tf.layers.dense(z,units=128,activation=tf.nn.relu)
            #z = tf.layers.dropout(z,rate=0.2,training=training)
        with tf.variable_scope("Dense3"):
            logits = tf.layers.dense(z,units=self.Y_train.shape[1],name="logits")
        y = tf.nn.softmax(logits,name="y_hat")
        #print('h')
        return y,logits

def regularize(path,newpath,shape= (100,100)):
    f = []
    for p,d,fi in os.walk(path):
        f = fi
    i = 0
    for filename in f:
        if i%100 == 0:
            print(i)
        temp = cv2.imread(path+filename,0)
        temp = cv2.resize(temp,shape)
        cv2.imwrite(newpath+filename,temp)
        i += 1
    print()

def get_data(path):
    f = []
    for p,d,fi in os.walk(path+"/regTrain/"):
        f = fi
    X_train = cv2.imread(path+"/regTrain/"+f[0])
    X_train = X_train[np.newaxis,:]
    for filename in f[1:]:
        temp = cv2.imread(path+"/regTrain/"+filename)
        X_train = np.concatenate((X_train,temp[np.newaxis,:]),axis=0)
    for p,d,fi in os.walk(path+"/regTest/"):
        f = fi
    X_test = cv2.imread(path+"/regTest/"+f[0])
    X_test = X_test[np.newaxis,:]
    for filename in f[1:]:
        temp = cv2.imread(path+"/regTest/"+filename)
        X_test = np.concatenate((X_test,temp[np.newaxis,:]),axis=0)
    return X_train,X_test

def get_label(path):
    label = pd.read_csv(path)
    labels = np.array(label.loc[:,"Animal"])
    names = list(set(labels))
    names.sort()
    print("names",names)
    label_to_id = {label:indx for indx,label in enumerate(names)}
    print(label_to_id)
    Y_train = np.array([label_to_id[name] for name in labels])
    Y_train = CNN.convert_to_one_hot(Y_train,30)
    return Y_train.T

if __name__ == '__main__':
    #regularize("/home/vitrioil/Downloads/DLBeg/train/","/home/vitrioil/Downloads/DLBeg/regTrain/")
    #regularize("/home/vitrioil/Downloads/DLBeg/test/","/home/vitrioil/Downloads/DLBeg/regTest/")
    X_train,X_test = np.load("D:/HEC/X_train.npy"),np.load("D:/HEC/X_test.npy")#get_data("/home/vitrioil/Downloads/DLBeg/")
    Y_train = np.load("D:/HEC/Y_train.npy")
    split = 10000
    X_tr,X_ts,Y_tr,Y_ts = X_train[:split]/255,X_train[split:]/255,Y_train[:split],Y_train[split:]
#    print(np.argmax(Y_tr[-1]))
#    cv2.imshow("a",X_tr[-1])
#    cv2.waitKey()
#    cv2.destroyAllWindows()
    hec = HEC([X_tr,X_ts],[Y_tr,Y_ts],64,50,5e-6,"HEC",model_path="C:/Users/HiteshOza/Documents/TF_Tut/CNN/HEC/1532784650/model.ckpt-100156")
    hec.full_analysis(folder_name="HEC")
