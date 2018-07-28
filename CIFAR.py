from CNN import CNN
import tensorflow as tf
import numpy as np
import cv2
class CIFAR(CNN):
    def __init__(self,Xs,Ys,batch_size,epoch,lr):
        if epoch%10 == 0:
            epoch += 1
        CNN.__init__(self,Xs,Ys,batch_size,epoch,lr,folder_name = "CIFAR",
                 search_timestamp = "1530719298",
                 model_path = "C:/Users/HiteshOza/Documents/TF_Tut/CNN/CIFAR/1530719298/model.ckpt-2000000"
                 ,training_session = False)
        self.assign_model()
    
    def assign_model(self):
        CNN.model = self.model
    
    def model(self,x_t,training = True):
        with tf.variable_scope("Conv1"):
            z = self.conv2d(x_t, 16, 3, 2)
            z = self.max_pool(z, 2, 2)
        with tf.variable_scope("Conv2"):
            z = self.conv2d(z, 32, 3, 2)
            z = self.max_pool(z, 2, 2)
        with tf.variable_scope("Flatten1"):
            z = tf.layers.flatten(z)
        with tf.variable_scope("Dense1"):
            z = tf.layers.dense(z,units=256,activation=tf.nn.relu)
            z = tf.layers.dropout(z,rate=0.15,training=training)
        with tf.variable_scope("Dense2"):
            logits = tf.layers.dense(z,units=self.Y_train.shape[1],name="logits")
        y = tf.nn.softmax(logits,name="y_hat")
        return y,logits

def bs(name,img):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

def get_data(path,test = False):
    
    batches = []
    if not test:
        for i in range(1,6):
            batches.append(unpickle(path+str(i)))
    else:
        batches.append(unpickle(path))
    X = batches[0][b"data"]
    Y = batches[0][b"labels"]
    for batch in batches[1:]:
        X = np.concatenate((X,batch[b"data"]),axis=0)
        Y = Y + batch[b"labels"]
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape((-1,3,32,32))
    X = X.transpose([0,3,2,1])
    X = np.array([np.rot90(i,3) for i in X])
    return X,CNN.convert_to_one_hot(Y,10).transpose([1,0])
    
if __name__ == "__main__":
    X_train,Y_train = get_data("C:/Users/HiteshOza/Documents/cifar-10-batches-py/data_batch_")
    X_test,Y_test = get_data("C:/Users/HiteshOza/Documents/cifar-10-batches-py/test_batch",True)
    cifar = CIFAR([X_train,X_test],[Y_train,Y_test],64,51,1e-4)
    cifar.full_analysis(train = False,mask = "001",adv_X = None,
                    adv_Y = None,adv_epoch = 50,eps = 0.0001,folder_name = "CIFAR")