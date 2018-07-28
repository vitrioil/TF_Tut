from GAN import *
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
logdir="C:/Users/HiteshOza/Documents/TF_Tut/GAN/1529861776/model.ckpt-700"


def evaluate(gan,inputX):
    
#    tf.reset_default_graph()
    noise_input_t,output = gan.sample_generator(inputX.shape[0])    
    feed_dict = {noise_input_t: inputX}
    sess = tf.Session(graph=g)
    saver=tf.train.Saver()
    saver.restore(sess,logdir)
    
    output = output.eval(feed_dict,session = sess)
    output = output.astype(np.uint8).squeeze()
    print(output.shape)
    output = output.reshape(28,28,-1)
    cv2.imshow("out",output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sess.close()
if __name__ == "__main__":
    g = tf.Graph()
    with g.as_default():
        gan = GAN(16)
        inputX = np.random.randn(1,28*28)
        evaluate(gan,inputX) 