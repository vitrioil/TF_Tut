import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
import os
import psutil
import numpy as np
import cv2
# In[1]:
class GAN: 
    new_folder = str(int(time.time()))
    mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
    def __init__(self,batch_size):
#        self.X = X
#        self.Y = Y
        self.batch_size = batch_size
        
        ## Constants
        self.dimW1 = 256
        self.dimW2 = 128
        self.dimW3 = 64
        self.dimW4 = 1
        
        
        ## Variables needed
        self.helper_tensors = {}
        names = ["filter_gen1","filter_gen2","filter_gen3","filter_gen4"]
        shapes = [
                  [28*28,self.dimW1*4*4],
                  [3,3,self.dimW2,self.dimW1],
                  [3,3,self.dimW3,self.dimW2],
                  [3,3,self.dimW4,self.dimW3]
                 ]
        for indx,n in enumerate(names):
            self.helper_tensors[n] = tf.Variable(tf.truncated_normal(shapes[indx]))
        

    def conv2d(self,input_data,filter_size,kernel_size,activation = None,padding="same"):
        return tf.layers.conv2d(
                input_data,
                filter_size,
                kernel_size,
                activation = activation,
                padding=padding
                )
        
    def max_pool(self,input_data,strides_size,kernel_size,padding = "same"):
        return tf.layers.max_pooling2d(
                                        input_data,
                                        kernel_size,
                                        strides_size,
                                        padding = padding
                                       )
        
    def leaky_relu(self,input_data,alpha = 0.1):
        return tf.maximum(input_data,input_data*alpha)
    
    def batch_norm(self,input_data,training):
        return tf.layers.batch_normalization(input_data,training = training)
        
    def transpose_conv2d(self,input_data, output_space,fil,strides,padding="SAME"):
        return tf.nn.conv2d_transpose(
                                        input_data,fil,output_space,
                                        strides, padding=padding
                                    )
# In[2]:                                        
    def generator(self,noise_input,output_dimension,training = True):
        
        
# =============================================================================
#         fc1 = tf.layers.dense(noise_input,product)
#         
#         fc1 = tf.reshape(fc1,(-1,vector_shape[0],vector_shape[1],vector_shape[2]))
#         fc1 = self.batch_norm(fc1,training)
#         fc1 = tf.nn.relu(fc1)
# =============================================================================
        fc1 = tf.nn.relu(tf.matmul(noise_input,self.helper_tensors["filter_gen1"]))
        fc1 = tf.reshape(fc1,[self.batch_size,4,4,self.dimW1])
        
                                            #input,output_shape,filter,strides,padding
        output_shape_2 = [self.batch_size,7,7,self.dimW2]
        transpose_c1 = self.transpose_conv2d(fc1,output_shape_2,
                                             self.helper_tensors["filter_gen2"],[1,2,2,1],padding="SAME")
        #transpose_c1 = self.batch_norm(transpose_c1,training)
        transpose_c1 = tf.nn.relu(transpose_c1)
                                            #input,output_shape,filter,strides,padding
        output_shape_3 = [self.batch_size,14,14,self.dimW3]
        transpose_c2 = self.transpose_conv2d(transpose_c1,output_shape_3,
                                             self.helper_tensors["filter_gen3"],[1,2,2,1],padding="SAME")
        #transpose_c2 = self.batch_norm(transpose_c2,training)
        transpose_c2 = tf.nn.relu(transpose_c2)
        
        output_shape_4 = [self.batch_size,28,28,self.dimW4]
        logits = self.transpose_conv2d(transpose_c2, output_shape_4,
                                       self.helper_tensors["filter_gen4"],[1,2,2,1],padding="SAME")

        out = tf.tanh(logits)
        return out
    
    def sample_generator(self,batch_size):
        training = True
        noise_input = tf.placeholder(tf.float32,[batch_size,28*28])
        fc1 = tf.nn.relu(tf.matmul(noise_input,self.helper_tensors["filter_gen1"]))
        fc1 = tf.reshape(fc1,[batch_size,4,4,self.dimW1])
        
                                            #input,output_shape,filter,strides,padding
        output_shape_2 = [batch_size,7,7,self.dimW2]
        transpose_c1 = self.transpose_conv2d(fc1,output_shape_2,
                                             self.helper_tensors["filter_gen2"],[1,2,2,1],padding="SAME")
        #transpose_c1 = self.batch_norm(transpose_c1,training)
        transpose_c1 = tf.nn.relu(transpose_c1)
                                            #input,output_shape,filter,strides,padding
        output_shape_3 = [batch_size,14,14,self.dimW3]
        transpose_c2 = self.transpose_conv2d(transpose_c1,output_shape_3,
                                             self.helper_tensors["filter_gen3"],[1,2,2,1],padding="SAME")
        #transpose_c2 = self.batch_norm(transpose_c2,training)
        transpose_c2 = tf.nn.relu(transpose_c2)
        
        output_shape_4 = [batch_size,28,28,self.dimW4]
        logits = self.transpose_conv2d(transpose_c2, output_shape_4,
                                       self.helper_tensors["filter_gen4"],[1,2,2,1],padding="SAME")

        out = tf.tanh(logits)
        return noise_input,out
# In[3]:                                                            ##alpha is for leaky relu  
    def discriminator(self,input_data,vector_shape = [4,4,256],alpha = 0.1,training = True):       
                            # input,filter,kernel(,optional activation)
        conv1 = self.conv2d(input_data,64,5)
        conv1 = self.batch_norm(conv1,training)
        conv1 = self.leaky_relu(conv1,alpha)
        
        conv2 = self.conv2d(conv1,128,5)
        conv2 = self.batch_norm(conv2,training)
        conv2 = self.leaky_relu(conv2,alpha)
        
        conv3 = self.conv2d(conv2,256,5)
        conv3 = self.batch_norm(conv3,training)
        conv3 = self.leaky_relu(conv3,alpha)
        
        flatten = tf.reshape(conv3,(-1,vector_shape[0],vector_shape[1],vector_shape[2]))
        logits = tf.layers.dense(flatten,1)
        
        out = tf.nn.sigmoid(logits)
        return logits,out
# In[4]:        
    def loss(self,real_input,noise_input,output_dimension,alpha = 0.1,smooth=0.1):
        
        gen_out = self.generator(noise_input,output_dimension)
        disc_logits_real,disc_out_real = self.discriminator(real_input)
        
        disc_logits_fake,disc_out_fake = self.discriminator(gen_out)
        
        disc_real_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                        logits = disc_logits_real,labels = tf.ones_like(disc_logits_real)*(1-smooth)
                        )
                )
        disc_fake_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                        logits = disc_logits_fake,labels = tf.zeros_like(disc_out_fake)
                        )
                )
        gen_loss  = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                        logits = disc_logits_fake,
                        labels = tf.ones_like(disc_out_fake)
                        )
                )
        disc_loss = disc_real_loss + disc_fake_loss
        
        return gen_loss,disc_loss
# In[5]:    
    def model(self,X,noise_X,output_dimension):
        lr_d,lr_g = 1e-3,1e-3
        gen_loss,disc_loss = self.loss(X,noise_X,output_dimension)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate = lr_d).minimize(gen_loss)
        optimizer_gen = tf.train.AdamOptimizer(learning_rate = lr_g).minimize(disc_loss)
        
        
        epoch = 10
        logdir='C:/Users/HiteshOza/Documents/TF_Tut/GAN/'+self.new_folder
        sess = tf.Session()
        writer=tf.summary.FileWriter(logdir)
        writer.add_graph(sess.graph) 
        sess.close()
        saver = tf.train.Saver()
        vec_dimension = output_dimension[0]*output_dimension[1]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            summ = tf.summary.merge_all() 
            vis_batch_size = 1
            noise_input_t,output_sample = self.sample_generator(vis_batch_size)
            noise_input_t_sample = np.random.randn(vis_batch_size,vec_dimension)
            for ep in range(epoch):
           #     time.sleep(0.1)
                if(psutil.virtual_memory().percent>95):
                    print("Memory usage high: {}\n Killing the program!".format(psutil.virtual_memory().percent))
                    break
                ep_loss = 0
                for _ in range(self.mnist.train.num_examples//self.batch_size):
                   # time.sleep(0.1)
                    if(psutil.virtual_memory().percent>95):
                        print("Memory usage high: {}\n Killing the program!".format(psutil.virtual_memory().percent))
                        break
                    ep_x,ep_y = self.mnist.train.next_batch(self.batch_size)
                    ep_x = ep_x.reshape((-1,28,28,1))
                    ep_n_x = np.random.randn(ep_x.shape[0],vec_dimension)
                    __1,__2,g_c,d_c = sess.run(
                            [optimizer_disc,optimizer_gen,gen_loss,disc_loss],
                            feed_dict={X:ep_x,noise_X:ep_n_x}
                            )
                    step = ep*self.mnist.train.num_examples//self.batch_size+_
                   # writer.add_summary(s,step)
                    
                    it_loss = (g_c+d_c)/self.batch_size
                    ep_loss += it_loss
                    if _%2 == 0:
                        print("Epoch",ep,'/',epoch,"Current epoch loss:",ep_loss)
                        print("Current iteration loss",it_loss)
                        print("Generator loss:",g_c/self.batch_size)
                        print("Discriminator loss:",d_c/self.batch_size)
                        gen_sample = sess.run(output_sample,feed_dict={noise_input_t:noise_input_t_sample})
                        gen_sample = gen_sample.reshape(28,28,-1)
                        cv2.imwrite("C:/Users/HiteshOza/Documents/TF_Tut/GAN/Sample/"+self.new_folder+str(epoch)+str(_)+".png",gen_sample)
                    if _%100==0:
                        try:
                            #One save can take 2 GB of space!!
                            saver.save(sess, os.path.join(logdir, "model.ckpt"), step)
                        except Exception as e:
                            print(str(e))
                        else:
                            print("Saved")
# In[6]:            
if __name__ == '__main__':
    gan = GAN(16)
    X = tf.placeholder('float',[None,28,28,1])
    noise_X = tf.placeholder('float',[None,28*28])
    gan.model(X,noise_X,(28,28))