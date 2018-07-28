from CNN import CNN,ansi
import tensorflow as tf
import numpy as np
import cv2
import os
NUM_CLASSES = 2
class SEGMENT(CNN):
    def __init__(self,Xs,Ys,batch_size,epoch,lr):
        if epoch%10 == 0:
            epoch += 1
        CNN.__init__(self,Xs,Ys,batch_size,epoch,lr,folder_name = "CIFAR",
                 search_timestamp = "1530719298",
                 model_path = "C:/Users/HiteshOza/Documents/TF_Tut/CNN/SEMSEG/1530719298/model.ckpt-2000000"
                 ,training_session = True)
        self.assign_model()    
    def load_vgg(self,sess, vgg_path):
      
      # load the model and weights
      model = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    
      # Get Tensors to be returned from graph
      graph = tf.get_default_graph()
      image_input = graph.get_tensor_by_name('image_input:0')
      keep_prob = graph.get_tensor_by_name('keep_prob:0')
      layer3 = graph.get_tensor_by_name('layer3_out:0')
      layer4 = graph.get_tensor_by_name('layer4_out:0')
      layer7 = graph.get_tensor_by_name('layer7_out:0')
      print(image_input.get_shape().as_list())
      return image_input, keep_prob, layer3, layer4, layer7
    
    def assign_model(self):
        CNN.model = self.model
    
    def model(self,x_t,layers,training = True):
#        layer3, layer4, layer7 = layers
#    
#        fcn8 = self.conv2d(layer7, NUM_CLASSES, 1)
#        fcn9 = self.transpose_conv2d(fcn8, 
#                                     layer4.get_shape().as_list()[-1],4, (2, 2),func = tf.layers)
#        
#        fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")
#    
#        fcn10 = self.transpose_conv2d(fcn9_skip_connected,
#                                      layer3.get_shape().as_list()[-1],4,(2, 2),func = tf.layers)
#    
#        fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")
#    
#        fcn11 = self.transpose_conv2d(fcn10_skip_connected,
#                                      NUM_CLASSES,16,(8, 8),func = tf.layers)
#        
#        logits = tf.reshape(fcn11,[-1,NUM_CLASSES])
#        return fcn11,logits
        # Use a shorter variable name for simplicity
        layer3, layer4, layer7 = layers
        
        # Apply 1x1 convolution in place of fully connected layer
        fcn8 = tf.layers.conv2d(layer7, filters=NUM_CLASSES, kernel_size=1, name="fcn8")
        filter9 = tf.Variable(tf.random_normal(shape=[4,12,512,2],dtype=tf.float32))
        
        # Upsample fcn8 with size depth=(4096?) to match size of layer 4 so that we can add skip connection with 4th layer
        fcn9 = tf.nn.conv2d_transpose(fcn8, filter9,
        output_shape=tf.stack([16,4,12,512]), strides=[1,2,2,1], padding='SAME', name="fcn9")
    
        # Add a skip connection between current final layer fcn8 and 4th layer
        fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")
    
        # Upsample again
        fcn10 = tf.nn.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
        kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")
    
        # Add skip connection
        fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")
    
        # Upsample again
        fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=NUM_CLASSES,
        kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")
        
        logits = tf.reshape(fcn11,[-1,NUM_CLASSES])
        return fcn11,logits
    
    
    def trainSeg(self):
        tf.reset_default_graph()
        X_t = tf.placeholder("float64",[None,*self.X_train.shape[1:]])
        Y_t = tf.placeholder("float64",[None,*self.Y_train.shape[1:]])
        keep_prob = tf.placeholder(tf.float64)
        
        sess = tf.Session()
        
        writer=tf.summary.FileWriter(self.logdir)
        writer.add_graph(sess.graph)
        sess.close()
        with tf.Session() as sess:
            X_t,keep_prob,layer3,layer4,layer7 = self.load_vgg(sess,"C:/Users/HiteshOza/Documents/TF_Tut/CNN/SEMSEG/vgg")
            out,logits = self.model(X_t,(layer3,layer4,layer7))
            
            with tf.name_scope("accuracy"):
                equal = tf.equal(tf.argmax(out,axis=1),tf.argmax(tf.reshape(Y_t,[-1,NUM_CLASSES]),axis=1))
                accuracy = tf.reduce_mean(tf.cast(equal,tf.float64),name="accuracy")
            with tf.name_scope("loss"):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.reshape(Y_t,[-1,NUM_CLASSES]),
                                                                           logits = logits)
                cost = tf.reduce_mean(cross_entropy,name="loss")
            with tf.name_scope("optimizer"):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)
            saver = tf.train.Saver()
            tf.summary.scalar("cost",cost)
            tf.summary.scalar("accuracy",accuracy)
            init = tf.global_variables_initializer()
            sess.run(init)
            summ = tf.summary.merge_all()

            for epoch in range(self.epoch):
                epoch_loss = 0
                if self.check_usage():
                    print("{} CPU Usage very high:! {} {} {} {}".format(ansi.RED,ansi.END,ansi.BLUE,psutil.virtual_memory().percent,ansi.END))
                    print("{} Terminating {} epoch {}".format(ansi.RED,ansi.END,epoch))
                    break
                for minibatch in self.minibatch():
                    x,y,i = minibatch
                    step = epoch*self.X_train.shape[0] + i
                    _,c,s = sess.run([optimizer,cost,summ],
                                feed_dict={X_t:x,Y_t:y,keep_prob:0.75})
                    if self.check_usage():
                        print("{} CPU Usage very high:! {} {} {} {}".format(ansi.RED,ansi.END,ansi.BLUE,psutil.virtual_memory().percent,ansi.END))
                        print("{} Terminating {} iteration {}".format(ansi.RED,ansi.END,i))
                        break
                        writer.add_summary(s,step)
                    epoch_loss += c
                    if i % 1000 == 0:
                        print("Epoch {},Epoch Loss {}".format(epoch,epoch_loss))
                        print("Current iteration loss {}".format(c))
                        if (epoch % 10) == 0:
                            saver.save(sess,os.path.join(self.logdir,"model.ckpt"),step)
                            print("Model saved!")
                print("Epoch {} completed!".format(epoch))
                print("Accuracy :{} {} {}".format(ansi.GREEN,accuracy.eval(
                        feed_dict={X_t:self.X_test,Y_t:self.Y_test}),ansi.END))
            print("Testing accuracy: {} {} {}".format(ansi.BLUE,
                  accuracy.eval(feed_dict={X_t:self.X_test,Y_t:self.Y_test}),ansi.END))
            
def bs(name,img):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def make_data(path,train = True):
    file,X,Y = [],[],[]
    for _,dirname,file in os.walk(path):
        file = file
    X = cv2.imread(os.path.join(path+"consistent_image_2",file[0]))
    X = X[np.newaxis,:]
    print(X.shape)
    for file_name in file[1:]:
        img = cv2.imread(os.path.join(path+"consistent_image_2",file_name))
        X = np.concatenate((X,img[np.newaxis,:]),axis=0)
    print(X.shape)
    np.save(path+"consistent_image_2.npy",X)
    for _,dirname,file in os.walk(path+"consistent_gt_image_2"):
        file = file
    Y = cv2.imread(os.path.join(path+"consistent_gt_image_2",file[0]))
    Y = Y[np.newaxis,:]
    for file_name in file[1:]:
        img = cv2.imread(os.path.join(path+"consistent_gt_image_2",file_name))
        Y = np.concatenate((Y,img[np.newaxis,:]),axis=0)
    print(Y.shape)
    np.save(path+"consistent_gt_image_2.npy",Y)
    
if __name__ == "__main__":
    path = "C:/Users/HiteshOza/Documents/TF_Tut/CNN/SEMSEG/data_road/training/"
    X,Y = np.load(path+"consistent_image_2.npy"),np.load(path+"consistent_gt_image_2.npy")
    Y = np.concatenate((Y[:97],Y[192:]),axis=0)
    print(X.shape,Y.shape)
    seg = SEGMENT([X[:200],X[200:]],[Y[:200],Y[200:]],16,40,0.001)
    seg.trainSeg()
    seg.full_analysis(train = False,mask = "111",adv_X = None,
                    adv_Y = None,adv_epoch = 50,eps = 0.0001,folder_name = "SEMSEG")
#    make_data(path)