import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time,os
import numpy as np
import psutil
import cv2
import colorama
import matplotlib.pyplot as plt
colorama.init()
class ansi:
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    END = "\033[0m"


class CNN:
    def __init__(self,Xs,Ys,batch_size,epoch,lr,
                 folder_name = "",
                 search_timestamp = "1530634669",
                 model_path = "CNN/MNIST/1530634669/model.ckpt-2750000"
                 ,training_session = True):

        self.X_train,self.X_test = Xs
        self.Y_train,self.Y_test = Ys
        self.batch_size = batch_size
        self.epoch = epoch + 1 if epoch % 10 == 0 else epoch
        self.lr = lr
        self.training_session = training_session
        self.search_timestamp = search_timestamp
        self.timestamp =  str(int(time.time())) if self.training_session else self.search_timestamp
        self.folder_name = folder_name
        self.logdir="CNN/"+self.folder_name+"/"+self.timestamp
        self.model_path = model_path #if not training_session else "CNN/"+self.folder_name+"/"+self.timestamp+"/model.ckpt-"+str(epoch*self.X_train.shape[0])


    def check_usage(self):
        return (95 < psutil.virtual_memory().percent)

    def convert_to_one_hot(Y, C):

        Y = np.eye(C)[Y.reshape(-1)].T
        return Y

    def fsgm(self,model,x,y,epoch,eps = 0.01,min_val = 0,max_val = 1):
        """
        Check against Adversarial Attacks
        """
        x_adv = tf.identity(x)
        #x_adv = tf.reshape(x_adv,[*x_adv.shape[1:]])
        check_fn = tf.sign
        loss_fn = tf.nn.softmax_cross_entropy_with_logits_v2
        def cond(xadv,i):
            return tf.less(i,epoch)

        def body(xadv,i):
            if self.check_usage(): i = epoch
            yhat,logits = model(xadv)
            loss = loss_fn(logits=logits,labels=y)
            grad = tf.gradients(loss,xadv)
            xadv = tf.stop_gradient(xadv+eps*check_fn(grad))
            xadv = tf.clip_by_value(xadv,min_val,max_val)
            xadv = tf.reshape(xadv,[-1,*self.X_train.shape[1:]])
            return xadv,i+1

        xadv,_ = tf.while_loop(cond,body,(x_adv,0),back_prop=False)
        return tf.reshape(xadv,[-1,*self.X_train.shape[1:]])

    def conv2d(self,input_data,filter_size,kernel_size,strides,activation = tf.nn.relu,padding="same"):
        """ Uses tf.layers.conv2d
        Input Input_Data Tensor

        Input      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).

        Input       kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.

        Input       strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.

        [Opt]Input activation (string)

        [Opt]Input padding: One of `"valid"` or `"same"` (case-insensitive).
        inputs: Tensor input.

        """
        return tf.layers.conv2d(
                input_data,
                filter_size,
                kernel_size,
                activation = activation,
                padding=padding,
                kernel_initializer = tf.contrib.layers.xavier_initializer()
                )

    def max_pool(self,input_data,kernel_size,strides_size,padding = "same"):
        """
        Inputs: input: The tensor over which to pool. Must have rank 4.

        Inputs: kernel: pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
        specifying the size of the pooling window.
        Can be a single integer to specify the same value for
        all spatial dimensions.

        Inputs: strides: An integer or tuple/list of 2 integers,
        specifying the strides of the pooling operation.
        Can be a single integer to specify the same value for
        all spatial dimensions

        [Opt]Inputs: padding (string) "same"|"valid"
        """
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

    def transpose_conv2d(self,input_data,fil,output_space,strides,padding="SAME"):
        return tf.nn.conv2d_transpose(
                                        input_data,
                                        fil,
                                        output_space,
                                        strides,
                                        padding=padding
                                    )

    def minibatch(self,x = None,y = None,batch_size = None,training = True):
        if batch_size is None:
            batch_size = self.batch_size
        if x is None or y is None:
            (x,y) = (self.X_train,self.Y_train) if training else (self.X_test,self.Y_test)
        size_batch = x.shape[0]
        start = 0
        end = start + batch_size
        total_batch = size_batch//batch_size +  (size_batch % batch_size != 0)
        for i in range(total_batch):
            if end<size_batch:
                yield (x[start:end,:,:,:],y[start:end,:],i)
            else:
                yield (x[start:,:,:,:],y[start:,:],i)
            start = end
            end += batch_size

    def model(self,x_t,training = True):
        with tf.variable_scope("Conv1"):
            z = self.conv2d(x_t, 16, 3, 2,activation=tf.nn.relu)
            z = self.max_pool(z, 2, 2)
        with tf.variable_scope("Conv2"):
            z = self.conv2d(z, 32, 3, 2,activation=tf.nn.relu)
            z = self.max_pool(z, 2, 2)
        with tf.variable_scope("Flatten1"):
            z = tf.layers.flatten(z)
        with tf.variable_scope("Dense1"):
            z = tf.layers.dense(z,units=256,activation=tf.nn.relu)
            z = tf.layers.dropout(z,rate=0.15,training=training)
        with tf.variable_scope("Dense1"):
            logits = tf.layers.dense(z,units=self.Y_train.shape[1],name="logits")
        y = tf.nn.softmax(logits,name="y_hat")
        return y,logits

    def train(self):
        tf.reset_default_graph()
        X_t = tf.placeholder("float64",[None,*self.X_train.shape[1:]])
        Y_t = tf.placeholder("float64",[None,*self.Y_train.shape[1:]])

        out,logits = self.model(X_t)

        with tf.name_scope("accuracy"):
            equal = tf.equal(tf.argmax(out,axis=1),tf.argmax(Y_t,axis=1))
            accuracy = tf.reduce_mean(tf.cast(equal,tf.float64),name="accuracy")
        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_t,
                                                                       logits = logits)
            cost = tf.reduce_mean(cross_entropy,name="loss")
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        writer=tf.summary.FileWriter(self.logdir)
        writer.add_graph(sess.graph)
        sess.close()
        saver = tf.train.Saver()
        tf.summary.scalar("cost",cost)
        tf.summary.scalar("accuracy",accuracy)
        if self.X_test is not None and self.Y_test is not None:
            evalX,evalY = self.X_test,self.Y_test
        else:
            evalX,evalY = self.X_train[:int(0.3*self.X_train.shape[0])],self.Y_train[:int(0.3*self.Y_train.shape[0])]
        with tf.Session() as sess:
            sess.run(init)
            summ = tf.summary.merge_all()
            costs,accuracies = [],[]
            saver.restore(sess,self.model_path)
            for epoch in range(self.epoch):
                epoch_loss = 0
                if self.check_usage():
                    print("{} Memory Usage very high:! {} {} {} {}".format(ansi.RED,ansi.END,ansi.BLUE,psutil.virtual_memory().percent,ansi.END))
                    print("{} Terminating {} epoch {}".format(ansi.RED,ansi.END,epoch))
                    break
                for minibatch in self.minibatch():
                    x,y,i = minibatch
                    step = epoch*self.X_train.shape[0] + i
                    _,c,s = sess.run([optimizer,cost,summ],
                                feed_dict={X_t:x,Y_t:y})
                    if self.check_usage():
                        print("{} Memory Usage very high:! {} {} {} {}".format(ansi.RED,ansi.END,ansi.BLUE,psutil.virtual_memory().percent,ansi.END))
                        print("{} Terminating {} iteration {}".format(ansi.RED,ansi.END,i))
                        break
                        writer.add_summary(s,step)
                    epoch_loss += c
                    if i % 5 == 0:
                        print("Epoch {},Epoch Loss {}".format(epoch,epoch_loss))
                        print("Current iteration {} loss {}".format(i,c))
                        print('*'*10)
                if (epoch % 5) == 0 and epoch != 0:
                    print("Model saved!")
                    saver.save(sess,os.path.join(self.logdir,"model.ckpt"),step)
                print("Epoch {} completed!".format(epoch))
                
                accuracyEval,indx = 0,0
                if evalX.shape[0]*evalX.shape[1]*evalX.shape[2]*evalX.shape[3] < 1e6:
                    accuracyEval = accuracy.eval(
                            feed_dict={X_t:evalX,Y_t:evalY})
                else:                
                    for minibatch in self.minibatch(x=evalX,y=evalY,batch_size = 5):
                        x,y,indx = minibatch
                        accuracyEval += accuracy.eval(feed_dict={X_t:x,Y_t:y})
                    accuracyEval /= (indx + 1)
                print(f"Accuracy :{ansi.GREEN} {accuracyEval} {ansi.END}")
                print('='*10)
                
                costs.append(epoch_loss)
                accuracies.append(accuracyEval)
            f = plt.figure()
            s1 = f.subplot(211)
            s2 = f.subplot(212)
            s1.set_title("Costs")
            s2.set_title("Accuracies")
            s1.plot(costs)
            s2.plot(accuracies)
            plt.plot()

    def load_model(self):
        tf.reset_default_graph()
        X_pl = tf.placeholder("float64", [None, *self.X_train.shape[1:]])
        sess = tf.Session()
        output,logits = self.model(X_pl)
        saver=tf.train.Saver()
        saver.restore(sess,self.model_path)

        def predict(img):
            img = img.reshape((-1,*self.X_train.shape[1:]))
            feed_dict = {X_pl: img}
            classification = output.eval(feed_dict,session = sess)#
            out_class = np.argmax(classification,1)

            return out_class

        def close():
            sess.close()
        return predict,close

    def prepare_adversarial(self,X,Y,folder_name = "",epoch = 20,eps = 0.001):
        X_Adv_Location = "CNN/"+folder_name+"/"+self.timestamp+str(eps)[2:]+str(epoch)+".npy"
        try:
            X_Final_adv = np.empty_like(X) if self.training_session else np.load(X_Adv_Location)
        except Exception as e:
            print(e)
            X_Final_adv = np.empty_like(X)
        indices_Location = "CNN/"+folder_name+"/test_indices"+self.timestamp+".npy"

        def train_adversarial_attack(X = X,Y = Y,epoch = epoch,eps = eps):
            tf.reset_default_graph()
            X_t = tf.placeholder("float64",[None,*self.X_train.shape[1:]])
            Y_t = tf.placeholder("float64",[None,*self.Y_train.shape[1:]])
            X_adv_t = self.fsgm(self.model,X_t,Y_t,epoch,eps = eps)
            start,end = 0,self.batch_size
            sess = tf.Session()
            saver=tf.train.Saver()
            saver.restore(sess,self.model_path)
            for minibatch in self.minibatch(x = X,y = Y,training=False):
                x_mini,y_mini,i = minibatch
                X_adv_val = sess.run([X_adv_t],feed_dict={X_t:x_mini,Y_t:y_mini})
                X_adv_val = np.array(X_adv_val).reshape(-1,*self.X_train.shape[1:])
                X_Final_adv[start:end,:,:,:] = X_adv_val
                start = end
                end += x_mini.shape[0]
                print("\r%s" % "Minibatch number {}/{}".format(
                        i,X.shape[0]//self.batch_size+(X.shape[0]%self.batch_size != 0)-1),end="\r")
            self.evaluate(X_Final_adv,Y)
            sess.close()
            print(np.sum(X_Final_adv-X))
            np.save(X_Adv_Location,X_Final_adv)
            cv2.imshow("Real",X[-1,:,:,:])
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.imshow("Fake",X_Final_adv[-1,:,:,:])
            cv2.waitKey()
            cv2.destroyAllWindows()

        def find_adversarial():
            predict_fn,close = self.load_model()
            X_Adv = np.load(X_Adv_Location)
            indices = []
            for indx in range(self.X_test.shape[0]):
                  real_pred = predict_fn(self.X_test[indx,:,:,:])[0]
                  fake_pred = predict_fn(X_Adv[indx,:,:,:])[0]
                  if int(fake_pred) != int(real_pred):
                      #print("\r%s"%"Found an adversarial example",end="")
                      print("\r%s"%"Predicted {} Adversarial Prediction {} Actual {} ".format(
                              real_pred,fake_pred,np.argmax(self.Y_test[indx,:])),end="")
                      indices.append(indx)
            np.save(indices_Location,np.array(indices))
            close()

        def show_sample(start = 0,end = 5):
            predict_fn,close = self.load_model()
            sample = np.load(indices_Location)
            for i in sample[start:end]:
                 print(predict_fn(self.X_test[i,:,:,:])[0])
                 cv2.imshow("Real",self.X_test[i,:,:,:])
                 cv2.waitKey()
                 cv2.destroyAllWindows()
                 print(predict_fn(X_Final_adv[i,:,:,:])[0])
                 cv2.imshow("Fake",X_Final_adv[i,:,:,:])
                 cv2.waitKey()
                 cv2.destroyAllWindows()
            close()

        return train_adversarial_attack,find_adversarial,show_sample

    def evaluate(self,x,y):
        tf.reset_default_graph()
        X_t = tf.placeholder("float64",[None,*self.X_train.shape[1:]])
        Y_t = tf.placeholder("float64",[None,*self.Y_train.shape[1:]])
        out,logits = self.model(X_t)
        with tf.name_scope("accuracy"):
            equal = tf.equal(tf.argmax(out,axis=1),tf.argmax(Y_t,axis=1))
            accuracy = tf.reduce_mean(tf.cast(equal,tf.float64),name="accuracy")
        feed_dict = {X_t: x,Y_t:y}
        sess = tf.Session()
        saver=tf.train.Saver()
        saver.restore(sess,self.model_path)
        accuracy = accuracy.eval(session = sess,feed_dict=feed_dict)
        print("Accuracy: {} {} {}".format(ansi.BLUE,
                  accuracy,ansi.END))
        sess.close()
        return accuracy

    def full_analysis(self,train = True,mask = "111",adv_X = None,
                    adv_Y = None,adv_epoch = 20,eps = 0.001,folder_name = ""):
        if adv_X is None or adv_Y is None:
            adv_X,adv_Y = self.X_test,self.Y_test
        if train: self.train()
        train_adversarial_attack,find_adversarial,show_sample = self.prepare_adversarial(
                adv_X,adv_Y,folder_name = folder_name,epoch = adv_epoch,eps = eps)
        if mask[0] == '1': train_adversarial_attack()
        if mask[1] == '1': find_adversarial()
        if mask[2] == '1': show_sample()

if __name__ == "__main__":
    X = input_data.read_data_sets("/tmp/data/", one_hot = True)
    training_size = 55000
    testing_size = 10000
    X_train,X_test=X.train.images[:training_size,:].reshape(training_size,28,28,1),X.test.images[:testing_size,:].reshape(testing_size,28,28,1)
    Y_train,Y_test=X.train.labels[:training_size,:],X.test.labels[:testing_size,:]
    cnn = CNN((X_train,X_test),(Y_train,Y_test),64,51,9e-6)
    #cnn.train()
    #cnn.evaluate(X_test,Y_test)
    cnn.full_analysis(train=False,mask="001",adv_epoch = 50,eps=0.0002)
