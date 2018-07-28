import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image,ImageOps
import cv2
import os
import struct

def read(dataset = "training", path = "C:/Users/HiteshOza/Documents/gzip"):
    if dataset is "training":
        fname_img = os.path.join(path, 'emnist-byclass-train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'emnist-byclass-train-labels-idx1-ubyte')
    elif dataset is "testing":
        #pass
        fname_img = os.path.join(path, 'emnist-byclass-test-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'emnist-byclass-test-labels-idx1-ubyte')
    else:
        raise ValueError( "dataset must be 'testing' or 'training'")

    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
        print(len(lbl))
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8)      
        img = img.reshape(len(lbl),rows,cols)

    get_img = lambda idx: (lbl[idx], img[idx])
   # return get_img
    for i in range(len(lbl)):
        yield get_img(i)
def one_hot(X,depth):
    print(X.shape)
    b=np.zeros((X.shape[1],depth))
    b[np.arange(X.shape[1]),X]=1
    return b

def load_dataset():
    X=input_data.read_data_sets('MNIST_data',one_hot=True)

    X_train,X_test=X.train.images[:55000,:].reshape(55000,28,28,1),X.test.images[:10000,:].reshape(10000,28,28,1)
    Y_train,Y_test=X.train.labels[:55000,:],X.test.labels[:10000,:]
    print(np.max(X_train))

    classes = np.array([0,1,2,3,4,5,6,7,8,9])
    
    
    return  X_train, Y_train,X_test,Y_test,classes


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    
    m = X.shape[0]        
    mini_batches = []
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def convert_to_one_hot(Y, C):
    
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def forward_propagation_for_predict(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                      
    Z1 = tf.add(tf.matmul(W1, X), b1)                    
    A1 = tf.nn.relu(Z1)                                    
    Z2 = tf.add(tf.matmul(W2, A1), b2)                   
    A2 = tf.nn.relu(Z2)                                   
    Z3 = tf.add(tf.matmul(W3, A2), b3)                   
    
    return Z3

def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction


def create_placeholders(n_H0, n_W0, n_C0, n_y):

    X = tf.placeholder(tf.float32,shape=[None,n_H0,n_W0,n_C0])
    Y = tf.placeholder(tf.float32,shape=[None,n_y])
    
    return X, Y
def initialize_parameters():
 
    tf.set_random_seed(1) 
     
    tf.contrib.layers.xavier_initializer(seed=0)    

    W1 = tf.get_variable("W1",[3,3,1,8],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2",[5,5,8,16],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    #W3 = tf.get_variable("W3",[3,3,32,64],initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2}
     #             "W3": W3}
    
    return parameters


def forward_propagation(X, parameters):
    with tf.name_scope('conv'):
   
        W1 = parameters['W1']
        W2 = parameters['W2']
       # W3 = parameters['W3']
        
        Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding="SAME")
       
        A1 = tf.nn.relu(Z1)
        
        P1 = tf.nn.max_pool(A1,ksize=[1,7,7,1],strides=[1,6,6,1],padding="SAME")
       
        Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding="SAME")
    
        A2 = tf.nn.relu(Z2)
    
        P2 = tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,6,6,1],padding="SAME")

        #Z3 = tf.nn.conv2d(P2,W3,strides=[1,1,1,1],padding="SAME")
    
        #A3 = tf.nn.relu(Z3)
    
        #P3 = tf.nn.max_pool(A3,ksize=[1,3,3,1],strides=[1,1,1,1],padding="SAME")

        tf.summary.histogram('weigths_1',W1)
        tf.summary.histogram('z__1',Z1)
        tf.summary.histogram('act_1',A1)
        tf.summary.histogram('weigths_2',W2)
        tf.summary.histogram('z__2',Z2)
        tf.summary.histogram('act_2',A2)
         
          
    with tf.name_scope('fc'):

        P2 = tf.contrib.layers.flatten(P2)

        Z3 = tf.contrib.layers.fully_connected(P2,10,activation_fn=None)

    return Z3

def compute_cost(Z3, Y):
   
    with tf.name_scope('cost'):

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))

        tf.summary.scalar('cost',cost)
        
    return cost


def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def img_break(path,show=False):

    image=cv2.imread(path,1)
    #cv2.imshow(path)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1] 
    digitCnts = []
    #print(len(cnts))
    # loop over the digit area candidates
    for c in cnts:
    	# compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        #print((x,y,w,h))
    	# if the contour is sufficiently large, it must be a digit
        if(h>=15) :#and (w<=10 or w>=15): # (w<=10 and h>=15) or (h>=15 and w>=15):
            digitCnts.append(c)
 #   print(len(digitCnts))
    digitCnts = sort_contours(digitCnts,method="left-to-right")[0]
  #  digitCnts = sort_contours(digitCnts,method="top-to-bottom")[0]
    images,data={},{}
    i=0
    for c in digitCnts:
       i+=1
       (x, y, w, h) = cv2.boundingRect(c)
       roi = thresh[y:y + h, x:x + w]
       im = Image.fromarray(roi)
       im=im.resize((16,16),Image.ANTIALIAS)
       im=ImageOps.expand(im,(6,6,6,6),fill='black') 
       im=np.asarray(im)
       im_c=cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
       im_c=Image.fromarray(im_c)
       im_c.save('exp'+str(i)+'.png')
       im=Image.open('exp'+str(i)+'.png')
       im=im.resize((28,28),Image.ANTIALIAS)
       im.load()
       if show:
           im.show()
       data["d"+str(i)] = np.asarray(im)
       print(im.size,type(data["d"+str(i)]))
       data["d"+str(i)]=np.sum((data["d"+str(i)]),axis=-1)/3
  
       images["img"+str(i)]=im
       #data["d"+str(i)] = np.asarray(images["img"+str(i)])
       data["d"+str(i)]=(data["d"+str(i)].reshape(1,28,28,1))/255
         
    return images,data,len(digitCnts)
    




def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.04,
          num_epochs = 1, minibatch_size = 64, print_cost = True,num_break=2):
    
    logdir='C:/Users/HiteshOza/Documents'
    sess=tf.Session()
    ops.reset_default_graph()                     
    tf.set_random_seed(1)                           
    seed = 3                                       
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = []                                      
    

    X, Y = create_placeholders(n_H0,n_W0,n_C0,n_y)

    parameters = initialize_parameters()

    with tf.variable_scope('convolutional'):    
        Z3= forward_propagation(X, parameters)

    cost = compute_cost(Z3,Y)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    summ=tf.summary.merge_all()
    saver=tf.train.Saver()
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
    writer=tf.summary.FileWriter(logdir)
    writer.add_graph(sess.graph)
    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) 
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
              
                _ , temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})

                
                minibatch_cost += temp_cost / num_minibatches
            

            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))

            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)


        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()


        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        saver.save(sess,'C:/Users/HiteshOza/Documents/tmnst_model.ckpt')
        '''
        img = Image.open('C:/Users/HiteshOza/Documents/5.png')
        img = img.resize((28,28), Image.ANTIALIAS)
        img.load()
       # img.show()
        data = np.asarray(img)
        data=np.sum((data),axis=-1)
        data=(255-(data)/4)/255
        #data=data/255
    #    print(data)
        data=data.reshape(1,28*28)
        data=data.reshape(1,28,28,1)
        Y_A=np.array([[2]])
        Y_A=np.array([[0,0,0,0,0,0,0,1,0,0]])#one_hot(Y,10)
        print(accuracy.eval({X:data,Y:Y_A}))    
        print(predict_op.eval({X:data}))
        
        image,data,num_break=img_break(num_break,'C:/Users/HiteshOza/Documents/187.png')
        ans=[]
        for i in range(1,num_break+1):
            a=predict_op.eval({X:data["d"+str(i)]})
            ans.append(a[0])
        print(ans)
        '''
        return train_accuracy, test_accuracy, parameters
def check(X_train,num):
    tf.reset_default_graph()
    data = X_train
    sess=tf.Session()
    ops.reset_default_graph()                       
    tf.set_random_seed(1)                           
    #(m, n_H0, n_W0, n_C0) = X_train.shape             
    #n_y = Y_train.shape[1]                            
    X, Y = create_placeholders(28,28,1,10)
    parameters=initialize_parameters()
    with tf.variable_scope('convolutional'):
        Z3=forward_propagation(X,parameters)
    predict_op=tf.argmax(Z3,1)
    correct_prediction=tf.equal(predict_op,tf.argmax(Y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
    sess=tf.Session()
    saver=tf.train.Saver()
    saver.restore(sess,"C:/Users/HiteshOza/Documents/mnst_model.ckpt")
    #a=accuracy.eval({X:X_train,Y:Y_train})
    ans=[]
    for i in range(num):
        a=predict_op.eval({X:data["d"+str(i+1)]},session=sess)
        ans.append(a[0])            
    return ans
'''
#imgtr = read("training")
imgts = read("testing")
imgtr = read("testing")
X_train,X_test = np.array([i[1] for i in imgtr])/255,np.array([i[1] for i in imgtr])/255
Y_train,Y_test = np.array([i[0] for i in imgtr]),np.array([i[0] for i in imgts])
Y_test = convert_to_one_hot(Y_test.reshape(Y_test.shape[0],1),10)
#X_train, Y_train, X_test, Y_test, classes = load_dataset()
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_train,X_test = X_train[:81000,:,:,:],X_train[81001:,:,:,:]
Y_train,Y_test = Y_test[:,:81000],Y_test[:,81001:]
X_train, Y_train,X_test,Y_test,classes = load_dataset()
_, _, parameters = model(X_train, Y_train, X_test, Y_test)
'''
if __name__=='__main__':

    image,data,num_break=img_break('C:/Users/HiteshOza/Documents/187_1.png')
    print(data['d'+str(1)].shape)
    ans = check(data,num_break)
    print(ans)

   
