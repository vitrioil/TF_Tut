import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os,time

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
tf.reset_default_graph()
layers = [28*28,300,500,200,50,10]
X = tf.placeholder('float', [None, 784])
Y = tf.placeholder('float')
batch_size = 128
new_folder = str(int(time.time()))
def neural_net(X,layers = [28*28,300,500,200,50,10]):
    n_layers = len(layers)
    Weights = {}
    Biases = {}
    Layers = {}
    with tf.name_scope('Parameters'):
        for i in range(1,n_layers):
            Weights['w'+str(i)] = tf.Variable(tf.random_normal([layers[i-1],layers[i]]))
            Biases['b'+str(i)] = tf.Variable(tf.random_normal([layers[i]]))
        Layers['l1'] = tf.add(tf.matmul(X,Weights['w1']),Biases['b1'])
    with tf.name_scope('Layers'):
        for i in range(2,n_layers-1):
            Layers['l'+str(i)] = tf.add(tf.matmul(Layers['l'+str(i-1)],Weights['w'+str(i)]),Biases['b'+str(i)])
            Layers['l'+str(i)] = tf.nn.relu(Layers['l'+str(i)])
        Layers['l'+str(n_layers-1)] = tf.add(tf.matmul(Layers['l'+str(n_layers-2)],Weights['w'+str(n_layers-1)],),Biases['b'+str(n_layers-1)])
    return Layers['l'+str(n_layers-1)] 
def train_net(X):
    output = neural_net(X)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.00111).minimize(cost)
    
    epoch = 10
    logdir='C:/Users/HiteshOza/Documents/TF_Tut/TF_1/'+new_folder
    sess = tf.Session()
    writer=tf.summary.FileWriter(logdir)
    writer.add_graph(sess.graph) 
    sess.close()
    saver = tf.train.Saver()
    with tf.name_scope('Accuracy'):    
            correct = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    tf.summary.scalar("Cost",cost)
    tf.summary.scalar("Accuracy",accuracy)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summ = tf.summary.merge_all()
        for ep in range(epoch):
            ep_loss = 0
            for _ in range(mnist.train.num_examples//batch_size):
                ep_x,ep_y = mnist.train.next_batch(batch_size)
                __,c,s = sess.run([optimizer,cost,summ],feed_dict={X:ep_x,Y:ep_y})
                step = ep*mnist.train.num_examples//batch_size+_
                writer.add_summary(s,step)
                
                ep_loss += c/batch_size
                if _%50 == 0:
                    saver.save(sess, os.path.join(logdir, "model.ckpt"), step)
            print('Epoch',ep,'/',epoch,'loss:',ep_loss)
        print('Accuracy:',accuracy.eval({X:mnist.test.images, Y:mnist.test.labels}))

if __name__ == '__main__':
    print('-'*10,'\n',new_folder,'\n','-'*10)
    train_net(X)
