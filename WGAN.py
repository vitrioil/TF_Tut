import os
import time
import numpy as np
import functools
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import cifar10
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data

class WGAN:
	
	def __init__(self, latent_size, img_size, epochs=1_000_000, clip = 0.01, batch_size=32, mnist = False, data_path = ""):
		self.logdir = str(int(time.time()))+"/"
		os.mkdir(self.logdir)
		self.epochs = epochs
		self.img_size = img_size 
		self.latent_size = latent_size		
		self.features = functools.reduce(lambda x,y: x*y, self.img_size)
		if mnist:
			self.data = input_data.read_data_sets("../../MNIST_data", one_hot = True)
		else:
			self.data = self.create_generator(data_path)
		self.clip = clip
		self.classes = 1
		self.train_d_epoch = 5
		self.batch_size = batch_size

	def create_generator(self, path):
		if path == "":	
			assert False, "No data!"
		if not path.endswith("npy"):
			assert False, "Provide numpy.ndarray"

		return np.load(path, mmap_mode='r')

	def generate_data(self, batch=None):
		if batch is None:
			batch = self.batch_size

		start_indx = 0 
		end_indx = batch
		reset = False
		while True:
			yield self.data[start_indx: end_indx].reshape((-1, self.features)), None
			if reset:
				start_indx = 0
				end_indx = batch
			start_indx = end_indx
			end_indx += batch
			if end_indx >= self.data_size:
				end_indx = self.data_size - 1
				reset = True

	def set_weights(self, disc_hidden_size, gen_hidden_size):
		self.d_w_1 = tf.Variable(tf.random_normal([self.features, disc_hidden_size], stddev = 1/tf.sqrt(self.features/2)))
		self.d_b_1 = tf.Variable(tf.zeros([disc_hidden_size]))

		self.d_w_2 = tf.Variable(tf.random_normal([disc_hidden_size, self.classes], stddev = 1/tf.sqrt(disc_hidden_size/2))) 
		self.d_b_2 = tf.Variable(tf.zeros([self.classes]))
		
		self.g_w_1 = tf.Variable(tf.random_normal([self.latent_size, gen_hidden_size], stddev = 1/tf.sqrt(self.latent_size/2))) 
		self.g_b_1 = tf.Variable(tf.zeros([gen_hidden_size]))
	
		self.g_w_2 = tf.Variable(tf.random_normal([gen_hidden_size, self.features], stddev = 1/tf.sqrt(gen_hidden_size/2)))
		self.g_b_2 = tf.Variable(tf.zeros([self.features]))

		gen_weights = [self.g_w_1, self.g_w_2, self.g_b_1, self.g_b_2]
		disc_weights = [self.d_w_1, self.d_w_2, self.d_b_1, self.d_b_2]

		return gen_weights, disc_weights

	def random_vector(self, size):	
		return np.random.uniform(-1., 1., [size, self.latent_size])

	def critic(self, x):
		z = x @ self.d_w_1 + self.d_b_1
		a = tf.nn.relu(z)
		
		z = a @ self.d_w_2 + self.d_b_2
		return z
	
	def generator(self, z):
		z_ = tf.matmul(z, self.g_w_1) + self.g_b_1
		a = tf.nn.relu(z_)

		z_ = tf.matmul(a, self.g_w_2) + self.g_b_2
		return tf.nn.sigmoid(z_)

	def loss(self, z_real, z_fake):
		d_loss = -(tf.reduce_mean(z_real) - tf.reduce_mean(z_fake))
		g_loss = -tf.reduce_mean(z_fake)

		return g_loss, d_loss
	
	def train(self, gen_weights, disc_weights,lr = 1e-4):
		Z_T = tf.placeholder(tf.float32, shape = [None, self.latent_size])
		X_T = tf.placeholder(tf.float32, shape = [None, self.features])

		fake_inp = self.generator(Z_T)
		z_real = self.critic(X_T)
		z_fake = self.critic(fake_inp)
		
		gen_loss, disc_loss = self.loss(z_real, z_fake)

		#disc_weights = [self.d_w_1, self.d_w_2, self.d_b_1, self.d_b_2]
		#gen_weights = [self.g_w_1, self.g_w_2, self.g_b_1, self.g_b_2]

		opt_gen = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(gen_loss,var_list = gen_weights )
		opt_disc = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(disc_loss, var_list = disc_weights)

		clip_disc = [t.assign(tf.clip_by_value(t, -self.clip, self.clip)) for t in disc_weights]

		saver = tf.train.Saver()
		init = tf.global_variables_initializer()
		with tf.Session() as sess:	
			sess.run(init)
			for epoch in range(self.epochs):
				G_Loss, D_Loss = 0, 0
				for _ in range(self.train_d_epoch):
					x,_ = self.data.train.next_batch(self.batch_size)
					z = self.random_vector(self.batch_size)
					_, d_loss,__  = sess.run(
					  	[
							opt_disc, disc_loss, clip_disc
						],
						feed_dict = {Z_T: z, X_T: x}
					  )
					D_Loss += d_loss

				z = self.random_vector(self.batch_size)
				_, g_loss = sess.run(
						[
							opt_gen, gen_loss
						],
						feed_dict = {Z_T: z}
				)		
				G_Loss += g_loss
				
				if epoch % 10 == 0:
					print(f"{epoch}/{self.epochs} epochs, Generator_loss {G_Loss:.4f}, Discriminator loss {D_Loss:.4f}",end="\r")
					if epoch % 100 == 0:
						z = self.random_vector(16)

						fig = plt.figure(figsize = (4, 4))
						sample_output = sess.run(fake_inp, feed_dict = {Z_T: z})
						grid = gridspec.GridSpec(4, 4)
						for indx, sample in enumerate(sample_output):
							ax = plt.subplot(grid[indx])
							plt.axis("off")
							ax.set_xticklabels([])
							ax.set_yticklabels([])
							ax.set_aspect("equal")
							plt.imshow(sample.reshape(self.img_size))

						plt.savefig(f"{self.logdir}sample_output_{epoch}.png")
						plt.close(fig)
						saver.save(sess, self.logdir+"model.ckpt", epoch)

if __name__ == "__main__":
	wgan = WGAN(10, (28, 28), mnist=True) 
	g,d = wgan.set_weights(128, 128)
	wgan.train(g, d)
	print(wgan.logdir)
