import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


gan = tf.Graph()
batch = 100

class GAN:
	
	def __init__(self):

		self.keep_prob = tf.placeholder(tf.float32, [])
		self.training = tf.placeholder(tf.bool)
		

	def generator(self, inputs_G):


		with tf.variable_scope("Generator"):
			
			layer_G_1 = tf.contrib.layers.fully_connected(inputs=inputs_G, num_outputs=100, activation_fn=tf.nn.relu)
			layer_G_2 = tf.contrib.layers.dropout(tf.contrib.layers.fully_connected(inputs=layer_G_1, num_outputs=400, activation_fn=tf.nn.relu), is_training=self.training)
			layer_G_3 = tf.contrib.layers.dropout(tf.contrib.layers.fully_connected(inputs=layer_G_2, num_outputs=600, activation_fn=tf.nn.relu), is_training=self.training)
			output_G = tf.contrib.layers.fully_connected(inputs=layer_G_3, num_outputs=784, activation_fn=tf.nn.tanh)

		return output_G

	def discriminator(self, inputs_D, reuse):

		with tf.variable_scope("Discriminator", reuse=reuse) as Discriminator:

			layer_D_1 = tf.contrib.layers.fully_connected(inputs=inputs_D, num_outputs=600, activation_fn=tf.nn.relu)
			layer_D_2 = tf.contrib.layers.dropout(tf.contrib.layers.fully_connected(inputs=layer_D_1, num_outputs=400, activation_fn=tf.nn.relu), is_training=self.training)
			layer_D_3 = tf.contrib.layers.dropout(tf.contrib.layers.fully_connected(inputs=layer_D_2, num_outputs=100, activation_fn=tf.nn.relu), is_training=self.training)
			output_D = tf.contrib.layers.fully_connected(inputs=layer_D_3, num_outputs=1, activation_fn=tf.sigmoid)

	

		return output_D

with gan.as_default():

	inputs_G = tf.placeholder(tf.float32, [batch, 10])
	inputs_D = tf.placeholder(tf.float32, [batch, 784])

	model = GAN()

	

	r = model.discriminator(inputs_D, None)
	sample = model.generator(inputs_G)
	f = model.discriminator(sample, True)

	


	loss_D = -0.5*tf.reduce_mean(tf.log(tf.clip_by_value(r, clip_value_min=1e-15, clip_value_max=0.9999999)) + tf.log(tf.clip_by_value(1 - f, clip_value_min=1e-15, clip_value_max=0.9999999)))
	loss_G = -0.5*tf.reduce_mean(tf.log(tf.clip_by_value(f, clip_value_min=1e-15, clip_value_max=0.9999999)))

	opt_D = tf.train.AdamOptimizer()
	opt_G = tf.train.AdamOptimizer()
	train_opt_D = opt_D.minimize(loss_D)
	train_opt_G = opt_G.minimize(loss_G)

	

	summary_D = tf.summary.scalar('Discriminator loss', loss_D) 
	summary_G = tf.summary.scalar('Generator loss', loss_G) 
	summary_I = tf.summary.image('Image', tf.expand_dims(tf.reshape(sample, [batch, 28, 28]), axis=-1), 10)
	summary_op = tf.summary.merge_all()

	init = tf.global_variables_initializer()


with tf.Session(graph=gan) as sess:

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
	sess.run(init)
	train_writer = tf.summary.FileWriter('summary_directory', sess.graph)

	#for n in tf.get_default_graph().as_graph_def().node:
	#	print n.name

	#for op in tf.get_default_graph().get_operations():
	#	print op.values() 


	for iterations in range(0, 1000):

		for d in range(0, 3):

			batch_xs, batch_ys = mnist.train.next_batch(batch)
			l_d, _= sess.run([loss_D, train_opt_D], feed_dict={inputs_D: batch_xs, inputs_G: 2.0*np.random.random_sample((batch, 10))-1.0, model.keep_prob:0.5, model.training:True})
			print l_d, iterations, d, 'Discriminator loss'


		for g in range(0, 2):

			batch_xs, batch_ys = mnist.train.next_batch(batch)
			l_g, _= sess.run([loss_G, train_opt_G], feed_dict={inputs_D: batch_xs, inputs_G: 2.0*np.random.random_sample((batch, 10))-1.0, model.keep_prob:0.5, model.training:True})
			print l_g, iterations, g, 'Generator loss'

		summary_full = sess.run(summary_op, feed_dict={inputs_D: batch_xs, inputs_G: 2.0*np.random.random_sample((batch, 10))-1.0, model.keep_prob:0.5, model.training:False})
		train_writer.add_summary(summary_full, iterations)


			
	

	

