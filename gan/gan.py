import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


gan = tf.Graph()
batch = 100
projection = 20
learning_rate=0.0002

class GAN:
	
	def __init__(self):


		self.layer_G_1_neurons = 200
		self.layer_G_2_neurons = 400
		self.layer_G_3_neurons = 400
		self.layer_G_4_neurons = 784

		self.layer_D_1_neurons = 200
		self.layer_D_2_neurons = 400
		self.layer_D_3_neurons = 400
		self.layer_D_4_neurons = 1

	def generator(self, inputs_G, p):

		with tf.variable_scope("Generator"):

			layer_G_1_W = tf.get_variable("layer_G_1_W", shape=[projection, self.layer_G_1_neurons], dtype=tf.float32)
			layer_G_1_b = tf.get_variable("layer_G_1_b", shape=[self.layer_G_1_neurons], dtype=tf.float32)

			layer_G_2_W = tf.get_variable("layer_G_2_W", shape=[self.layer_G_1_neurons, self.layer_G_2_neurons], dtype=tf.float32)
			layer_G_2_b = tf.get_variable("layer_G_2_b", shape=[self.layer_G_2_neurons], dtype=tf.float32)
 
			layer_G_3_W = tf.get_variable("layer_G_3_W", shape=[self.layer_G_2_neurons, self.layer_G_3_neurons], dtype=tf.float32)
			layer_G_3_b = tf.get_variable("layer_G_3_b", shape=[self.layer_G_3_neurons], dtype=tf.float32)

			layer_G_4_W = tf.get_variable("layer_G_4_W", shape=[self.layer_G_3_neurons, self.layer_G_4_neurons], dtype=tf.float32)
			layer_G_4_b = tf.get_variable("layer_G_4_b", shape=[self.layer_G_4_neurons], dtype=tf.float32)

		
			layer_G_1 = tf.nn.relu(tf.add(tf.matmul(inputs_G, layer_G_1_W), layer_G_1_b))
			layer_G_2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_G_1, layer_G_2_W), layer_G_2_b)), keep_prob=p)
			layer_G_3 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_G_2, layer_G_3_W), layer_G_3_b)), keep_prob=p)
			sample = tf.sigmoid(tf.add(tf.matmul(layer_G_3, layer_G_4_W), layer_G_4_b))

		return sample
		

	def discriminator(self, inputs_D, reuse, p):

		with tf.variable_scope("Discriminator", reuse=reuse):			

			layer_D_1_W = tf.get_variable("layer_D_1_W", shape=[784, self.layer_D_1_neurons], dtype=tf.float32)
			layer_D_1_b = tf.get_variable("layer_D_1_b", shape=[self.layer_D_1_neurons], dtype=tf.float32)

			layer_D_2_W = tf.get_variable("layer_D_2_W", shape=[self.layer_D_1_neurons, self.layer_D_2_neurons], dtype=tf.float32)
			layer_D_2_b = tf.get_variable("layer_D_2_b", shape=[self.layer_D_2_neurons], dtype=tf.float32)
 
			layer_D_3_W = tf.get_variable("layer_D_3_W", shape=[self.layer_D_2_neurons, self.layer_D_3_neurons], dtype=tf.float32)
			layer_D_3_b = tf.get_variable("layer_D_3_b", shape=[self.layer_D_3_neurons], dtype=tf.float32)

			layer_D_4_W = tf.get_variable("layer_D_4_W", shape=[self.layer_D_3_neurons, self.layer_D_4_neurons], dtype=tf.float32)
			layer_D_4_b = tf.get_variable("layer_D_4_b", shape=[self.layer_D_4_neurons], dtype=tf.float32)

		with tf.variable_scope("Discriminator", reuse=reuse):

			layer_D_1 = tf.nn.relu(tf.add(tf.matmul(inputs_D, layer_D_1_W), layer_D_1_b))
			layer_D_2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_D_1, layer_D_2_W), layer_D_2_b)), keep_prob=p)# , is_training=self.is_training)
			layer_D_3 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_D_2, layer_D_3_W), layer_D_3_b)), keep_prob=p)#, is_training=self.is_training)
			real_or_fake = tf.sigmoid(tf.add(tf.matmul(layer_D_3, layer_D_4_W), layer_D_4_b))

		return real_or_fake



	



gan = tf.Graph()

with gan.as_default():

	model = GAN()

	
	p = inputs_G = tf.placeholder(tf.float32, [])
	inputs_G = tf.placeholder(tf.float32, [batch, projection])

	inputs_D = tf.placeholder(tf.float32, [batch, 784])


	sample = model.generator(inputs_G, p)
	f = model.discriminator(sample, None, p)
	r = model.discriminator(inputs_D, True, p)

	# Loss and  training

	weights_of_model = tf.trainable_variables()

	Generator_knowledge = [variables for variables in weights_of_model if 'Generator' in variables.name]
	print Generator_knowledge
	Discriminator_knowledge = [variables for variables in weights_of_model if 'Discriminator' in variables.name]
	print Discriminator_knowledge

	loss_D = -0.5*tf.reduce_mean(tf.log(tf.clip_by_value(r, clip_value_min=1e-15, clip_value_max=0.9999999)) + tf.log(tf.clip_by_value(1 - f, clip_value_min=1e-15, clip_value_max=0.9999999)))
	loss_G = -0.5*tf.reduce_mean(tf.log(tf.clip_by_value(f, clip_value_min=1e-15, clip_value_max=0.9999999)))

	opt_D = tf.train.AdamOptimizer(learning_rate=learning_rate)
	opt_G = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_opt_D = opt_D.minimize(loss_D, var_list=Discriminator_knowledge)
	train_opt_G = opt_G.minimize(loss_G, var_list=Generator_knowledge)

	

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

	for epoch in range(0, 20):

		for iterations in range(0, 600):

			for d in range(0, 1):

				batch_xs, batch_ys = mnist.train.next_batch(batch)
				l_d, _= sess.run([loss_D, train_opt_D], feed_dict={inputs_D: batch_xs, inputs_G: np.random.uniform(0.0, 1.0,(batch, projection)), p: 0.5})
				print l_d, iterations, d, 'Discriminator loss'

		

			for g in range(0, 1):

				batch_xs, batch_ys = mnist.train.next_batch(batch)
				l_g, _= sess.run([loss_G, train_opt_G], feed_dict={inputs_D: batch_xs, inputs_G: np.random.uniform(0.0, 1.0,(batch, projection)), p: 0.5})
				print l_g, iterations, g, 'Generator loss'


		summary_full = sess.run(summary_op, feed_dict={inputs_D: batch_xs, inputs_G: np.random.uniform(0.0, 1.0,(batch, projection)), p: 1.0})
		train_writer.add_summary(summary_full, epoch)



			
	

	

