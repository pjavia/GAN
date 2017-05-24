import tensorflow as tf


gan = tf.Graph()

with gan.as_default():

	z = tf.random_normal([100, 32], mean=-1, stddev=4)
	x = tf.placeholder(tf.float32, shape=[784])
	real = tf.zeros(shape=[], dtype=tf.float32)
	fake = tf.ones(shape=[], dtype=tf.float32)


	W_discriminator_layer1 = tf.get_variable(name='W_1', shape=[784, 800])
	b_discriminator_layer1 = tf.get_variable(name='b_1', shape=[800])
	W_generator_layer1 = tf.get_variable(name='W_2', shape=[32, 784])
	b_generator_layer1 = tf.get_variable(name='b_2', shape=[784])
	W_discriminator_layer2 = tf.get_variable(name='W_3', shape=[784, 800])
	b_discriminator_layer2 = tf.get_variable(name='b_3', shape=[800])

	W_discriminator_real = tf.get_variable(name='W_4', shape=[800, 1])
	b_discriminator_real = tf.get_variable(name='b_4', shape=[1])

	W_discriminator_fake = tf.get_variable(name='W_5', shape=[800, 1])
	b_discriminator_fake = tf.get_variable(name='b_5', shape=[1])

	


	G_z = tf.nn.relu_layer(z, W_generator_layer1, b_generator_layer1)

	# Discriminator

	D_x = tf.nn.relu_layer(x, W_discriminator_layer1, b_discriminator_layer1)
	D_G_z = tf.nn.relu_layer(G_z, W_discriminator_layer2, b_discriminator_layer2)

	D_x_probability = tf.sigmoid(tf.matmul(D_x, W_discriminator_real)+b_discriminator_real)
	D_G_z_probability = tf.sigmoid(tf.matmul(D_G_z, W_discriminator_fake)+b_discriminator_fake)
	

	


	W_2 = tf.get_variable(name='W_2', shape=[800, 800])
	

