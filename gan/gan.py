import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


gan = tf.Graph()
batch = 100

with gan.as_default():


	z = tf.random_uniform([batch, 32, 1], minval= 0, maxval = 2, seed=2)
	y_inp = tf.placeholder(tf.float32, [batch, 10])
	y = tf.expand_dims(y_inp, axis= -1)
	x_inp = tf.placeholder(tf.float32, shape=[batch, 784])
	x = tf.expand_dims(x_inp, axis= -1)
	fake = tf.ones(shape=[], dtype=tf.float32)

	with tf.variable_scope("GAN"):


		W_discriminator_layer1 = tf.get_variable(name='W_1', shape=[784, 800])
		b_discriminator_layer1 = tf.get_variable(name='b_1', shape=[800, 1])
		W_discriminator_layer2 = tf.get_variable(name='W_2', shape=[800, 800])
		b_discriminator_layer2 = tf.get_variable(name='b_2', shape=[800, 1])

		W_generator_layer1 = tf.get_variable(name='W_3', shape=[32, 100])
		b_generator_layer1 = tf.get_variable(name='b_3', shape=[100, 1])
		W_generator_layer2 = tf.get_variable(name='W_4', shape=[100, 784])
		b_generator_layer2 = tf.get_variable(name='b_4', shape=[784, 1])
		
		W_softmax = tf.get_variable(name='W_5', shape=[800, 10])
		b_softmax = tf.get_variable(name='b_5', shape=[10, 1])

	


	
	# Generator
	G_z_ind = []
	g_input = tf.unstack(z)
		

	

	# Discriminator

	d_input = tf.unstack(x)
	D_x_ind = []
	D_G_z_ind = []
	D_loss = []
	G_loss =[]
	G_display_batch = []
	y_loss = []
	y_true = tf.unstack(y)

	with tf.variable_scope("GAN", reuse=True):

		for b in range(0, batch):

			# Let Generator generate

			G_h = tf.nn.relu(tf.matmul(W_generator_layer1, g_input[b], transpose_a = True) + b_generator_layer1) 
			G = tf.nn.relu(tf.matmul(W_generator_layer2, G_h, transpose_a = True) + b_generator_layer2)
			G_z_ind.append(G)

			# pass real

			D = tf.nn.relu(tf.matmul(W_discriminator_layer1, d_input[b], transpose_a = True) + b_discriminator_layer1)
			D_hidden = tf.nn.relu(tf.matmul(W_discriminator_layer2, D, transpose_a = True) + b_discriminator_layer2)
			D_x_ind.append(D_hidden)

			# pass fake

			D_g_1 = tf.nn.relu(tf.matmul(W_discriminator_layer1, G_z_ind[b], transpose_a = True) + b_discriminator_layer1)
			D_g_2 = tf.nn.relu(tf.matmul(W_discriminator_layer2, D_g_1, transpose_a = True) + b_discriminator_layer2)
			D_G_z_ind.append(D_g_2)

			# Find probabilities 
			D_x_probability = tf.sigmoid(D_x_ind[b])
			D_G_z_probability = tf.sigmoid(D_G_z_ind[b])

			# Find loss
			D_loss.append((-1*tf.reduce_mean(tf.log(D_x_probability))) +  (-1*tf.reduce_mean(tf.log(fake - D_G_z_probability))))
			G_loss.append(tf.reduce_mean(tf.log(D_x_probability)) + tf.reduce_mean(tf.log(fake - D_G_z_probability)))

			# Sample
			G_display = tf.reshape(tf.squeeze(G, axis=-1), [28, 28, 1])
			G_display_batch.append(G_display)

			y_pred = tf.nn.softmax(tf.nn.relu(tf.matmul(W_softmax, D_hidden, transpose_a=True) + b_softmax), dim=0)
			y_loss.append(tf.losses.mean_squared_error(y_true[b], y_pred))


	cross_entropy = tf.reduce_mean(tf.stack(y_loss))

	G_z = tf.stack(G_display_batch) 

	J_D = tf.reduce_mean(tf.stack(D_loss))
	J_G = tf.reduce_mean(tf.stack(G_loss))
	

	opt_cross_entropy = tf.train.AdamOptimizer()
	opt_D = tf.train.AdamOptimizer()
	opt_G = tf.train.AdamOptimizer()
	train_opt_D = opt_D.minimize(J_D)
	train_opt_G = opt_G.minimize(J_G)
	train_opt_classifier = opt_cross_entropy.minimize(cross_entropy) 

	
	summary_D = tf.summary.scalar('Discriminator loss', J_D) 
	summary_G = tf.summary.scalar('Generator loss', J_G) 
	summary_I = tf.summary.image('img', G_z, 10)
	summary_op = tf.summary.merge_all()

	

	

	init = tf.global_variables_initializer()

	saver = tf.train.Saver()

with tf.Session(graph=gan) as sess:

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	sess.run(init)
	train_writer = tf.summary.FileWriter('summary_directory')
	count = 0

	for keep_C_ahead in range(0, 10):

		print 'iteration - - - - - - - - >', keep_C_ahead+1
		batch_xs, batch_ys = mnist.train.next_batch(batch)
		T_C, J_C_loss, = sess.run([train_opt_classifier, cross_entropy], feed_dict={x_inp: batch_xs, y_inp: batch_ys})
		print J_C_loss, 'classifier loss'

	for k in range(0, 100):

		print 'epoch - - - - - - - - >', k+1

		
		for keep_D_ahead in range(0, 5):
			count += 1
			batch_xs, batch_ys = mnist.train.next_batch(batch)
			

			T_D, J_D_loss, summary_full = sess.run([train_opt_D, J_D, summary_op], feed_dict={x_inp: batch_xs})
			train_writer.add_summary(summary_full, count)
			print J_D_loss, 'Discriminator loss'
			

		T_G, J_G_loss = sess.run([train_opt_G, J_G], feed_dict={x_inp: batch_xs})
		print J_G_loss, 'Generator loss'
			

	

	




	

	

	

