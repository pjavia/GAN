import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


gan = tf.Graph()
batch = 100

with gan.as_default():

	z = tf.random_normal([batch, 32, 1], mean=-1, stddev=4)
	y_inp = tf.placeholder(tf.float32, [batch, 10])
	y = tf.expand_dims(y_inp, axis= -1)
	x_inp = tf.placeholder(tf.float32, shape=[batch, 784])
	x = tf.expand_dims(x_inp, axis= -1)
	fake = tf.ones(shape=[], dtype=tf.float32)


	W_discriminator_layer1 = tf.get_variable(name='W_1', shape=[784, 800])
	b_discriminator_layer1 = tf.get_variable(name='b_1', shape=[800, 1])
	W_generator_layer1 = tf.get_variable(name='W_2', shape=[32, 784])
	b_generator_layer1 = tf.get_variable(name='b_2', shape=[784, 1])
	W_discriminator_layer2 = tf.get_variable(name='W_3', shape=[784, 800])
	b_discriminator_layer2 = tf.get_variable(name='b_3', shape=[800, 1])

	W_softmax = tf.get_variable(name='W_4', shape=[800, 10])
	b_softmax = tf.get_variable(name='b_4', shape=[10, 1])

	


	# W_discriminator_real = tf.get_variable(name='W_4', shape=[800, 1])
	# b_discriminator_real = tf.get_variable(name='b_4', shape=[1])

	# W_discriminator_fake = tf.get_variable(name='W_5', shape=[800, 1])
	# b_discriminator_fake = tf.get_variable(name='b_5', shape=[1])

	
	# Generator
	G_z_ind = []
	g_input = tf.unstack(z)
		

	# G_z = tf.stack(G_z_ind)

	# Discriminator

	d_input = tf.unstack(x)
	D_x_ind = []
	D_G_z_ind = []
	D_loss = []
	G_loss =[]
	G_display_batch = []
	y_loss = []

	for b in range(0, batch):
		G = tf.nn.relu(tf.matmul(W_generator_layer1, g_input[b], transpose_a = True) + b_generator_layer1)
		G_z_ind.append(G)
		D = tf.nn.relu(tf.matmul(W_discriminator_layer1, d_input[b], transpose_a = True) + b_discriminator_layer1)
		D_x_ind.append(D)
		D_G_z_ind.append(tf.nn.relu(tf.matmul(W_discriminator_layer2, G_z_ind[b], transpose_a = True) + b_discriminator_layer2))
		D_x_probability = tf.sigmoid(D_x_ind[b])
		D_G_z_probability = tf.sigmoid(D_G_z_ind[b])
		D_loss.append((-1*tf.reduce_mean(tf.log(D_x_probability))) +  (-1*tf.reduce_mean(tf.log(fake - D_G_z_probability))))
		G_loss.append(tf.reduce_mean(tf.log(D_x_probability)) + tf.reduce_mean(tf.log(fake - D_G_z_probability)))
		G_display = tf.reshape(tf.squeeze(G, axis=-1), [28, 28, 1])
		G_display_batch.append(G_display)
		y_pred = tf.nn.softmax(tf.matmul(W_softmax, D, transpose_a=True) + b_softmax)
		y_loss.append(-tf.reduce_mean(y_pred*tf.log(tf.clip_by_value(y,1e-10,1.0))))

	cross_entropy = tf.reduce_mean(tf.stack(y_loss))

	G_z = tf.stack(G_display_batch) 

	J_D = tf.reduce_mean(tf.stack(D_loss))
	J_G = tf.reduce_mean(tf.stack(G_loss))

	opt_D = tf.train.AdamOptimizer()
	opt_G = tf.train.AdamOptimizer()
	opt_cross_entropy = tf.train.AdamOptimizer()
	train_opt_D = opt_D.minimize(J_D)
	train_opt_G = opt_G.minimize(J_G)
	train_opt_classifier = opt_cross_entropy.minimize(cross_entropy) 

	# D_x_probability = tf.sigmoid(tf.matmul(D_x, W_discriminator_real)+b_discriminator_real)
	# D_G_z_probability = tf.sigmoid(tf.matmul(D_G_z, W_discriminator_fake)+b_discriminator_fake)

	summary_D = tf.summary.scalar('Discriminator loss', J_D) # , name='discriminator')
	summary_C = tf.summary.scalar('Classifier loss', cross_entropy) # , name='cross entropy')
	summary_G = tf.summary.scalar('Generator loss', J_G) # , name='generator')
	# summary_G_hist = tf.summary.histogram("histogram loss", J_G)
	summary_I = tf.summary.image('img', G_z, 10)
	summary_op = tf.summary.merge_all()

	

	

	init = tf.global_variables_initializer()

	saver = tf.train.Saver()

with tf.Session(graph=gan) as sess:

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	sess.run(init)
	train_writer = tf.summary.FileWriter('summary_directory')
	# c_D = []
	# c_G = []
	# c_C = []

	for k in range(1, 50):

		print 'epoch - - - - - - - - >', k

		if k%20 == 0:
			save_path = saver.save(sess, "model_gan")
		

		batch_xs, batch_ys = mnist.train.next_batch(batch)
		for keep_D_ahead in range(0, 10):
			T_D, T_C, J_D_loss, J_C_loss, summary_full = sess.run([train_opt_D, train_opt_classifier, J_D, cross_entropy, summary_op], feed_dict={x_inp: batch_xs, y_inp: batch_ys})
			print J_C_loss, 'classifier loss'
			print J_D_loss, 'Discriminator loss'
			# c_D.append(sess.run(tf.summary.scalar('Discriminator loss', J_D, name='discriminator'), feed_dict={x_inp: batch_xs, y_inp: batch_ys}))
			# c_C.append(sess.run(tf.summary.scalar('Classifier loss', cross_entropy, name='cross entropy'), feed_dict={x_inp: batch_xs, y_inp: batch_ys}))

		T_G, J_G_loss = sess.run([train_opt_G, J_G], feed_dict={x_inp: batch_xs})
		print J_G_loss, 'Generator loss'
		# c_G.append(sess.run(tf.summary.scalar('Generator loss', J_G, name='generator'), feed_dict={x_inp: batch_xs}))
		

	# train_writer.add_summary(tf.summary.merge(c_D))
	# train_writer.add_summary(tf.summary.merge(c_G))
	# train_writer.add_summary(tf.summary.merge(c_C))



	# summary_display = sess.run(tf.summary.image('img', G_z, 1))
	train_writer.add_summary(summary_full)

	




	

	

	

