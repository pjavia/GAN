import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as weight_init
import cPickle
import numpy as np

batch = 32
learning_rate = 0.0002


class Net(nn.Module):

    def __init__(self):
        
        super(Net, self).__init__()

        # Generator 

        self.latent_variable = Variable(torch.FloatTensor(batch, 100))
        self.latent_distribution = weight_init.uniform(self.latent_variable, -1, 1)
        self.project = nn.Linear(100, 4*4*512)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.g_layer_1 = nn.ConvTranspose2d(512, 256, 5, 2, 2, 1)
        self.g_layer_2 = nn.ConvTranspose2d(256, 128, 5, 2, 2, 1)
        self.g_layer_3 = nn.ConvTranspose2d(128, 3, 5, 2, 2, 1)

        # Discriminator
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reduce = nn.Linear(512, 1)
        self.d_layer_1 = nn.Conv2d(3, 128, 5, 2)
        self.d_layer_2 = nn.Conv2d(128, 256, 5, 2)
        self.d_layer_3 = nn.Conv2d(256, 512, 5, 2)
         


    def forward(self, x):

    	# Generator

    	z = self.project(self.latent_distribution)
    	z = z.view(batch, 512, 4, 4)
    	z = self.relu(self.g_layer_1(z))
    	z = self.relu(self.g_layer_2(z))
    	G_z = self.relu(self.g_layer_3(z))

    	# Discriminator feed fake

    	f = self.leaky_relu(self.d_layer_1(G_z))
    	f = self.leaky_relu(self.d_layer_2(f))
    	f = self.tanh(self.d_layer_3(f))
    	f = f.view(batch, 512)
    	f = self.sigmoid(self.reduce(f))



    	# Discriminator feed real

    	r = self.leaky_relu(self.d_layer_1(x))
    	r = self.leaky_relu(self.d_layer_2(r))
    	r = self.tanh(self.d_layer_3(r))
    	r = r.view(batch, 512)
    	r = self.sigmoid(self.reduce(r))

    	loss_d =  torch.neg(torch.mean(torch.log(f) + torch.log(1 - f)))
    	loss_g =  torch.neg(torch.mean(torch.log(1 - f)))

    	return loss_d, loss_g

    def sample(self):

    	z = self.project(self.latent_distribution)
    	z = z.view(batch, 512, 4, 4)
    	z = self.relu(self.g_layer_1(z))
    	z = self.relu(self.g_layer_2(z))
    	G_z = self.relu(self.g_layer_3(z))

    	return G_z

    	









# Training

net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

with open('collection.pickle', 'rb') as f:
    images = cPickle.load(f)


for i in range(0, len(images), batch):
	batch_x = []
	for j in range(i, i + batch):

		batch_x.append(images[j])

	batch_x = torch.from_numpy(np.array(batch_x, dtype=np.float32))
	batch_x = batch_x.permute(0, 3, 1, 2)

	loss_d, loss_g = net(Variable(batch_x))

	print loss_d.data[0], loss_g.data[0]

	optimizer.zero_grad()
	loss_d.backward(retain_variables=True)
	optimizer.step()

	optimizer.zero_grad()
	loss_g.backward()
	optimizer.step()














