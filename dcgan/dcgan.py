import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as weight_init
import cPickle
import numpy as np
import cv2
import matplotlib.pyplot as plt


batch = 100
learning_rate = 0.0002


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # Generator

        if torch.cuda.is_available():
            self.latent_variable = Variable(torch.FloatTensor(batch, 100).uniform_(-1, 1).cuda(), requires_grad=True)
        else:
            self.latent_variable = Variable(torch.FloatTensor(batch, 100).uniform_(-1, 1),requires_grad=True)
        #self.latent_distribution = weight_init.uniform(self.latent_variable, -1, 1)
        self.latent_distribution = self.latent_variable
        self.project = nn.Linear(100, 3 * 3 * 512)
        #self.batch_norm_1 = nn.BatchNorm2d(256)
        self.batch_norm_2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.g_layer_1 = nn.ConvTranspose2d(512, 256, 3, 2)
        self.g_layer_2 = nn.ConvTranspose2d(256, 128, 3, 2)
        self.g_layer_3 = nn.ConvTranspose2d(128, 3, 4, 2)

    def forward(self):
        # Generator

        z = self.project(self.latent_distribution)
        z = z.view(batch, 512, 3, 3)
        #z = self.batch_norm_1(self.relu(self.g_layer_1(z)))
        z = self.relu(self.g_layer_1(z))
        z = self.batch_norm_2(self.relu(self.g_layer_2(z)))
        G_z = self.tanh(self.g_layer_3(z))

        return G_z

    def criterion(self, f):

        loss_g = 0.5*torch.neg(torch.mean(torch.log(torch.clamp(f, min=1e-15, max=0.9999))))
        #loss_g = 0.5*torch.mean(torch.log1p(f))

        return loss_g


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        # Discriminator
        self.sigmoid = nn.Sigmoid()
        #self.batch_norm_1 = nn.BatchNorm2d(128)
        self.batch_norm_2 = nn.BatchNorm2d(256)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reduce = nn.Linear(512, 1)
        self.d_layer_1 = nn.Conv2d(3, 128, 5, 2)
        self.d_layer_2 = nn.Conv2d(128, 256, 5, 2)
        self.d_layer_3 = nn.Conv2d(256, 512, 5, 2)

    def forward(self, x):

        # Discriminator feed

        #x = self.batch_norm_1(self.leaky_relu(self.d_layer_1(x)))
        x = self.leaky_relu(self.d_layer_1(x))
        x = self.batch_norm_2(self.leaky_relu(self.d_layer_2(x)))
        x = self.leaky_relu(self.d_layer_3(x))
        x = x.view(batch, 512)
        x = self.sigmoid(self.reduce(x))

        return x

    def criterion(self, r, f):

        loss_d = 0.5*torch.neg(torch.mean(torch.log(torch.clamp(r, min=1e-15, max=0.9999999)) + torch.log(torch.clamp(1 - f, min=1e-15, max=0.9999999))))
        #loss_d = 0.5*torch.mean(torch.log1p(r) + torch.log1p(1 - f))

        return loss_d

# Training

discriminator = Discriminator()
generator = Generator()
if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)

with open('collection.pickle', 'rb') as f:
    images = cPickle.load(f)

print len(images)
count = 0
for epoch in xrange(1, 50):
    for i in range(0, len(images), batch):
        count += 1
        batch_x = []
        for j in range(i, i + batch):
            batch_x.append(images[j])
        batch_x = np.array(batch_x, dtype=np.float32)
        batch_x = (batch_x / 255.0) - 1.0
        #print batch_x

        batch_x = torch.from_numpy(batch_x)
        r = Variable(batch_x.permute(0, 3, 1, 2), requires_grad=True)

        if torch.cuda.is_available():
            r = r.cuda()

        f = generator()
        real = discriminator(r)
        fake = discriminator(f)
        loss_d = discriminator.criterion(real, fake)

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        if count%3 == 0:

            f = generator()
            fake = discriminator(f)
            loss_g = generator.criterion(fake)

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            print ('epoch -->', epoch, 'iteration -->', i, 'Loss_D -->', loss_d.data[0], 'Loss_G -->', loss_g.data[0])

        print ('epoch -->', epoch, 'iteration -->', i, 'Loss_D -->', loss_d.data[0])

    gen = generator()
    #print gen.data, 'Generated Data'
    #print r, 'Real Data'
    #gen = r
    gen = gen.permute(0, 2, 3, 1)
    gen = gen.data.cpu().numpy()
    img = []
    l = 10
    for k_img in range(0, batch, 10):
        h_structure = gen[k_img]
        for k_r in range(k_img + 1, k_img + 10):
            h_structure = np.hstack((h_structure, gen[k_r]))
        img.append(h_structure)
    collage = img[0]
    for m_img in range(1, len(img)):
        collage = np.vstack((collage, img[m_img]))
    #collage = (collage*255.0) + 255.0
    #print collage
    #collage = np.asarray(collage, dtype=np.uint8)
    collage = collage + 1.0
    #print collage
    location = 'results/image_' + str(epoch) + '.png'
    #cv2.imwrite(location, collage)
    plt.imsave(location, collage)


