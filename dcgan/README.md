# DCGAN
Discriminator (<br>
  (sigmoid): Sigmoid ()<br>
  (batch_norm_2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)<br>
  (leaky_relu): LeakyReLU (0.2)<br>
  (reduce): Linear (512 -> 1)<br>
  (d_layer_1): Conv2d(3, 128, kernel_size=(5, 5), stride=(2, 2))<br>
  (d_layer_2): Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2))<br>
  (d_layer_3): Conv2d(256, 512, kernel_size=(5, 5), stride=(2, 2))<br>
)<br>
Generator (<br>
  (project): Linear (100 -> 4608)<br>
  (batch_norm_2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)<br>
  (relu): ReLU ()<br>
  (tanh): Tanh ()<br>
  (g_layer_1): ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2))<br>
  (g_layer_2): ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2))<br>
  (g_layer_3): ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=(2, 2))<br>
)<br>

![alt text](https://github.com/pjavia/GAN/blob/master/gan/Discriminator%20loss.png)
![alt text](https://github.com/pjavia/GAN/blob/master/gan/Generator%20loss.png)

