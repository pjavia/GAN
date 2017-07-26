# DCGAN

### Architecture
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


#### epoch 1
![alt text](https://github.com/pjavia/GAN/blob/master/dcgan/images/image_1.png)
#### epoch 2
![alt text](https://github.com/pjavia/GAN/blob/master/dcgan/images/image_2.png)
#### epoch 3
![alt text](https://github.com/pjavia/GAN/blob/master/dcgan/images/image_3.png)
#### epoch 4
![alt text](https://github.com/pjavia/GAN/blob/master/dcgan/images/image_4.png)
#### epoch 5
![alt text](https://github.com/pjavia/GAN/blob/master/dcgan/images/image_5.png)
#### epoch 6
![alt text](https://github.com/pjavia/GAN/blob/master/dcgan/images/image_6.png)
#### epoch 7
![alt text](https://github.com/pjavia/GAN/blob/master/dcgan/images/image_7.png)
#### epoch 8
![alt text](https://github.com/pjavia/GAN/blob/master/dcgan/images/image_8.png)
#### epoch 9
![alt text](https://github.com/pjavia/GAN/blob/master/dcgan/images/image_9.png)
#### epoch 10
![alt text](https://github.com/pjavia/GAN/blob/master/dcgan/images/image_10.png)
#### epoch 11
![alt text](https://github.com/pjavia/GAN/blob/master/dcgan/images/image_11.png)
#### epoch 12
![alt text](https://github.com/pjavia/GAN/blob/master/dcgan/images/image_12.png)
#### epoch 13
![alt text](https://github.com/pjavia/GAN/blob/master/dcgan/images/image_13.png)
#### epoch 14
![alt text](https://github.com/pjavia/GAN/blob/master/dcgan/images/image_14.png)
#### epoch 15
![alt text](https://github.com/pjavia/GAN/blob/master/dcgan/images/image_15.png)
