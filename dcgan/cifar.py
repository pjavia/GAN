import cPickle
import numpy as np
import cv2

repository = []

for i in range(1, 6):

    name = 'cifar/data_batch_'+str(i)

    with open(name, 'rb') as fo:
        data = cPickle.load(fo)


    collect = data.get('data')


    for j in collect:
        red = []
        green = []
        blue = []
        image = []
        red = j[0:1024]
        green = j[1024:2048]
        blue = j[2048:3072]
        repo = []
        for k in range(0, 1024):
            image.append([red[k], green[k], blue[k]])
        for l in range(0, 1024, 32):
            repo.append(image[l:l+32])
        #print np.array(repo).shape
        repository.append(np.array(repo))


with open('collection.pickle', 'wb') as f:
    cPickle.dump(repository ,f)
