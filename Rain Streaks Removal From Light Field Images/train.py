import numpy as np
import scipy.io
from glob import glob
from copy import deepcopy
import random
import matplotlib.pyplot as plt
from libs.derainnet import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import time,datetime
import os
import cv2
import numpy as np
import matplotlib.pylab  as plt


plt.ioff()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# SETTINGS

image_path3 = 'F:/Rain/patch_datasum'
image_path4 = 'F:/Rain/patch_dispsum'

file_name3 = glob(image_path3 + "/*mat")
file_name4 = glob(image_path4 + "/*mat")

valdata = []
valmask = []

for file3, file4 in zip(file_name3, file_name4):
    org = scipy.io.loadmat(file3)
    org = org['imghorcat'].astype("float32")
    label = scipy.io.loadmat(file4)
    label = label['imghorcat'].astype("float32")

    org = org / 255  # converts image to numpy array
    label = label / 255

    valdata.append(org)
    valmask.append(label)

    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(valdata)
    random.seed(randnum)
    random.shuffle(valmask)

valrealdataset = np.array(valdata)
valmaskdataset = np.array(valmask)

# Create testing generator
n_epoch = 502
batch_size = 4
critic_updates = 4

BASE_DIR = './weight_batch/'

txtpath = './'

def process_line(line):
    line = line.strip('\n')
    rainpath = 'D:/DD/Dataset/sum_3rain2/' + line
    disppath = 'D:/DD/Dataset/sum_4disp2/' + line
    lablepath = 'D:/DD/Dataset/sum_1ori2/' + line
    rainrespath = 'D:/DD/Dataset/sum_2res2/' + line


    rainpath = h5py.File(rainpath)
    rainpath = rainpath['imghorcat'][:].astype("float32")
    disppath = h5py.File(disppath)
    disppath = disppath['imghorcat'][:].astype("float32")
    lablepath = h5py.File(lablepath)
    lablepath = lablepath['imghorcat'][:].astype("float32")
    rainrespath = h5py.File(rainrespath)
    rainrespath = rainrespath['imghorcat'][:].astype("float32")

    rainpath = np.transpose(rainpath)
    disppath = np.transpose(disppath)
    lablepath = np.transpose(lablepath)
    rainrespath = np.transpose(rainrespath)

    blurpath = np.zeros((9,128,128,3))
    for i in range(9):
        blur = cv2.bilateralFilter(np.array((rainpath[i, :, :, :])), 5, 75, 75)
        blurpath[i, :, :, :] = blur

    blurpath = blurpath/255

    rainpath = rainpath/255  # converts image to numpy array

    disppath = disppath
    lablepath = lablepath/255
    rainrespath = rainrespath/255
    return blurpath,rainpath,disppath,rainrespath,lablepath


def generate_arrays_from_file(path, batch_size,index):
    i = 0
    while 1:
        f = open(path)
        line = f.readlines()
        count = len(open(path, 'r').readlines())
        realdataset = []
        blurdateset = []
        dispdataset = []
        resdataset = []
        labledataset = []
        a = 0
        for i, rows in enumerate(line):
            #print(i)
            #print(rows)
            
            if i in range(index*batch_size, (index+1)*batch_size):
            #if i == index*batch_size and i < (index+1)*batch_size:
                #print(i)
                blur,real,disp,rainres, lable = process_line(rows)
                blurdateset.append(blur)
                realdataset.append(real)
                dispdataset.append(disp)
                resdataset.append(rainres)
                labledataset.append(lable)

            
        f.close()

        blurdateset = np.array(blurdateset)
        dispdataset = np.array(dispdataset)
        labledataset = np.array(labledataset)
        realdataset = np.array(realdataset)
        resdataset = np.array(resdataset)

        return blurdateset,realdataset,dispdataset,resdataset,labledataset


def save_all_weights(d1,d2, g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d1.save_weights(os.path.join(save_dir, 'discriminator1_{}.h5'.format(epoch_number)), True)
    d2.save_weights(os.path.join(save_dir, 'discriminator2_{}.h5'.format(epoch_number)), True)


def val_data_generator(valrealdataset, vallabledataset, batch_size):
    batches = 81  # (len(data) + batch_size - 1)//batch_size
    while (True):
        for i in range(batches):
            realdata = valrealdataset[i * batch_size: (i + 1) * batch_size]
            labledata = vallabledataset[i * batch_size: (i + 1) * batch_size]
            yield realdata, labledata

model = PConvUnet()
g = model.g
d1 = model.d1
d2 = model.d2
d_on_g = model.generator_containing_discriminator_multiple_outputs2(g,d1,d2)


output_true_batch, output_false_batch = np.ones((batch_size, 1)), np.zeros((batch_size,1))
output_true_batch2, output_false_batch2 = np.ones((batch_size, 1)), np.zeros((batch_size,1))

for epoch in range(n_epoch):
    print('epoch: {}/{}'.format(epoch, n_epoch))
    #print('batches: {}'.format(x_train.shape[0] / batch_size))

    #permutated_indexes = np.random.permutation(realdataset.shape[0])

    d_losses = []
    d_losses_fake = []
    d_losses_real = []
    d_losses2 = []
    d_losses2_fake = []
    d_losses2_real = []
    d_on_g_losses = []
    for index in range(5733):

        image_blur_batch,image_rain_batch,image_disp_batch,image_res_batch,image_real_batch = generate_arrays_from_file(txtpath,batch_size,index)

        image_disp_batch = np.reshape(image_disp_batch, image_disp_batch.shape + (1,))


        disp_image,resrain,generated_images = g.predict(x=[image_blur_batch,image_rain_batch], batch_size=batch_size)


        generated_rain_batch = resrain + generated_images

        for _ in range(critic_updates):
            
            d_loss_real = d1.train_on_batch(image_real_batch, output_true_batch)
            d_loss_fake = d1.train_on_batch(generated_images, output_false_batch)
            #d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            #d_losses.append(d_loss)
            d_losses_real.append(d_loss_real)
            d_losses_fake.append(d_loss_fake)
        print('batch {} d_loss_real : {} d_loss_fake : {}'.format(index+1, np.mean(d_losses_real),np.mean(d_losses_fake)))
        
        d1.trainable = False


        for _ in range(critic_updates):
            d_loss_real2 = d2.train_on_batch(image_rain_batch,
                                           output_true_batch2)
            d_loss_fake2 = d2.train_on_batch(generated_rain_batch,
                                           output_false_batch2)
            #d_loss2 = 0.5 * np.add(d_loss_fake2, d_loss_real2)
            #d_losses2.append(d_loss2)
            d_losses2_real.append(d_loss_real2)
            d_losses2_fake.append(d_loss_fake2)
        print('batch {} d_loss2_real : {} d_loss2_fake : {}'.format(index + 1, np.mean(d_losses2_real),np.mean(d_losses2_fake)))
        
        d2.trainable = False

        d_on_g_loss = d_on_g.train_on_batch([image_blur_batch,image_rain_batch,image_real_batch,image_res_batch], [image_disp_batch,image_res_batch,image_real_batch,output_true_batch,output_true_batch2])

        d_on_g_losses.append(d_on_g_loss)
        print('batch {} d_on_g_loss : {}'.format(index+1, d_on_g_loss))
        d1.trainable = True
        d2.trainable = True
        #with open('log.txt', 'a') as f:
        #    f.write('epoch {} d_loss_real : {} d_loss_fake : {} d_loss2_real : {} d_loss2_fake : {} d_on_g_loss : {}\n'.format(epoch, np.mean(d_losses_real),np.mean(d_losses_fake), np.mean(d_losses2_real),np.mean(d_losses2_fake), np.mean(d_on_g_losses)))
        if index == 500:
            save_all_weights(d1, d2, g, epoch, int(np.mean(d_on_g_losses)))


