import os
import numpy as np
from datetime import datetime
from keras.activations import *
from keras.models import *
from keras.models import load_model
from keras.optimizers import *
from keras.layers import *  # Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, BatchNormalization, Activation
from keras.layers.merge import Concatenate
from keras.applications import VGG16
from keras import backend as K
from libs.pconv_3dlayer import PConv3D


class PConvUnet(object):

    def __init__(self, img_rows=512, img_cols=512, weight_filepath=None, vgg_weights="imagenet", inference_only=False,
                 batchsize=4):
        """Create the PConvUnet. If variable image size, set img_rows and img_cols to None"""

        # Settings
        self.img_rows = 128
        self.img_cols = 128

        self.weight_filepath = weight_filepath

        self.img_overlap = 30
        self.inference_only = inference_only

        self.current_epoch = 0
        self.df = 64

        self.vgg_layers = [3, 6, 10]
        self.vgg = self.build_vgg(vgg_weights)
        self.batchsize = batchsize


        self.g = self.build_pconv_unet()
        self.d1 = self.discriminator_model()
        self.d2 = self.discriminator_model2()
        self.d_on_g = self.generator_containing_discriminator_multiple_outputs2(self.g, self.d1,self.d2)

    def build_vgg(self, weights="imagenet"):
        """
        Load pre-trained VGG16 from keras applications
        Extract features to be used in loss function from last conv layer, see architecture at:
        https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py
        """

        # Input image to extract features from
        img = Input(shape=(self.img_rows, self.img_cols, 3))

        # If inference only, just return empty model
        if self.inference_only:
            model = Model(inputs=img, outputs=[img for _ in range(len(self.vgg_layers))])
            model.trainable = False
            model.compile(loss='mse', optimizer='adam')
            return model

        # Get the vgg network from Keras applications
        if weights in ['imagenet', None]:
            vgg = VGG16(weights=weights, include_top=False, input_shape=(128, 128, 3))
        else:
            vgg = VGG16(weights=None, include_top=False, input_shape=(128, 128, 3))
            vgg.load_weights(weights)

        # Output the first three pooling layers
        vgg.outputs = [vgg.layers[i].output for i in self.vgg_layers]

        # Create model and compile
        model = Model(inputs=img, outputs=vgg(img))
        model.trainable = False
        model.compile(loss='mse', optimizer='adam')

        return model

    def build_pconv_unet(self, train_bn=True, lr=0.0002):

        # INPUTS
        inputs_img = Input((9, self.img_rows, self.img_cols, 3), name='inputs_img')

        inputs_blur = Input((9, self.img_rows, self.img_cols, 3), name='inputs_blur')

        conv00 = Conv3D(32, 3, activation=None, padding='same', kernel_initializer='he_normal',
                      data_format='channels_last')(inputs_blur)

        conv00 = LeakyReLU(alpha=1)(conv00)
        conv00 = BatchNormalization()(conv00, training=train_bn)

        conv12 = Conv3D(64, 3, activation=None, padding='same', kernel_initializer='he_normal',
                        data_format='channels_last')(conv00)

        conv12 = LeakyReLU(alpha=1)(conv12)
        conv12 = BatchNormalization()(conv12, training=train_bn)

        pool12 = MaxPooling3D((1, 2, 2))(conv12)

        pool12 = LeakyReLU(alpha=1)(pool12)

        ### non-local
        indata1 = Conv3D(64, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation=None, padding='same',
                     kernel_initializer='he_normal', data_format='channels_last')(pool12)

        indata1 = LeakyReLU(alpha=0.05)(indata1)

        indata1 = MaxPooling3D((1,4,4))(indata1)

        indata2 = Conv3D(64, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation=None, padding='same',
                     kernel_initializer='he_normal', data_format='channels_last')(pool12)

        indata2 = LeakyReLU(alpha=0.05)(indata2)

        indata2 = MaxPooling3D((1, 4, 4))(indata2)

        indata1 = Lambda(reshape32, name='ureshape11')(indata1)


        indata2 = Lambda(reshape42, name='ureshape12')(indata2)


        f = Lambda(matmul, name='umatmul11')([indata1,indata2])

        f = Lambda(softmax2, name='usoftmax11')(f)

        indata3 = Conv3D(64, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation=None, padding='same',
                     kernel_initializer='he_normal', data_format='channels_last')(pool12)

        indata3 = LeakyReLU(alpha=0.05)(indata3)

        indata3 = MaxPooling3D((1, 4, 4))(indata3)

        g = Lambda(reshape12, name='ureshape13')(indata3)

        y = Lambda(matmul, name='umatmul12')([f, g])

        y = Lambda(reshape22, name='ureshape14')(y)

        y = Conv3D(64, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation=None, padding='same',
               kernel_initializer='he_normal', data_format='channels_last')(y)

        y = LeakyReLU(alpha=0.05)(y)

        y = UpSampling3D((1,4,4))(y)

        pool12 = Add()([pool12, y])

        ###

        conv22 = Conv3D(128, 3, activation=None, padding='same', kernel_initializer='he_normal',
                        data_format='channels_last')(pool12)

        conv22 = LeakyReLU(alpha=1)(conv22)

        conv22 = BatchNormalization()(conv22, training=train_bn)

        pool22 = MaxPooling3D((1, 2, 2))(conv22)

        pool22 = LeakyReLU(alpha=1)(pool22)

        ###

        conv332 = Conv3D(256, 3, activation=None, padding='same', kernel_initializer='he_normal',
                         data_format='channels_last')(pool22)

        conv332 = LeakyReLU(alpha=1)(conv332)

        conv332 = BatchNormalization()(conv332, training=train_bn)

        pool32 = MaxPooling3D((1, 2, 2))(conv332)

        pool32 = LeakyReLU(alpha=1)(pool32)

        conv42 = Conv3D(512, 3, activation=None, padding='same', kernel_initializer='he_normal',
                        data_format='channels_last')(pool32)

        conv42 = LeakyReLU(alpha=1)(conv42)

        conv42 = BatchNormalization()(conv42, training=train_bn)

        conv42 = Conv3D(1024, 3, activation=None, padding='same', kernel_initializer='he_normal',
                        data_format='channels_last')(conv42)

        conv42 = LeakyReLU(alpha=1)(conv42)

        up72 = UpSampling3D((1, 2, 2))(conv42)

        up72 = LeakyReLU(alpha=1)(up72)

        up72 = Conv3D(512, 3, activation=None, padding='same', kernel_initializer='he_normal',
                      data_format='channels_last')(up72)

        up72 = LeakyReLU(alpha=1)(up72)

        merge72 = concatenate([conv332, up72], axis=4)
        conv72 = Conv3D(256, 3, activation=None, padding='same', kernel_initializer='he_normal',
                        data_format='channels_last')(merge72)

        conv72 = LeakyReLU(alpha=1)(conv72)

        up82 = UpSampling3D((1, 2, 2))(conv72)


        up82 = LeakyReLU(alpha=1)(up82)

        up82 = Conv3D(128, 3, activation=None, padding='same', kernel_initializer='he_normal',
                      data_format='channels_last')(up82)

        up82 = LeakyReLU(alpha=1)(up82)

        merge82 = concatenate([conv22, up82], axis=4)
        conv82 = Conv3D(128, 3, activation=None, padding='same', kernel_initializer='he_normal',
                        data_format='channels_last')(merge82)

        conv82 = LeakyReLU(alpha=1)(conv82)

        up92 = UpSampling3D((1, 2, 2))(conv82)


        up92 = LeakyReLU(alpha=1)(up92)

        up92 = Conv3D(64, 3, activation=None, padding='same', kernel_initializer='he_normal',
                      data_format='channels_last')(up92)

        up92 = LeakyReLU(alpha=1)(up92)

        merge92 = concatenate([conv12, up92], axis=4)
        conv92 = Conv3D(64, 3, activation=None, padding='same', kernel_initializer='he_normal',
                        data_format='channels_last')(merge92)

        conv92 = LeakyReLU(alpha=1)(conv92)

        conv92 = Conv3D(3, 3, activation=None, padding='same', kernel_initializer='he_normal',
                        data_format='channels_last')(conv92)

        conv92 = LeakyReLU(alpha=1)(conv92)

        disp = Conv3D(1, 1, activation=None, data_format='channels_last')(conv92)

        disp = LeakyReLU(alpha=1)(disp)

        dispout = disp

        inputs = concatenate([inputs_img, dispout], axis=4)

        conv0 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       data_format='channels_last')(inputs)

        conv1 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       data_format='channels_last')(conv0)
        conv1 = BatchNormalization()(conv1, training=train_bn)

        conv1 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       data_format='channels_last')(conv1)
        conv1 = BatchNormalization()(conv1, training=train_bn)

        convd2 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=2,
                        data_format='channels_last')(conv0)

        convd2 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=2,
                        data_format='channels_last')(convd2)
        convd2 = BatchNormalization()(convd2, training=train_bn)

        convd3 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=3,
                        data_format='channels_last')(conv0)

        convd3 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=3,
                        data_format='channels_last')(convd3)
        convd3 = BatchNormalization()(convd3, training=train_bn)

        conv_dilat = concatenate([conv1, convd2, convd3], axis=4)

        conv_dilat = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=3,
                        data_format='channels_last')(conv_dilat)

        pool1 = MaxPooling3D((1, 2, 2))(conv_dilat)

        conv2 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       data_format='channels_last')(pool1)
        conv2 = BatchNormalization()(conv2, training=train_bn)


        pool2 = MaxPooling3D((1, 2, 2))(conv2)

        conv31 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                        data_format='channels_last')(pool2)
        conv31 = BatchNormalization()(conv31, training=train_bn)

        pool3 = MaxPooling3D((1, 2, 2))(conv31)

        conv41 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                        data_format='channels_last')(pool3)
        conv41 = BatchNormalization()(conv41, training=train_bn)


        pool4 = MaxPooling3D((1, 2, 2))(conv41)

        conv3 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       data_format='channels_last')(pool4)  # (1, 64, 64, 256)
        conv3 = BatchNormalization()(conv3, training=train_bn)


        pool5 = MaxPooling3D((1, 2, 2))(conv3)

        conv5 = Conv3D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       data_format='channels_last')(pool5)  # (1, 64, 64, 256)
        conv5 = BatchNormalization()(conv5, training=train_bn)

        up6 = UpSampling3D((1, 2, 2))(conv5)
        up6 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                     data_format='channels_last')(up6)

        up6 = BatchNormalization()(up6, training=train_bn)
        merge6 = concatenate([conv3, up6], axis=4)
        conv6 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       data_format='channels_last')(merge6)

        conv6 = BatchNormalization()(conv6, training=train_bn)
        up7 = UpSampling3D((1, 2, 2))(conv6)
        up7 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                     data_format='channels_last')(up7)

        merge7 = concatenate([conv41, up7], axis=4)
        conv7 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       data_format='channels_last')(merge7)

        conv7 = BatchNormalization()(conv7, training=train_bn)

        up8 = UpSampling3D((1, 2, 2))(conv7)
        up8 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                     data_format='channels_last')(up8)

        up8 = BatchNormalization()(up8, training=train_bn)
        merge8 = concatenate([conv31, up8], axis=4)
        conv8 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       data_format='channels_last')(merge8)

        conv8 = BatchNormalization()(conv8, training=train_bn)


        up9 = UpSampling3D((1, 2, 2))(conv8)
        up9 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                     data_format='channels_last')(up9)

        up9 = BatchNormalization()(up9, training=train_bn)
        merge9 = concatenate([conv2, up9], axis=4)

        up10 = UpSampling3D((1, 2, 2))(merge9)
        up10 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                      data_format='channels_last')(
            up10)

        up10 = BatchNormalization()(up10, training=train_bn)
        merge10 = concatenate([conv1, up10], axis=4)

        conv9 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       data_format='channels_last')(merge10)
        conv9 = Conv3D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       data_format='channels_last')(conv9)
        conv10 = Conv3D(1, 1, activation='sigmoid', data_format='channels_last')(conv9)

        conv11 = concatenate([conv10, conv10, conv10], axis=4)

        resrain = conv11

        up62 = UpSampling3D((1, 2, 2))(conv5)
        up62 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                      data_format='channels_last')(up62)

        up62 = BatchNormalization()(up62, training=train_bn)
        merge62 = concatenate([conv3, up62], axis=4)
        conv62 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                        data_format='channels_last')(merge62)

        conv62 = BatchNormalization()(conv62, training=train_bn)
        up772 = UpSampling3D((1, 2, 2))(conv62)
        up772 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       data_format='channels_last')(up772)

        up772 = BatchNormalization()(up772, training=train_bn)
        merge772 = concatenate([conv41, up772], axis=4)
        conv772 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                         data_format='channels_last')(merge772)


        conv772 = BatchNormalization()(conv772, training=train_bn)
        up882 = UpSampling3D((1, 2, 2))(conv772)
        up882 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       data_format='channels_last')(up882)

        up882 = BatchNormalization()(up882, training=train_bn)
        merge882 = concatenate([conv31, up882], axis=4)
        conv882 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                         data_format='channels_last')(merge882)


        conv882 = BatchNormalization()(conv882, training=train_bn)
        up992 = UpSampling3D((1, 2, 2))(conv882)
        up992 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       data_format='channels_last')(
            up992)

        up992 = BatchNormalization()(up992, training=train_bn)
        merge992 = concatenate([conv2, up992], axis=4)

        up101 = UpSampling3D((1, 2, 2))(merge992)
        up101 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       data_format='channels_last')(up101)

        up101 = BatchNormalization()(up101, training=train_bn)
        merge101 = concatenate([conv1, up101], axis=4)

        conv992 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                         data_format='channels_last')(merge101)
        conv992 = Conv3D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                         data_format='channels_last')(conv992)
        conv102 = Conv3D(3, 1, activation='sigmoid', data_format='channels_last')(conv992)

        outputs = conv102


        model = Model(inputs=[inputs_blur, inputs_img], outputs=[disp, resrain, outputs], name='Generator')

        return model

    def discriminator_model(self):
        """Build discriminator architecture."""

        inputs_img = Input((9, self.img_rows, self.img_cols, 3), name='dis_inputs_img')

        dis_conv_block = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same')(inputs_img)

        dis_conv_block = Activation('relu')(dis_conv_block)
        dis_conv_block = Conv3D(32, (1, 3, 3), strides=(1, 2, 2), padding='same')(dis_conv_block)
        dis_conv_block = Activation('relu')(dis_conv_block)

        dis_conv_block = MaxPooling3D((1, 2, 2))(dis_conv_block)

        dis_conv_block = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', dilation_rate=1)(dis_conv_block)
        dis_conv_block = Activation('relu')(dis_conv_block)
        dis_conv_block = Conv3D(64, (1, 3, 3), strides=(1, 2, 2), padding='same')(dis_conv_block)
        dis_conv_block = Activation('relu')(dis_conv_block)

        dis_conv_block = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same', dilation_rate=2)(dis_conv_block)
        dis_conv_block = Activation('relu')(dis_conv_block)
        dis_conv_block = Conv3D(128, (1, 3, 3), strides=(1, 2, 2), padding='same')(dis_conv_block)
        dis_conv_block = Activation('relu')(dis_conv_block)

        dis_conv_block = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same', dilation_rate=3)(dis_conv_block)
        dis_conv_block = Activation('relu')(dis_conv_block)
        dis_conv_block = Conv3D(256, (1, 3, 3), strides=(1, 2, 2), padding='same')(dis_conv_block)
        dis_conv_block = Activation('relu')(dis_conv_block)

        flat_1 = Flatten()(dis_conv_block)

        dis_fc_0 = Dense(1024)(flat_1)
        dis_fc_0 = Activation('relu')(dis_fc_0)

        dis_fc_3 = Dense(1)(dis_fc_0)
        dis_similarity_output = Activation('sigmoid')(dis_fc_3)
        outputs = dis_similarity_output

        model = Model(inputs=inputs_img, outputs=outputs, name='Discriminator')
        # self.trainable = True
        model.compile(optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss='binary_crossentropy')
        return model

    def discriminator_model2(self):
        """Build discriminator architecture."""

        inputs_img = Input((9, self.img_rows, self.img_cols, 3), name='dis_inputs_img')


        dis_conv_block = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same')(inputs_img)

        dis_conv_block = Activation('relu')(dis_conv_block)
        dis_conv_block = Conv3D(32, (1, 3, 3), strides=(1, 2, 2), padding='same')(dis_conv_block)
        dis_conv_block = Activation('relu')(dis_conv_block)
        dis_conv_block = Conv3D(32, (1, 3, 3), strides=(1, 2, 2), padding='same')(dis_conv_block)
        dis_conv_block = Activation('relu')(dis_conv_block)

        dis_conv_block = MaxPooling3D((1, 2, 2))(dis_conv_block)

        dis_conv_block = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', dilation_rate=1)(dis_conv_block)
        dis_conv_block = Activation('relu')(dis_conv_block)
        dis_conv_block = Conv3D(64, (1, 3, 3), strides=(1, 2, 2), padding='same')(dis_conv_block)
        dis_conv_block = Activation('relu')(dis_conv_block)

        dis_conv_block = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same', dilation_rate=2)(dis_conv_block)
        dis_conv_block = Activation('relu')(dis_conv_block)
        dis_conv_block = Conv3D(128, (1, 3, 3), strides=(1, 2, 2), padding='same')(dis_conv_block)
        dis_conv_block = Activation('relu')(dis_conv_block)

        dis_conv_block = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same', dilation_rate=3)(dis_conv_block)
        dis_conv_block = Activation('relu')(dis_conv_block)
        dis_conv_block = Conv3D(256, (1, 3, 3), strides=(1, 2, 2), padding='same')(dis_conv_block)
        dis_conv_block = Activation('relu')(dis_conv_block)

        flat_1 = Flatten()(dis_conv_block)

        dis_fc_0 = Dense(1024)(flat_1)
        dis_fc_0 = Activation('relu')(dis_fc_0)

        dis_fc_3 = Dense(1)(dis_fc_0)
        dis_similarity_output = Activation('sigmoid')(dis_fc_3)
        outputs = dis_similarity_output

        model = Model(inputs=inputs_img, outputs=outputs, name='Discriminator2')
        # self.trainable = True
        model.compile(optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss='binary_crossentropy')
        return model

    def generator_containing_discriminator_multiple_outputs2(self, generator,discriminator1, discriminator2):
        inputs_img = Input((9, self.img_rows, self.img_cols, 3))
        inputs_real = Input((9, self.img_rows, self.img_cols, 3), name='inputs_real')
        inputs_blur = Input((9, self.img_rows, self.img_cols, 3), name='inputs_blur')
        inputs_realres = Input((9, self.img_rows, self.img_cols, 3), name='inputs_realres')

        disp_image, resrain_image, generated_image = generator([inputs_blur, inputs_img])

        rain_image = Add()([resrain_image, generated_image])

        outputs1 = discriminator1(generated_image)

        outputs2 = discriminator2(rain_image)
        model = Model(inputs=[inputs_blur,inputs_img,inputs_real,inputs_realres], outputs=[disp_image,resrain_image,generated_image,outputs1,outputs2])


        losstotal = self.loss_total(inputs_real)

        lossdisp = self.loss_disp()

        loss_res = self.loss_restotal(inputs_realres)
        discriminator1.trainable = False
        discriminator2.trainable = False
        loss = [lossdisp,loss_res, losstotal,'binary_crossentropy','binary_crossentropy']

        model.compile(optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss=loss,
                      loss_weights=[10, 10, 10, 1,1])
        discriminator1.trainable = True
        discriminator2.trainable = True
        return model

    def loss_total(self, inputs_real):
        def loss(y_true, y_pred):

            y_true = inputs_real

            l1 = 0
            l3 = 0
            l4 = 0


            for i in range(0, 9):
                y_pred_single = y_pred[:, i, :, :, :]
                y_true_single = y_true[:, i, :, :, :]


                vgg_out_single = self.vgg(y_pred_single)


                vgg_gt_single = self.vgg(y_true_single)


                l3 = l3 + self.loss_perceptual(vgg_out_single, vgg_gt_single)
                l4 = l4 + self.loss_style(vgg_out_single, vgg_gt_single)

            l1 = self.l1(y_true, y_pred)

            l6 = self.loss_tv(y_pred)

            return l1 + 0.05 * l3 + 0.05 * l4 + 0.1 * l6

        return loss

    def loss_restotal(self, inputs_real):
        def loss(y_true, y_pred):

            y_true = inputs_real

            l1 = 0
            l3 = 0
            l4 = 0


            for i in range(0, 9):
                y_pred_single = y_pred[:, i, :, :, :]
                y_true_single = y_true[:, i, :, :, :]


                vgg_out_single = self.vgg(y_pred_single)


                vgg_gt_single = self.vgg(y_true_single)


                l3 = l3 + self.loss_perceptual(vgg_out_single, vgg_gt_single)
                l4 = l4 + self.loss_style(vgg_out_single, vgg_gt_single)

            l1 = self.l1(y_true, y_pred)

            l6 = self.loss_tv(y_pred)

            return l1 + 0.05 * l3 + 0.05 * l4 + 0.1 * l6

        return loss

    def loss_disp2(self, inputs_real):
        def loss(y_true, y_pred):

            y_true = inputs_real

            l1 = 0
            l3 = 0
            l4 = 0

            for i in range(0, 9):
                y_pred_single = y_pred[:, i, :, :, :]
                y_true_single = y_true[:, i, :, :, :]


                vgg_out_single = self.vgg(y_pred_single)


                vgg_gt_single = self.vgg(y_true_single)


                l3 = l3 + self.loss_perceptual(vgg_out_single, vgg_gt_single)
                l4 = l4 + self.loss_style(vgg_out_single, vgg_gt_single)

            l1 = self.l1(y_true, y_pred)

            l6 = self.loss_tv(y_pred)

            return l1 + 0.05 * l3 + 0.05 * l4 + 0.1 * l6

        return loss

    def loss_img(self):
        def loss(y_true, y_pred):
            lossdisp = self.l1(y_true, y_pred)
            return lossdisp

        return loss

    def loss_disp(self):
        def loss(y_true, y_pred):
            lossdisp = self.l1(y_true, y_pred)
            return lossdisp

        return loss

    def loss_res(self):
        def loss(y_true, y_pred):
            lossres = self.l1(y_true, y_pred)
            return lossres

        return loss

    def loss_hole(self, mask, y_true, y_pred):
        """Pixel L1 loss within the hole / mask"""
        return self.l1((1 - mask) * y_true, (1 - mask) * y_pred)

    def loss_valid(self, mask, y_true, y_pred):
        """Pixel L1 loss outside the hole / mask"""
        return self.l1(mask * y_true, mask * y_pred)

    def loss_perceptual(self, vgg_out, vgg_gt):
        """Perceptual loss based on VGG16, see. eq. 3 in paper"""
        loss = 0
        for o, g in zip(vgg_out, vgg_gt):
            loss += self.l1(o, g)
        return loss

    def loss_style(self, output, vgg_gt):
        """Style loss based on output/computation, used for both eq. 4 & 5 in paper"""
        loss = 0
        for o, g in zip(output, vgg_gt):
            loss += self.l1(self.gram_matrix(o), self.gram_matrix(g))
        return loss

    def loss_tv(self, y_comp):
        """Total variation loss, used for smoothing the hole region, see. eq. 6"""

        # Create dilated hole region using a 3x3 kernel of all 1s.
        # if dtype(mask) == 'float64':
        '''
        mask = tf.cast(mask, 'float32')
        kernel = K.ones(shape=(3, 3, mask.shape[3], mask.shape[3]))

        dilated_mask = K.conv2d(1 - mask, kernel, data_format='channels_last', padding='same')

        # Cast values to be [0., 1.], and compute dilated hole region of y_comp
        dilated_mask = K.cast(K.greater(dilated_mask, 0), 'float32')
        P = dilated_mask * y_comp

        # Calculate total variation loss
        a = self.l1(P[:, 1:, :, :], P[:, :-1, :, :])
        b = self.l1(P[:, :, 1:, :], P[:, :, :-1, :])
        '''
        a = 0
        b = 0
        # Create dilated hole region using a 3x3 kernel of all 1s.
        for i in range(0, 9):
            # mask_one = mask[:,i,:,:,:]
            y_comp_one = y_comp[:, i, :, :, :]
            P = y_comp_one
            # print(y_comp_one.shape)
            # kernel = K.ones(shape=(3, 3, mask_one.shape[3], mask_one.shape[3]))
            # dilated_mask = K.conv2d(1-mask_one, kernel, data_format='channels_last', padding='same')
            # Cast values to be [0., 1.], and compute dilated hole region of y_comp
            # dilated_mask = K.cast(K.greater(dilated_mask, 0), 'float32')
            # P = dilated_mask * y_comp_one
            # Calculate total variation loss
            a = a + self.l1(P[:, 1:, :, :], P[:, :-1, :, :])
            b = b + self.l1(P[:, :, 1:, :], P[:, :, :-1, :])
        return a + b

    def wasserstein_loss(self, y_true, y_pred):
        wloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true))
        return wloss
        # return K.mean(y_true * y_pred)

    def fit(self, generator, epochs=10, plot_callback=None, *args, **kwargs):
        """Fit the U-Net to a (images, targets) generator

        param generator: training generator yielding (maskes_image, original_image) tuples
        param epochs: number of epochs to train for
        param plot_callback: callback function taking Unet model as parameter
        """

        # Loop over epochs
        for _ in range(epochs):

            # Fit the model
            self.model.fit_generator(
                generator,
                epochs=self.current_epoch + 1,
                initial_epoch=self.current_epoch,
                *args, **kwargs
            )

            # Update epoch
            self.current_epoch += 1

            # After each epoch predict on test images & show them
            if plot_callback:
                plot_callback(self.model)

            # Save logfile
            if self.current_epoch % 10 == 1:
                if self.weight_filepath:
                    self.save()

    def summary(self):
        """Get summary of the UNet model"""
        print(self.model.summary())

    def save(self):
        self.model.save_weights(self.current_weightfile())

    def load(self, filepath, train_bn=False, lr=0.0002):

        # Create UNet-like model
        self.model = self.build_pconv_unet(train_bn, lr)

        # Load weights into model
        # epoch = int(os.path.basename(filepath).split("_")[0])
        epoch = 1
        assert epoch > 0, "Could not parse weight file. Should start with 'X_', with X being the epoch"
        self.current_epoch = epoch
        self.model.load_weights(filepath)

    def current_weightfile(self):
        assert self.weight_filepath != None, 'Must specify location of logs'
        return self.weight_filepath + "{}_weights_{}.h5".format(self.current_epoch, self.current_timestamp())

    @staticmethod
    def current_timestamp():
        return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    @staticmethod
    def l1(y_true, y_pred):
        """Calculate the L1 loss used in all loss calculations"""
        if K.ndim(y_true) == 4:
            return K.sum(K.abs(y_pred - y_true), axis=[1, 2, 3])
        elif K.ndim(y_true) == 5:
            return K.sum(K.abs(y_pred - y_true), axis=[1, 2, 3, 4])
        elif K.ndim(y_true) == 3:
            return K.sum(K.abs(y_pred - y_true), axis=[1, 2])
        else:
            raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")

    @staticmethod
    def l2(y_true, y_pred):
        if K.ndim(y_true) == 4:
            return K.mean(K.square(y_pred - y_true), axis=[1, 2, 3])
        elif K.ndim(y_true) == 5:
            return K.mean(K.square(y_pred - y_true), axis=[1, 2, 3, 4])
        elif K.ndim(y_true) == 3:
            return K.mean(K.square(y_pred - y_true), axis=[1, 2])

    @staticmethod
    def gram_matrix(x, norm_by_channels=False):
        """Calculate gram matrix used in style loss"""

        # Assertions on input
        assert K.ndim(x) == 4, 'Input tensor should be a 4d (B, H, W, C) tensor'
        assert K.image_data_format() == 'channels_last', "Please use channels-last format"

        # Permute channels and get resulting shape
        x = K.permute_dimensions(x, (0, 3, 1, 2))
        shape = K.shape(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3]

        # Reshape x and do batch dot product
        features = K.reshape(x, K.stack([B, C, H * W]))
        gram = K.batch_dot(features, features, axes=2)

        # Normalize with channels, height and width
        gram = gram / K.cast(C * H * W, x.dtype)

        return gram

    # Prediction functions
    ######################
    def predict(self, sample, **kwargs):
        """Run prediction using this model"""
        return self.model.predict(sample, **kwargs)

    def scan_predict(self, sample, **kwargs):
        """Run prediction on arbitrary image sizes"""

        # Only run on a single image at a time
        img = sample[0]
        mask = sample[1]
        assert len(img.shape) == 3, "Image dimension expected to be (H, W, C)"
        assert len(mask.shape) == 3, "Image dimension expected to be (H, W, C)"

        # Chunk up, run prediction, and reconstruct
        chunked_images = self.dimension_preprocess(img)
        chunked_masks = self.dimension_preprocess(mask)
        pred_imgs = self.predict([chunked_images, chunked_masks], **kwargs)
        reconstructed_image = self.dimension_postprocess(pred_imgs, img)

        # Return single reconstructed image
        return reconstructed_image

    def perform_chunking(self, img_size, chunk_size):
        """
        Given an image dimension img_size, return list of (start, stop)
        tuples to perform chunking of chunk_size
        """
        chunks, i = [], 0
        while True:
            chunks.append(
                (i * (chunk_size - self.img_overlap / 2), i * (chunk_size - self.img_overlap / 2) + chunk_size))
            i += 1
            if chunks[-1][1] > img_size:
                break
        n_count = len(chunks)
        chunks[-1] = tuple(
            x - (n_count * chunk_size - img_size - (n_count - 1) * self.img_overlap / 2) for x in chunks[-1])
        chunks = [(int(x), int(y)) for x, y in chunks]
        return chunks

    def get_chunks(self, img):
        """Get width and height lists of (start, stop) tuples for chunking of img"""
        x_chunks, y_chunks = [(0, 512)], [(0, 512)]
        if img.shape[0] > self.img_rows:
            x_chunks = self.perform_chunking(img.shape[0], self.img_rows)
        if img.shape[1] > self.img_cols:
            y_chunks = self.perform_chunking(img.shape[1], self.img_cols)
        return x_chunks, y_chunks

    def dimension_preprocess(self, img):
        """
        In case of prediction on image of different size than 512x512,
        this function is used to add padding and chunk up the image into pieces
        of 512x512, which can then later be reconstructed into the original image
        using the dimension_postprocess() function.
        """

        # Assert single image input
        assert len(img.shape) == 3, "Image dimension expected to be (H, W, C)"

        # Check if height is too small
        if img.shape[0] < self.img_rows:
            padding = np.ones((self.img_rows - img.shape[0], img.shape[1], img.shape[2]))
            img = np.concatenate((img, padding), axis=0)

        # Check if width is too small
        if img.shape[1] < self.img_cols:
            padding = np.ones((img.shape[0], self.img_cols - img.shape[1], img.shape[2]))
            img = np.concatenate((img, padding), axis=1)

        # Get chunking of the image
        x_chunks, y_chunks = self.get_chunks(img)

        # Chunk up the image
        images = []
        for x in x_chunks:
            for y in y_chunks:
                images.append(
                    img[x[0]:x[1], y[0]:y[1], :]
                )
        images = np.array(images)
        return images

    def dimension_postprocess(self, chunked_images, original_image):
        """
        In case of prediction on image of different size than 512x512,
        the dimension_preprocess  function is used to add padding and chunk
        up the image into pieces of 512x512, and this function is used to
        reconstruct these pieces into the original image.
        """

        # Assert input dimensions
        assert len(original_image.shape) == 3, "Image dimension expected to be (H, W, C)"
        assert len(chunked_images.shape) == 4, "Chunked images dimension expected to be (B, H, W, C)"

        # Check if height is too small
        if original_image.shape[0] < self.img_rows:
            new_images = []
            for img in chunked_images:
                new_images.append(img[0:original_image.shape[0], :, :])
            chunked_images = np.array(new_images)

        # Check if width is too small
        if original_image.shape[1] < self.img_cols:
            new_images = []
            for img in chunked_images:
                new_images.append(img[:, 0:original_image.shape[1], :])
            chunked_images = np.array(new_images)

        # Put reconstruction into this array
        reconstruction = np.zeros(original_image.shape)

        # Get the chunks for this image
        x_chunks, y_chunks = self.get_chunks(original_image)
        i = 0
        for x in x_chunks:
            for y in y_chunks:
                prior_fill = reconstruction != 0

                chunk = np.zeros(original_image.shape)
                chunk[x[0]:x[1], y[0]:y[1], :] += chunked_images[i]
                chunk_fill = chunk != 0

                reconstruction += chunk
                reconstruction[prior_fill & chunk_fill] = reconstruction[prior_fill & chunk_fill] / 2

                i += 1

        return reconstruction


def minus(inputs):
    x, y = inputs
    return x - y


def matmul(inputs):
    indata1, indata2 = inputs
    return tf.matmul(indata1, indata2)



def softmax2(inputs):
    return softmax(inputs)


def reshape13(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [4, -1, 128])


def reshape23(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [-1, 9, 16, 16, 128])


def reshape33(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [4, -1, 128])


def reshape43(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [4, 128, -1])


def reshape12(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [4, -1, 64])


def reshape22(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [-1, 9, 16, 16, 64])


def reshape32(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [4, -1, 64])


def reshape42(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [4, 64, -1])


def reshape14(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [4, -1, 256])


def reshape24(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [-1, 9, 8, 8, 256])


def reshape34(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [4, -1, 256])


def reshape44(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [4, 256, -1])



def reshape1m(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [4, -1, 1024])


def reshape2m(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [-1, 9, 2, 2, 1024])


def reshape3m(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [4, -1, 1024])


def reshape4m(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [4, 1024, -1])

def reshape10(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [4, -1, 64])


def reshape20(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [-1, 9, 32, 32, 64])


def reshape30(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [4, -1, 64])


def reshape40(inputs):
    indata1 = inputs
    return tf.reshape(indata1, [4, 64, -1])