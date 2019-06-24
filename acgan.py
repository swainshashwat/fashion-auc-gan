from __future__ import print_function, division

from keras.layers import (Input, Dense, Reshape, Flatten,
                             Dropout, multiply)
from keras.layers import (BatchNormalization, Activation,
                            Embedding, ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np

class ACGAN():
    def __init__(self, input_rows, input_cols, input_channels,
                        input_classes):
        # Input shape
        self.img_rows = input_rows
        self.img_cols = input_cols
        self.channels = input_channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = input_classes
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy',
                 'sparse_categorical_crossentropy']

        # build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
                                optimizer=optimizer,
                                metrics=['accuracy'])

        # build the generator
        self.generator = self.build_generator()

        # generator input: noise and target label
        # generator ouput: generates corresponding label image
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # discriminator input: generated image
        # discriminator output: validity of image and image label
        valid, target_label = self.discriminator(img)

        # combined model( stacked generator and discriminator )
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses,
                            optimizer=optimizer)
    
    def build_generator(self):
        '''
        Build the Generator Model
        '''
        model = Sequential()

        # Adding Dense layer and Reshaping output
        model.add(Dense(7 * 7 * 128, activation='relu',
         input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        
        # Batch Normalization and Upsampling (1)
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        
        # Convolution and Activation layer (1)
        model.add(Conv2D(128, kernel_size=3, padding='same'))
        model.add(Activation('relu'))

        # Batch-Normalization and Upsampling (2)
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        
        # Convolution and Activation layer (2)
        model.add(Conv2D(64, kernel_size=3, padding='same'))
        model.add(Activation('relu'))

        # Batch-Normalization and Upsampling (3)
        model.add(BatchNormalization(momentum=0.8))
        
        # Convolving to the output shape
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        model.summary()

        # Defining Image(noise) shape and label shape
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')

        # embedding label
        embed = Embedding(self.num_classes, 100)(label)
        label_embedding = Flatten()(embed)

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)


    def build_discriminator(self):

        model = Sequential()
        
        # Simple Convolution Block with Leaky ReLU (1)
        model.add(Conv2D(16, kernel_size=3, strides=2, padding='same',
                            input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        
        # Convolution Block with Batch Normalization
        model.add(Conv2D(32, kernel_size=3, strides=2, padding='same'))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        # Convolution Block with Batch Normalization
        model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        # Simple Convolution Block with Leaky ReLU (1)
        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        
        model.add(Flatten())

        # plot model graph
        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determining validity and label of the Image
        validity = Dense(1, activation='sigmoid')(features)
        label = Dense(self.num_classes, activation='softmax')(features)

        return Model(img, [validity, label])


    def train(self, X, y, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train, y_train = X, y

        # Configuring inputs

        # Normalizing color pixels
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        # Adversial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))


        for epoch in range(epochs):

            # ------------ Train Discriminator ------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Labels of the digit that the generator tries to create
            # an image representation of
            sampled_labels = np.random.randint(0, 10, (batch_size, 1))

            # Generating half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels])

            # Image labels: 0-9
            img_labels = y_train[idx]

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs,
                                                [valid, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs,
                                                [fake, sampled_labels])

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------ Train Generator ------------

            # Train the generator
            g_loss = self.combined.train_on_batch([noise,sampled_labels],
                                                [valid, sampled_labels])

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%]\
                [G loss: %f"]
                      % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4],\
                         g_loss[0]))

            # Save generated image samples at intervals
            if epoch % sample_interval==0:
                self.save_model() # save model weights
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.array([num for _ in range(r)\
                                       for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])

        # rescale images 0-1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(32, 20))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i, j].axis('off')
                cnt+=1

        fig.savefig("gen_images/%d.png" %epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = 'saved_model/%s.json' % model_name
            weights_path = 'saved_model/%s_weights.hdf5' % model_name

            options = {"file_arch" : model_path,
                        "file_weight" : weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        
        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == "__main__":
    print('Loading ACGAN')
    acgan = ACGAN()
    acgan.train(epoch=14000, batch_size=32, sample_interval=200)