import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Dataset import *

class GAN():
    def __init__(self,gen_learn,disc_learn,gen_regularization_factor,disc_regularization_factor,number_of_discriminators,discriminator_dropout,IMAGE_SIZE,NOISE_SIZE):
        self.IMAGE_SIZE=IMAGE_SIZE
        self.NOISE_SIZE=NOISE_SIZE
        self.disc_regularization_factor=disc_regularization_factor
        self.gen_regularization_factor = gen_regularization_factor
        self.number_of_discriminators=number_of_discriminators
        self.discriminator_dropout=discriminator_dropout

        self.X=tf.placeholder(tf.float32,[None,IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]])
        self.noise=tf.placeholder(tf.float32,[None,1,1,NOISE_SIZE])
        self.discriminator_choice=tf.placeholder(tf.float32,[None])

        self.fake_images=self.generator(self.noise)
        self.plot_images=self.generator(self.noise)

        disc_out = self.discriminator(self.X,0)
        disc_fake_out = self.discriminator(self.fake_images,0)

        disc_fake_out=tf.reshape(disc_fake_out,[-1,1])

        self.gen_loss=-tf.log(disc_fake_out)

        self.disc_loss=[]
        self.disc_loss.append(-tf.reduce_mean(tf.log(disc_out) + tf.log(1 - disc_fake_out)))

        self.step_disc=[]
        self.step_disc.append(tf.train.AdamOptimizer(disc_learn).minimize(self.disc_loss[0],var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='DISCRIMINATOR0')))

        for i in range(1,number_of_discriminators):

            disc_out=self.discriminator(self.X,i)
            disc_fake_out=self.discriminator(self.fake_images,i)
            disc_fake_out = tf.reshape(disc_fake_out, [-1, 1])

            self.gen_loss=tf.concat([self.gen_loss,-tf.log(disc_fake_out)],axis=0)
            self.disc_loss.append(-tf.reduce_mean(tf.log(disc_out)+tf.log(1-disc_fake_out)))

            self.step_disc.append(tf.train.AdamOptimizer(disc_learn).minimize(self.disc_loss[i],var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='DISCRIMINATOR'+str(i))))

        self.gen_loss = tf.reduce_mean(self.gen_loss*tf.transpose(self.discriminator_choice))
        self.step_gen = tf.train.AdamOptimizer(gen_learn).minimize(self.gen_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='GENERATOR'))

        self.session = tf.Session()

        #SUMMARIES
        tf.summary.scalar('gen_loss',self.gen_loss)
        for i in range(0, number_of_discriminators):
            tf.summary.scalar('disc_loss'+str(i), self.disc_loss[i])
        self.merged_summaries=tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('./summaries', self.session.graph)

    def discriminator(self,X,index):
        with tf.variable_scope('DISCRIMINATOR'+str(index),reuse=tf.AUTO_REUSE):
            h=X

            regularization = tf.contrib.layers.l2_regularizer(self.disc_regularization_factor)


            h = tf.layers.conv2d(h, 128, 4,strides=2,padding='SAME',kernel_regularizer=regularization)
            h = tf.nn.relu(tf.layers.batch_normalization(h))

            h = tf.layers.conv2d(h, 256, 4, strides=2, padding='SAME', kernel_regularizer=regularization)
            h = tf.nn.relu(tf.layers.batch_normalization(h))

            h = tf.layers.conv2d(h, 512, 4, strides=2, padding='SAME', kernel_regularizer=regularization)
            h = tf.nn.relu(tf.layers.batch_normalization(h))

            h = tf.layers.conv2d(h, 1, 4, strides=1, padding='VALID', kernel_regularizer=regularization)
            h=tf.nn.sigmoid(h)
			
			print("discriminator")
            print(h.shape)

        return h
   
    def generator(self, noise):
        with tf.variable_scope('GENERATOR', reuse=tf.AUTO_REUSE):
            h = noise

            regularization = tf.contrib.layers.l2_regularizer(self.gen_regularization_factor)

            h = tf.layers.conv2d_transpose(h, 256, kernel_size=5, strides=1, kernel_regularizer=regularization)
            h = tf.nn.relu(tf.layers.batch_normalization(h))

            h = tf.layers.conv2d_transpose(h, 128, kernel_size=5, strides=2, kernel_regularizer=regularization)
            h = tf.nn.relu(tf.layers.batch_normalization(h))

            h = tf.layers.conv2d_transpose(h, 1, kernel_size=4, strides=2, kernel_regularizer=regularization)
            h = tf.nn.tanh(tf.layers.batch_normalization(h))

            print("generator")
            print(h.shape)

        return h
    def train(self,images,noise,iteration):
        discriminator_choice=np.random.uniform(0,1,[self.number_of_discriminators,1])
        discriminator_choice=np.where(discriminator_choice<self.discriminator_dropout,1,0)
        if 1 not in discriminator_choice:
            discriminator_choice[np.random.randint(0,self.number_of_discriminators)]=1
        discriminator_choice=np.repeat(discriminator_choice,images.shape[0])
        outs=self.session.run([self.gen_loss,self.disc_loss,self.merged_summaries,self.step_gen]+[step for step in self.step_disc],feed_dict={self.X:images,self.noise:noise,self.discriminator_choice:discriminator_choice})

        if iteration%10==0:
            self.train_writer.add_summary(outs[2],iteration)
        return outs[0],np.mean(outs[1])

    def get_new_images(self,noises):
        fake_images = self.session.run(self.plot_images, feed_dict={self.noise: noises})
        return fake_images.reshape([-1,self.IMAGE_SIZE[0],self.IMAGE_SIZE[1],self.IMAGE_SIZE[2]])
    def get_noises(self,num_noises):
        return np.random.normal(0, 1, size=[num_noises, 1, 1, self.NOISE_SIZE])

from tensorflow.examples.tutorials.mnist import input_data
from MNISTDataset import *
if __name__=='__main__':
    #'''
    #TEST DIO
    IMAGE_SIZE = (28, 28, 1)
    NOISE_SIZE = 100

    gan=GAN(0.0002,0.0002,0.0001,0.0001,1,1.,IMAGE_SIZE,NOISE_SIZE)

    saver=tf.train.Saver()
    saver.restore(gan.session,'./model_3disc/gan20000.ckpt')


    #8x8 RANDOM GRID
    noise=gan.get_noises(64)
    images=gan.get_new_images(noise)

    whole_image=np.zeros([28*8,28*8])
    for i in range(8):
        for j in range(8):
            whole_image[i*28:(i+1)*28,j*28:(j+1)*28]=images[i*8+j].reshape([IMAGE_SIZE[0], IMAGE_SIZE[1]])

    plt.imshow(whole_image,cmap=plt.get_cmap("gray"))
    plt.show()
    '''

    '''

    #SMOOTH INTERPOLATION
    while True:
        noise = gan.get_noises(2)
        fake_images = gan.get_new_images(noise)

        whole_image=np.zeros([28,280])
        for i in range(10):
            tau=i/10.
            interpolated_noise=(1-tau)*noise[0]+(tau)*noise[1]
            interpolated_image=gan.get_new_images(interpolated_noise.reshape([1,1,1,100]))

            whole_image[0:28,i*28:(i+1)*28]=interpolated_image[0].reshape([IMAGE_SIZE[0], IMAGE_SIZE[1]])

        plt.imshow(whole_image, cmap=plt.get_cmap("gray"))
        plt.show()

        input()
    #'''



    #TRAIN DIO
    '''
    tf.app.flags.DEFINE_string('data_dir', './tmp/data/', 'Directory for storing data')
    mnist = input_data.read_data_sets(tf.app.flags.FLAGS.data_dir, one_hot=True)


    IMAGE_SIZE=(28,28,1)
    NOISE_SIZE=100

    gan=GAN(0.0002,0.0002,0.0001,0.0001,1,1.,IMAGE_SIZE,NOISE_SIZE)

    ITERATIONS=30000
    BATCH_SIZE=32

    gan.session.run(tf.global_variables_initializer())

    saver=tf.train.Saver()

    same_noise = gan.get_noises(1)
    same_noise2 = gan.get_noises(1)

    for i in range(ITERATIONS):
        X,_ = mnist.train.next_batch(BATCH_SIZE)

        X=X.reshape([-1,IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]])
        noise=gan.get_noises(BATCH_SIZE)

        gen_loss,disc_loss=gan.train(X,noise,i)

        if i%200==0:
            print("%d: gen loss: %s disc loss: %s" % (i, gen_loss, disc_loss))
            same_fake_image = gan.get_new_images(same_noise)
            plt.imshow(same_fake_image[0].reshape([IMAGE_SIZE[0],IMAGE_SIZE[1]]) ,cmap=plt.get_cmap("gray"))
            plt.show()
            same_fake_image = gan.get_new_images(same_noise2)
            plt.imshow(same_fake_image[0].reshape([IMAGE_SIZE[0],IMAGE_SIZE[1]]), cmap=plt.get_cmap("gray"))
            plt.show()

        if i%5000==0:
            saver.save(gan.session, './model/gan' + str(i) + '.ckpt')

    noise = gan.get_noises(16)
    fake_images = gan.get_new_images(noise)

    for fake_image in fake_images:
        plt.imshow(fake_image.reshape([IMAGE_SIZE[0],IMAGE_SIZE[1]]), cmap=plt.get_cmap("gray"))
        plt.show()
    '''

