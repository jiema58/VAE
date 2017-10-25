
# coding: utf-8

import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

#load data 
data=pd.read_csv('C:/Users/jiema/Documents/data/Mnist/train.csv')
tests=pd.read_csv('C:/Users/jiema/Documents/data/Mnist/test.csv')

trains=data.iloc[:,1:].values
labels=data.iloc[:,:1].values
test=tests.iloc[:,:].values
train_f=trains/255 #as normalize pixel value to be between 0-1


class VAE:
    '''#### VAE model ####
    VAE is composed of 2 parts: 1.Encoder, 2.Decoder. 
    For this VAE model, both Encoder and Decoder employ 2 layers MLP network with activation function: softplus
    Loss function follows the original papaer as: KL loss - cross entropy. 
    KL loss is defined as distribution distance between latent posterior probability P(z|x) and proposed posterior probability Q(z|x)
    corss entropy is to caculate error between orginial image and generated image
    Args:
       input_size: image dimension (28*28)
       encoder_layer_x: encoder MLP layer x output size
       decoder_layer_x: decoder MLP layer x output size
       z_size: latent coder dimension
    '''
    def __init__(self,**param_size):
        self.data_size=param_size.get('input_size',784)
        self.encoder_layer_1=param_size.get('Encoder_layer_1',500)
        self.encoder_layer_2=param_size.get('Encoder_layer_2',500)
        self.decoder_layer_1=param_size.get('Decoder_layer_1',500)
        self.decoder_layer_2=param_size.get('Decoder_layer_2',500)
        self.z_size=param_size.get('z_size',20)
        self.lr=param_size.get('learning_rate',0.001)
        self.batch_size=param_size.get('batch_size',64)
        self.X=tf.placeholder(dtype=tf.float32,shape=[None,self.data_size])
        self.Z=tf.placeholder(dtype=tf.float32,shape=[None,self.z_size])
        
    def _get_weights(self,input_size,output_size):
        val=tf.sqrt(6./(input_size+output_size))
        w=tf.get_variable(name='weights',shape=[input_size,output_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        b=tf.get_variable(name='bias',shape=[output_size],dtype=tf.float32,initializer=tf.constant_initializer(0.))
        return (w,b)
    
    def _build_full_layers(self,data,ins,outs,name):
        with tf.variable_scope(name):
            (w,b)=self._get_weights(ins,outs)
            output=tf.matmul(data,w)+b
        return output
    
    def Encoder(self,reuse):
        with tf.variable_scope('Encoder',reuse=reuse):
            out1=self._build_full_layers(self.X,self.data_size,self.encoder_layer_1,name='encoder_layer1')
            out1=tf.nn.softplus(out1)
            out2=self._build_full_layers(out1,self.encoder_layer_1,self.encoder_layer_2,name='encoder_layer2')
            out2=tf.nn.softplus(out2)
            z_mean=self._build_full_layers(out2,self.encoder_layer_2,self.z_size,name='latent_mean')
            z_var=self._build_full_layers(out2,self.encoder_layer_2,self.z_size,name='latent_var')
        return z_mean,z_var
        
    def Decoder(self,z,reuse):
        with tf.variable_scope('Decoder',reuse=reuse):
            out1=self._build_full_layers(z,self.z_size,self.decoder_layer_1,name='decoder_layer1')
            out1=tf.nn.softplus(out1)
            out2=self._build_full_layers(out1,self.decoder_layer_1,self.decoder_layer_2,name='decoder_layer2')
            out2=tf.nn.softplus(out2)
            output=self._build_full_layers(out2,self.decoder_layer_2,self.data_size,name='output_layer')
            output=tf.nn.sigmoid(output)
        return output
        
    def cost(self,reuse):
        z_mean,z_var=self.Encoder(reuse)
        eps=tf.random_normal((self.batch_size,self.z_size),0,1,dtype=tf.float32)
        z=tf.add(z_mean,tf.multiply(tf.sqrt(tf.exp(z_var)),eps))
        x_=self.Decoder(z,reuse)
        KL_loss=-0.5*tf.reduce_sum(1.+z_var-tf.square(z_mean)-tf.exp(z_var),1) #KL divergence 
        E_loss=-tf.reduce_sum(self.X*tf.log(1e-10+x_)+(1.-self.X)*tf.log(1e-10+1.-x_),1) #cross entropy
        KL_loss=tf.reduce_mean(KL_loss)
        E_loss=tf.reduce_mean(E_loss)
        cost=KL_loss+E_loss #Loss: KL - cross entropy
        return cost,x_,E_loss,KL_loss
    
    def train(self,reuse):
        cost,x_,E_loss,KL_loss=self.cost(reuse)
        op=tf.train.AdamOptimizer(learning_rate=self.lr)
        train_op=op.minimize(cost)
        encoder_var_list=[]
        decoder_var_list=[]
        for var in tf.trainable_variables():
            if 'Encoder' in var.name:
                encoder_var_list.append(var)
            elif 'Decoder' in var.name:
                decoder_var_list.append(var)
        
        optim_e=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(E_loss,var_list=decoder_var_list)
        optim_kl=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(KL_loss,var_list=encoder_var_list)
        return train_op,cost,E_loss,KL_loss,optim_e,optim_kl
    
    def generate(self,reuse):
        x_=self.Decoder(self.Z,reuse)
        return x_

def train(x,vae,epoch):
    ''' Train VAE'''
    train_op,loss,E_loss,KL_loss,optim_e,optim_kl=vae.train(reuse=False)
    x_reconstruct=myvae.generate(reuse=True)
    m=len(x)
    total_batch=m//vae.batch_size
    init=tf.global_variables_initializer()
    saver=tf.train.Saver(tf.all_variables())
    with tf.Session() as sess:
        sess.run(init)
        e=0
        while e<epoch:
            vae.lr=0.001
            for i in range(total_batch):
                curr_batch=x[i*vae.batch_size:(i+1)*vae.batch_size]
                #_,e_l=sess.run([optim_e,E_loss],{vae.X:curr_batch})
                #_,kl=sess.run([optim_kl,KL_loss],{vae.X:curr_batch})
                _,cost,e_loss,kl=sess.run([train_op,loss,E_loss,KL_loss],feed_dict={vae.X:curr_batch})
                #if i%2==0:
                    #vae.lr*=0.97**(i/2)
                if e%2==0 and i%50==0:
                    print('***epoch {}, step {}: total loss: {}, E_loss: {}, KL_loss:{}'.format(e,i,cost,e_loss,kl))
                if i%100==0:
                    x_r=sess.run(x_reconstruct,{myvae.Z:np.random.normal(size=(1,myvae.z_size))})
                    img=x_r[0]
                    print('**********result at Epoch {} Step {}*************'.format(e,i))
                    plt.imshow(img.reshape(28,28))
                    plt.show()
            e+=1    
        saver.save(sess,os.path.join(os.getcwd(),'B_2_VAE.ckpt'))       


#training phase latent code dimension:2
myvae=VAE(batch_size=100,z_size=2)
train(train_f,myvae,80)

#reconstruct sample according to the input  
new=VAE(z_size=2)
cost,x_,E_loss,KL_loss=new.cost(reuse=True)
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,os.path.join(os.getcwd(),'B_2_VAE.ckpt'))
    x_reconstruct=sess.run(x_,{new.X:train_f[:1]})

for i in range(len(x_reconstruct)):
    img=x_reconstruct[i]
    plt.imshow(img.reshape(28,28))
    plt.show()

#The visilization of the distribution of the latent code with size 2. 
#It is quite clear that the latent codes from the same digit cluster while the latent codes from different digit images are geometrically seperated 
new=VAE(batch_size=train_f.shape[0],z_size=2)
z_mean,z_var=new.Encoder(reuse=True)
saver=tf.train.Saver()
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess,os.path.join(os.getcwd(),'B_2_VAE.ckpt'))
    z_m=sess.run(z_mean,{new.X:train_f})
plt.figure(figsize=(8,6))
plt.scatter(z_m[:,0],z_m[:,1],c=labels.reshape(len(labels)))
plt.title('2D latent Space')
plt.colorbar()
plt.grid()
plt.show()

#We sample a group of 2-D latent codes from the uniform distribution to generate images. 
new=VAE(batch_size=1,z_size=2)
output=new.generate(reuse=False)
nx=ny=20
x_values=np.linspace(-3,3,nx)
y_values=np.linspace(-3,3,ny)
canvas=np.empty((28*ny,28*nx))
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,os.path.join(os.getcwd(),'B_2_VAE.ckpt'))
    for i,yi in enumerate(x_values):
        for j,xi in enumerate(y_values):
            z_mu=np.array([[xi,yi]]*new.batch_size)
            x_mean=sess.run(output,{new.Z:z_mu})
            canvas[(nx-i-1)*28:(nx-i)*28,j*28:(j+1)*28]=x_mean[0].reshape(28,28)

plt.figure(figsize=(8,10))
Xi,Yi=np.meshgrid(x_values,y_values)
plt.imshow(canvas,origin='upper',cmap='gray')
plt.title('latent space to generate images')
plt.tight_layout()  
plt.show()
