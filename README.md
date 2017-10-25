# VAE

##A vanila VAE model

Build a vanila VAE model. It is composed of 2 parts: 1.Encoder 2.Decoder

1.Encoder - a MLP with 2 hidden layers. Activation function: softplus

2.Decoder - a MLP with 2 hidden layers. Activation function: softplus

Thanks to [1] to deliver the inspiration of Bernoulli cross-entropy 

##Dependencies

1. Python - 3.6

2. Tensorflow - 1.0
 
##Result:

Latent Code Distribution:

![latent code](https://github.com/WoshidaCaiB/VAE/blob/master/image/latent.png)

The dimension of latent codes has an impact on VAE performance. I trained 3 models with latent dimension: 2, 20, 100. 20 model renders the best performance while 100 model ranks the last

##Reference:

[1].http://jmetzen.github.io/2015-11-27/vae.html



