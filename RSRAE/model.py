import tensorflow as tf
import numpy as np

class CAE:

    def __init__(self, input_shape, hidden_layer_sizes, intrinsic_size, 
                 norm_type="MSE", loss_norm_type="MSE",
                 activation=tf.nn.relu, if_rsr=True, enforce_proj=False, all_alt=False,
                 lambda1=0.1, lambda2=0.1, learning_rate=0.001,
                 batch_size=128, epoch_size=200, batch_show=10,
                 normalize=False, bn=True, random_seed=123):
        
        """
        PARAMETERS:
            - input_shape: shape of input data, e.g. (28,28,1) for fashion mnist, (10000,) for 20news, etc.
            - hidden_layer_sizes: number of channels
            - intrinsic_size: dimension of the latent code z
            - norm_type: norm for rsr error (z-AA^Tz)
            - loss_norm_type: norm for error (x-x_tilde)
            - if_rsr: if true, impose rsr loss
            - lambda1: weight of the rsr penalty term
            - batch_show: show validation results after this number of epochs
            - normalize: if true, do l2 normalization for latent code z
        """

        self.input_shape = input_shape
        if len(input_shape) > 1:
            if self.input_shape[0] % 8 == 0:
                self.pad3 = 'same'
            else:
                self.pad3 = 'valid'
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.intrinsic_size = intrinsic_size
        self.activation = activation
        
        self.norm_type = norm_type
        self.loss_norm_type = loss_norm_type
        self.if_rsr = if_rsr
        self.enforce_proj = enforce_proj
        self.all_alt = all_alt
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.learning_rate = learning_rate
        
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.batch_show = batch_show
        
        self.normalize = normalize
        self.bn = bn
        self.seed = random_seed
        

    def encoder(self, x):
        if len(self.input_shape) > 1:
            with tf.variable_scope("encoder"):
                z = x
                z = tf.layers.conv2d(z, self.hidden_layer_sizes[0], kernel_size=5, strides=2, padding='same', 
                                     activation=self.activation, name="conv1")
                if self.bn:
                    z = tf.layers.batch_normalization(z)
                z = tf.layers.conv2d(z, self.hidden_layer_sizes[1], kernel_size=5, strides=2, padding='same', 
                                     activation=self.activation, name="conv2")
                if self.bn:
                    z = tf.layers.batch_normalization(z)
                z = tf.layers.conv2d(z, self.hidden_layer_sizes[2], kernel_size=3, strides=2, padding=self.pad3,
                                     activation=self.activation, name="conv3")
                if self.bn:
                    z = tf.layers.batch_normalization(z)
        else:
            with tf.variable_scope("encoder"):
                z = x
                z = tf.layers.dense(z, self.hidden_layer_sizes[0], activation=self.activation, name="fc1")
                if self.bn:
                    z = tf.layers.batch_normalization(z)
                z = tf.layers.dense(z, self.hidden_layer_sizes[1], activation=self.activation, name="fc2")
                if self.bn:
                    z = tf.layers.batch_normalization(z)
                z = tf.layers.dense(z, self.hidden_layer_sizes[2], activation=self.activation, name="fc3")
                if self.bn:
                    z = tf.layers.batch_normalization(z)
        return z
    
    def rsr(self, y):
        with tf.variable_scope("rsr"):
            y = tf.layers.Flatten()(y)
            A = tf.Variable(tf.random_normal((int(y.get_shape()[-1]), self.intrinsic_size)),
                            name="layer_rsr")
            self.A = A
            z = tf.matmul(y, A)
        return z
    
    
    def renormalization(self, y):
        with tf.variable_scope("renormalization"):
            z = tf.math.l2_normalize(y, axis=-1, name="renormalization")
        return z
    
    
    def decoder(self, z):
        if len(self.input_shape) > 1:
            with tf.variable_scope("decoder"):
                z = tf.layers.dense(z, self.hidden_layer_sizes[2]*int(self.input_shape[0]/8)*int(self.input_shape[0]/8),
                                    activation=self.activation, name="revealed")
                z = tf.reshape(z, (-1, int(self.input_shape[0]/8), int(self.input_shape[0]/8), self.hidden_layer_sizes[2]))
                if self.bn:
                    z = tf.layers.batch_normalization(z)
                z = tf.layers.conv2d_transpose(z, self.hidden_layer_sizes[1], kernel_size=3, strides=2, padding=self.pad3,
                                              activation=self.activation, name="deconv3")
                if self.bn:
                    z = tf.layers.batch_normalization(z)
                z = tf.layers.conv2d_transpose(z, self.hidden_layer_sizes[0], kernel_size=5, strides=2, padding='same',
                                              activation=self.activation, name="deconv2")
                if self.bn:
                    z = tf.layers.batch_normalization(z)
                x = tf.layers.conv2d_transpose(z, self.input_shape[2], kernel_size=5, strides=2, padding='same',
                                              activation=self.activation, 
                                              name="deconv1")
        else:
            with tf.variable_scope("decoder"):
                z = tf.layers.dense(z, self.hidden_layer_sizes[2], activation=self.activation, name="revealed")
                if self.bn:
                    z = tf.layers.batch_normalization(z)
                z = tf.layers.dense(z, self.hidden_layer_sizes[1], activation=self.activation, name="dfc3")
                if self.bn:
                    z = tf.layers.batch_normalization(z)
                z = tf.layers.dense(z, self.hidden_layer_sizes[0], activation=self.activation, name="dfc2")
                if self.bn:
                    z = tf.layers.batch_normalization(z)
                x = tf.layers.dense(z, self.input_shape[0], activation=self.activation, name="dfc1")
        return x

    def ae(self, x):
        y = self.encoder(x)
        y_rsr = self.rsr(y)
        if self.normalize:
            z = self.renormalization(y_rsr)
        else:
            z = y_rsr
        x_tilde = self.decoder(z)
        return y, y_rsr, z, x_tilde
    
    
    def reconstruction_error(self, x, x_tilde):    
        loss_norm_type = self.loss_norm_type
        x = tf.reshape(x, (tf.shape(x)[0], -1))
        x_tilde = tf.reshape(x_tilde, (tf.shape(x_tilde)[0], -1))
        if loss_norm_type in ['MSE', 'mse', 'Frob', 'F']:
            return tf.reduce_mean(tf.square(tf.norm(x-x_tilde, ord=2, axis=1)))
        elif loss_norm_type in ['L1', 'l1']:
            return tf.reduce_mean(tf.norm(x-x_tilde, ord=1, axis=1))
        elif loss_norm_type in ['LAD', 'lad', 'L21', 'l21', 'L2', 'l2']:
            return tf.reduce_mean(tf.norm(x-x_tilde, ord=2, axis=1))
        else:
            raise Exception("Norm type error!")
            
            
    def pca_error(self, y, z):
        norm_type = self.norm_type
        z = tf.matmul( z , tf.transpose(self.A) )
        if norm_type in ['MSE', 'mse', 'Frob', 'F']:
            return tf.reduce_mean(tf.square(tf.norm(y-z, ord=2, axis=1)))
        elif norm_type in ['L1', 'l1']:
            return tf.reduce_mean(tf.norm(y-z, ord=1, axis=1))
        elif norm_type in ['LAD', 'lad', 'L21', 'l21', 'L2', 'l2']:
            return tf.reduce_mean(tf.norm(y-z, ord=2, axis=1))
        else:
            raise Exception("Norm type error!")   


    def proj_error(self):
        return tf.reduce_mean( tf.square (tf.matmul(tf.transpose(self.A), self.A) - \
                                          tf.eye(self.intrinsic_size) ) ) 
            
            
    def reconstruction_loss(self, x, x_tilde):    
        norm_type = self.loss_norm_type
        x = tf.reshape(x, (tf.shape(x)[0], -1))        
        x_tilde = tf.reshape(x_tilde, (tf.shape(x_tilde)[0], -1))
        if norm_type in ['MSE', 'mse', 'Frob', 'F']:
            return tf.square(tf.norm(x-x_tilde, ord=2, axis=1))
        elif norm_type in ['L1', 'l1']:
            return tf.norm(x-x_tilde, ord=1, axis=1)
        elif norm_type in ['LAD', 'lad', 'L21', 'l21', 'L2', 'l2']:
            return tf.norm(x-x_tilde, ord=2, axis=1)
        else:
            raise Exception("Norm type error!")   
            
            
    def fit(self, x, x_val=None):
        if len(self.input_shape) > 1:
            n_samples, n_height, n_width, n_channel = x.shape
        else:
            n_samples, n_dim = x.shape
        
        with tf.Graph().as_default() as graph:
            
            self.graph = graph
            if self.seed is not None:
                tf.set_random_seed(self.seed)
                np.random.seed(seed=self.seed)
            
            if len(self.input_shape) > 1:
                self.input = x_input = tf.placeholder(
                        dtype=tf.float32, shape=[None, n_height, n_width, n_channel])
            else:
                self.input = x_input = tf.placeholder(
                        dtype=tf.float32, shape=[None, n_dim])
            
            y, y_rsr, z, x_tilde = self.ae(x_input)
            
            self.y = y = tf.layers.Flatten()(y)
            self.y_rsr = y_rsr
            self.z = z
            self.x_tilde = x_tilde
            
            if self.if_rsr and not self.all_alt:
                loss = self.reconstruction_error(x_input, x_tilde) + \
                       self.lambda1 * self.pca_error(y,y_rsr) + \
                       self.lambda2 * self.proj_error()
            else:
                loss = self.reconstruction_error(x_input, x_tilde)
            
            minimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
            minimizer2 = tf.train.AdamOptimizer(10 * self.learning_rate).minimize(self.proj_error())
            minimizer3 = tf.train.AdamOptimizer(10 * self.learning_rate).minimize(self.pca_error(y, y_rsr))


            n_batch = (n_samples - 1) // self.batch_size + 1
            
            init = tf.global_variables_initializer()
            self.sess = tf.Session(graph=graph)
            self.sess.run(init)
            
            idx = np.arange(n_samples)
            np.random.shuffle(idx)
            
            for epoch in range(self.epoch_size):
                for batch in range(n_batch):
                    i_start = batch * self.batch_size
                    i_end = (batch + 1) * self.batch_size
                    x_batch = x[ idx[ i_start : i_end ] ]
                    self.sess.run(minimizer, feed_dict={x_input: x_batch})

                    if self.enforce_proj and self.all_alt:
                    	self.sess.run(minimizer2, feed_dict={x_input: x_batch})

                    if self.all_alt:
                    	self.sess.run(minimizer3, feed_dict={x_input: x_batch})

                if self.batch_show is not None:
                    if (epoch + 1) % self.batch_show == 0:
                        if x_val is not None:
                            loss_val = self.sess.run(loss, feed_dict={x_input:x_val})
                            print(f" epoch {epoch+1}/{self.epoch_size} : loss = {loss_val:.3f}")
                        
    
    def get_latent(self, x):
        if self.sess is None:
            raise Exception("Trained model does not exist.")
        z = self.sess.run(self.z, feed_dict={self.input:x})
        return z
    
    def get_output(self, x):
        if self.sess is None:
            raise Exception("Trained model does not exist.")
        x_tilde = self.sess.run(self.x_tilde, feed_dict={self.input:x})
        return x_tilde
