from tensorflow import keras 
import tensorflow as tf 
import numpy as np 


def conv_3x3_bn(out_dims, 
                image_size, 
                downsample=False):
  
  stride = 1 if downsample == False else 2

  input_ = tf.keras.layers.Input(shape=image_size)

  x = tf.keras.layers.Conv2D(
      kernel_size = 3, 
      filters = out_dims,
      strides = stride,
      use_bias = False
    )(input_)
  
  x = tf.keras.layers.BatchNormalization()(x)
  
  output_ = tf.nn.gelu(x)

  return tf.keras.models.Model(inputs=input_, outputs=output_)


class PreNorm(tf.keras.layers.Layer):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def call(self, x):
        return self.fn(self.norm(x))


class SE(tf.keras.layers.Layer):
  """
      Squeeze and Excitation module.
  """
  def __init__(self, inp_dims, out_dims, expansion=0.25):
        super().__init__()
        self.gap = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis = -1, keepdims = True))
        self.squeeze = tf.keras.layers.Dense(int(inp_dims * expansion), 
                                         use_bias=False)
        self.activation = tf.nn.gelu
        self.excitation = tf.keras.layers.Dense(out_dims, 
                                         use_bias=False)
        self.sigmoid = tf.nn.sigmoid

  def call(self, x, training=False):
        b, c, _, _ = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        y = self.gap(x)
        y = self.squeeze(y)
        y = self.activation(y)
        y = self.excitation(y)
        y = self.sigmoid(y)

        print(y.shape, x.shape)

        return x * y


class MLP(tf.keras.layers.Layer):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        hidden_dim: int,
        projection_dim: int,
        drop_rate: float,
        act_layer: str,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        **kwargs,
    ):
        super(MLP, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim 
        self.projection_dim = projection_dim 
        self.drop_rate = drop_rate
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        act_layer = act_layer_factory(act_layer)

        self.fc1 = tf.keras.layers.Dense(
            units=hidden_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="fc1",
        )
        self.act = act_layer()
        self.drop1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.fc2 = tf.keras.layers.Dense(
            units=projection_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="fc2",
        )
        self.drop2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x

    def get_config(self):
        config = super(MLP, self).get_config()
        config["hidden_dim"] = self.hidden_dim
        config["projection_dim"] = self.projection_dim
        config["drop_rate"] = self.drop_rate
        config["kernel_initializer"] = self.kernel_initializer
        config["bias_initializer"] = self.bias_initializer
        return config
                              

class MBConv(tf.keras.layers.Layer):
    def __init__(self, 
                 inp_dims, 
                 out_dims, 
                 image_size, 
                 downsample=False, 
                 expansion=4,
                 **kwargs
              ):
      
        super(MBConv, self).__init__(**kwargs)
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp_dims * expansion)

        if self.downsample:
            self.pool = tf.keras.layers.MaxPooling2D(pool_size=3, 
                                                     strides=2, 
                                                     padding='same')
            
            self.proj = tf.keras.layers.Conv2D(filters=out_dims, 
                                               kernel_size=1, 
                                               strides=1, 
                                               padding="valid", 
                                               use_bias=False)
        
        if expansion == 1:
            self.conv = tf.keras.models.Sequential([
                # dw
                tf.keras.layers.DepthwiseConv2D(depth_multiplier=hidden_dim, 
                                                kernel_size=3, 
                                                strides=stride, 
                                                padding="same", 
                                                use_bias=False),
                
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.keras.activations.gelu),

                # pw-linear
                tf.keras.layers.Conv2D(filters=out_dims, 
                          kernel_size=1, 
                          strides=1, 
                          padding='valid', 
                          use_bias=False),
                
                tf.keras.layers.BatchNormalization(),
           ])
            
        else:
            self.conv = tf.keras.models.Sequential([
                # pw
                # down-sample in the first conv
                tf.keras.layers.Conv2D(filters=hidden_dim, 
                          kernel_size=1, 
                          strides=stride, 
                          padding='valid', 
                          use_bias=False),
                
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.keras.activations.gelu),

                # dw
                tf.keras.layers.DepthwiseConv2D(depth_multiplier=hidden_dim, 
                                                kernel_size=3, 
                                                strides=1, 
                                                padding="same", 
                                                use_bias=False),

                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.keras.activations.gelu),
                SE(inp_dims, hidden_dim),

                # pw-linear
                tf.keras.layers.Conv2D(filters=out_dims, 
                          kernel_size=1, 
                          strides=1, 
                          padding='valid', 
                          use_bias=False),
                
                tf.keras.layers.BatchNormalization(),
            ])
        
        self.pre_norm = tf.keras.layers.BatchNormalization()
        #self.conv = PreNorm(inp_dims, self.conv, tf.keras.layers.BatchNormalization())

    def forward(self, x):
        shortcut = x 

        x = self.pre_norm(x)

        if self.downsample: 
            return self.proj(self.pool(x)) + self.conv(x)

        else: 
            return shortcut + self.conv(x) 