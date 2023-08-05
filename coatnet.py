from tensorflow import keras 
import tensorflow as tf 
import numpy as np 
from utils import act_layer_factory, norm_layer_factory


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
                tf.keras.layers.Conv2D(filters=hidden_dim,
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
                tf.keras.layers.Conv2D(filters=hidden_dim,
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

    def call(self, x):
        shortcut = x

        x = self.pre_norm(x)

        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)

        else:
            return shortcut + self.conv(x)


class Attention(tf.keras.layers.Layer):
  def __init__(self,
               inp_dims,
               out_dims,
               num_heads,
               head_dims,
               img_size,
               attn_drop,
               proj_drop,
               qkv_bias=False,
               **kwargs
            ):

    super(Attention, self).__init__(**kwargs)
    inner_dim = head_dims * num_heads
    self.inner_dim = inner_dim
    self.inp_dims = inp_dims
    self.num_heads = num_heads
    self.head_dims = head_dims

    self.ih, self.iw = img_size # self.ih, self.iw -> ih = image_height, iw -> image_width

    self.qkv = tf.keras.layers.Dense(
            inner_dim * 3, use_bias=qkv_bias, name="qkv"
        )
    self.attn_drop = tf.keras.layers.Dropout(attn_drop)
    self.proj = tf.keras.layers.Dense(out_dims, name="proj")
    self.proj_drop = tf.keras.layers.Dropout(proj_drop)


  def build(self, input_shape):

    # The weights have to be created inside the build() function for the right
    # name scope to be set.
    self.relative_position_bias_table = self.add_weight(
          name="relative_position_bias_table",
          shape=((2 * self.ih - 1) * (2 * self.iw - 1), self.num_heads),
          initializer=tf.initializers.Zeros(),
          trainable=True,
        )

    coords_h = np.arange(self.ih)
    coords_w = np.arange(self.iw)
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))
    coords_flatten = coords.reshape(2, -1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.transpose((1, 2, 0))
    relative_coords[:, :, 0] += self.ih - 1
    relative_coords[:, :, 1] += self.iw - 1
    relative_coords[:, :, 0] *= 2 * self.iw - 1
    relative_position_index = relative_coords.sum(-1).astype(np.int64)
    self.relative_position_index = tf.Variable(
          name="relative_position_index",
          initial_value=tf.convert_to_tensor(relative_position_index),
          trainable=False,
      )

  def call(self, x, training=False):
    _, n, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
    nb_heads = self.num_heads
    window_size = self.ih * self.iw

    # Inputs are the batch and the attention mask
    _, n, c = tf.unstack(tf.shape(x))

    qkv = self.qkv(x)
    qkv = tf.reshape(qkv, shape=(-1, n, 3, nb_heads, self.head_dims))

    qkv = tf.transpose(qkv, perm=(2, 0, 3, 1, 4))
    q, k, v = tf.unstack(qkv)

    scale = (self.inp_dims // self.num_heads) ** -0.5
    q = q * scale
    attn = q @ tf.transpose(k, perm=(0, 1, 3, 2))
    relative_position_bias = tf.gather(
        self.relative_position_bias_table,
        tf.reshape(self.relative_position_index, shape=(-1,)),
      )
    relative_position_bias = tf.reshape(
        relative_position_bias,
        shape=(window_size, window_size, -1),
      )
    relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
    attn = attn + tf.expand_dims(relative_position_bias, axis=0)

    attn = tf.nn.softmax(attn, axis=-1)
    attn = self.attn_drop(attn, training=training)

    x = tf.matmul(attn, v)
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    x = tf.reshape(x, shape=(1, n, self.inner_dim))

    x = self.proj(x)
    x = self.proj_drop(x, training=training)
    return x


class Transformer(tf.keras.layers.Layer):
    def __init__(self, 
                 inp, 
                 oup, 
                 img_size, 
                 num_heads=8, 
                 head_dims=32, 
                 downsample=False, 
                 dropout=0., 
                 norm_layer="layer_norm"
              ):
      
        super(Transformer, self).__init__()
        hidden_dim = int(inp * 4)

        self.out_dim = oup

        self.ih, self.iw = img_size
        self.downsample = downsample
        norm_layer = norm_layer_factory(norm_layer)

        if self.downsample:
            self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
            self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
            self.proj = tf.keras.layers.Conv2D(filters=oup, 
                                               kernel_size=1, 
                                               strides=1, 
                                               padding='valid', 
                                               use_bias=False
                                            )

        self.attn = Attention(inp, 
                              oup, 
                              num_heads, 
                              head_dims, 
                              img_size=img_size, 
                              attn_drop=dropout, 
                              proj_drop=dropout
                          )
        
        self.ff = MLP(hidden_dim,
                      oup, 
                      drop_rate=dropout, 
                      act_layer="gelu"
                    )
        
        self.norm1 = norm_layer(name="norm1")
        self.norm2 = norm_layer(name="norm2")

    def call(self, x):
        shortcut = x 

        if self.downsample:
          x = self.pool2(x)
          shortcut = self.pool1(shortcut)
          shortcut = self.proj(shortcut)

        # reshape
        _, h, w, c = tf.shape(x)
        x = tf.reshape(x, (-1, h*w, c))

        # residual 1 
        x = self.norm1(x)
        x = self.attn(x)
        x = tf.reshape(x, (-1, self.ih, self.iw, self.out_dim))
        x = shortcut + x

        shortcut = x 

        print(x.shape, "after first residual block")

        # residual 2 
        x = tf.reshape(x, (-1, self.ih*self.iw, self.out_dim))
        x = self.norm2(x)
        print(x.shape)
        x = self.ff(x)
        print(x.shape, 'ffn')
        x = tf.reshape(x, (-1, self.ih, self.iw, self.out_dim))
        x = shortcut + x 

        return x 

class CoAtNet(tf.keras.models.Model):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=1000, block_types=['C', 'C', 'T', 'T']):
        super(CoAtNet, self).__init__()
        self.block_types = block_types
        self.in_channels = in_channels
        self.ih, self.iw = image_size # iw - image_width, ih - image_height
        self.block = {'C': MBConv, 'T': Transformer}

        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.head = tf.keras.layers.Dense(units=num_classes, use_bias=False, name="classification_head")

    def call(self, x):

        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        print(x.shape, "======x=====")

        x = self.pool(x)
        x = self.head(x)
        return x

    def build(self, input_shape):
      self.s0 = self._make_layer(
            Conv_3x3_bn, self.in_channels, channels[0], num_blocks[0], (self.ih // 2, self.iw // 2))
      self.s1 = self._make_layer(
            self.block[self.block_types[0]], channels[0], channels[1], num_blocks[1], (self.ih // 4, self.iw // 4))
      self.s2 = self._make_layer(
            self.block[self.block_types[1]], channels[1], channels[2], num_blocks[2], (self.ih // 8, self.iw // 8))
      self.s3 = self._make_layer(
            self.block[self.block_types[2]], channels[2], channels[3], num_blocks[3], (self.ih // 16, self.iw // 16))
      self.s4 = self._make_layer(
            self.block[self.block_types[3]], channels[3], channels[4], num_blocks[4], (self.ih // 32, self.iw // 32))

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = []
        for i in range(depth):
            if i == 0:
                print(block(inp, oup, image_size, downsample=True))
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))

        return tf.keras.Sequential(layers) 




