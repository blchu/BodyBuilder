import tensorflow as tf

INITIALIZER = tf.random_normal_initializer() #tf.contrib.layers.xavier_initializer()
PADDING = 'SAME'
MIN_CLIP_VALUE = -1
MAX_CLIP_VALUE = 1

def conv2d(x, filter_size, stride, feature_map_dim, name, var_dict=None):
    with tf.variable_scope(name, initializer=INITIALIZER):
        # create weight variable and convolve
        filter_dims = [filter_size, filter_size, x.get_shape()[-1], feature_map_dim]
        stride_dims = [1, stride, stride, 1]
        W = tf.get_variable('weights', filter_dims)
        conv = tf.nn.conv2d(x, W, stride_dims, padding=PADDING)

        # add bias and relu activation
        b = tf.get_variable('bias', [feature_map_dim])
        h = tf.nn.relu(tf.add(conv, b))

    if var_dict is not None:
        var_dict[W.name] = W
        var_dict[b.name] = b

    return h

def pool(x, ksize, stride, name):
    with tf.variable_scope(name):
        # max pooling
        window_dims = [1, ksize, ksize, 1]
        stride_dims = [1, stride, stride, 1]
        pool = tf.nn.max_pool(x, window_dims, stride_dims, padding=PADDING)

    return pool

def fc(x, output_dim, name, var_dict=None, activation=True):
    with tf.variable_scope(name, initializer=INITIALIZER):
        # create weight variable and matrix multiply
        weight_shape = [x.get_shape()[-1], output_dim]
        W = tf.get_variable('weights', weight_shape)
        mm = tf.matmul(x, W)

        # add bias and relu activation (if true)
        b = tf.get_variable('bias', [output_dim])
        add_op = tf.add(mm, b)
        h = tf.nn.relu(add_op) if activation else add_op

    if var_dict is not None:
        var_dict[W.name] = W
        var_dict[b.name] = b

    return h

def dropout(x, keep_prob, name):
    with tf.variable_scope(name):
        drop = tf.nn.dropout(x, keep_prob)

    return drop

def softmax(x, output_dim, name, var_dict=None):
    with tf.variable_scope(name, initializer=INITIALIZER):
        # create weight variable and matrix multiply
        weight_shape = [x.get_shape()[-1], output_dim]
        W = tf.get_variable('weights', weight_shape)
        mm = tf.matmul(x, W)

        # add bias and relu activation (if true)
        b = tf.get_variable('bias', [output_dim])
        h = tf.nn.softmax(tf.add(mm, b))

    if var_dict is not None:
        var_dict[W.name] = W
        var_dict[b.name] = b

    return h

def cross_entropy(predict, target, clip=False):
    with tf.variable_scope('cross_entropy'):
        ce = tf.reduce_mean(-tf.reduce_sum(target * tf.log(predict), reduction_indices=[1]))
        if clip:
            ce = tf.clip_by_value(ce, MIN_CLIP_VALUE, MAX_CLIP_VALUE)

    return ce

def mse(predict, target, clip=False):
    with tf.variable_scope('mse'):
        mse = tf.square(tf.sub(predict, target))
        if clip:
            mse = tf.clip_by_value(mse, MIN_CLIP_VALUE, MAX_CLIP_VALUE)

    return mse
