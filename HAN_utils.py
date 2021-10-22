import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

seed = 1224


def BiGRU(cell_fw, cell_bw, inputs, seq_len, name, dropout_keep_rate):
    with tf.variable_scope('rnn_' + name, reuse=tf.AUTO_REUSE):
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=dropout_keep_rate)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=dropout_keep_rate)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_fw, inputs=inputs,
                                                     sequence_length=seq_len, dtype=tf.float32)

        output_fw, output_bw = outputs
        output = tf.concat([output_fw, output_bw], axis=-1)
        return output


def task_specific_attention(inputs, output_size,
                            initializer=layers.xavier_initializer(),
                            activation_fn=tf.tanh, scope=None):
    """
    Performs task-specific attention reduction, using learned
    attention context vector (constant within task of interest).
    Args:
        inputs: Tensor of shape [batch_size, units, input_size]
            `input_size` must be static (known)
            `units` axis will be attended over (reduced from output)
            `batch_size` will be preserved
        output_size: Size of output's inner (feature) dimension
    Returns:
        outputs: Tensor of shape [batch_size, output_dim].
    """
    assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

    with tf.variable_scope(scope or 'attention') as scope:
        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                   shape=[output_size],
                                                   initializer=initializer,
                                                   dtype=tf.float32)
        input_projection = layers.fully_connected(inputs, output_size,
                                                  activation_fn=activation_fn,
                                                  scope=scope)

        vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keepdims=True)
        attention_weights = tf.nn.softmax(vector_attn, axis=1)
        weighted_projection = tf.multiply(input_projection, attention_weights)

        outputs = tf.reduce_sum(weighted_projection, axis=1)

    return outputs


def batch_iter(data, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]
