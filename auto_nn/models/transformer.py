#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************************************************************
# @file: transformer.py
# @brief: 
# @author: niezhipeng
# @Created on 2018/10/13
# *************************************************************************************
import numpy as np
import tensorflow as tf
from common import tf_utils
from common.text_box import TextProcessBox
from models.basic_model import BasicModel
from common import layers as tf_layers
from tensorflow import set_random_seed
set_random_seed(1)

#layer_norm = tf.contrib.layer_norm
layer_norm = tf_utils.layer_norm
PAD = "<PAD>"
PAD_ID = 0
EOS = "<EOS>"
EOS_ID = 1

class Transformer(BasicModel):
    """ Transformer model by google,
        Refs: https://arxiv.org/pdf/1706.03762.pdf
        based on the official implementation ,
            https://github.com/tensorflow/models/tree/master/official/transformer
    """
    def __init__(self, feature_dims, target_dim_list, model_cfg):
        if isinstance(feature_dims, (list, tuple)):
            self._feature_dim_list = feature_dims
            self._feature_dim = len(feature_dims)
        else:
            self._feature_dim_list = [feature_dims]
            self._feature_dim = int(feature_dims)
        '''
        # load wordvec from file
        self._wordvec = TextProcessBox.read_wordvec(model_cfg["wordvec_path"])
        (self._vocab_size, self._wordvec_dim) = self._wordvec.shape
        if self._feature_dim_list[0] != self._wordvec_dim:
            print("[WARNING] change wordvec dim = %d" % self._wordvec_dim)
            self._feature_dim_list[0] = self._wordvec_dim
        '''
        self._wordvec = None
        self._vocab_size = int(model_cfg["vocab_size"])
        self._extra_decode_length = None

        super(Transformer, self).__init__(
            self._feature_dim, target_dim_list, model_cfg)

    def build_graph(self, scope=None):
        tf.set_random_seed(1)

        self._init_tensors()
        scope = scope or "model"
        with tf.name_scope(scope):
            with tf.name_scope("feature_embedding"):
                layer_in = self._embedding_feature_lookup(self._inputs)

            _, outputs_list = self._init_layers(layer_in, self._cfg)
            self._logits_list = outputs_list
            self._init_scoring()

        self.calc_loss(self._cfg["loss_weights"])

    def _init_layers(self, layer_in, model_cfg, scope=None):
        """
        args:
            layer_in: Tensor([B, T, num_features], tf.int32)
            model_cfg: DICT, {
                "encoder": DICT
                "output_layer": LIST for multi-task, hparams dict of decoder
            }
            scope:
        return:
        """
        hidden_layers = list()
        output_layers = list()

        hidden_size = int(model_cfg["encoder"]["hidden_size"])

        tf.summary.histogram("layer_in", layer_in)
        self.enc_paddings = tf_utils.get_padding_mask(layer_in)

        # FC for features
        _params = {"num_units": hidden_size, "activation": "linear"}
        enc_inputs, _ = tf_layers.apply_dense(layer_in, _params, dropout=0.,
                                              scope="input_project")
        hidden_layers.append(enc_inputs)

        # encoder, record the encoder outputs
        enc_outputs = self.encode(enc_inputs, model_cfg["encoder"],
                                  scope="encoder")
        self._enc_outputs = enc_outputs

        hidden_layers.append(enc_outputs)

        # prosody task
        _params = model_cfg["output_layer"][0]
        prosody_out = self.decode(
            self._targets_list[0], enc_outputs, _params, self.enc_paddings,
            scope="prosody_decode")
        output_layers.append(prosody_out)

        return hidden_layers, output_layers

    def encode(self, enc_inputs, hparams, enc_paddings=None,
               is_training=True, scope="encode"):
        """ self-attention encoder stack
        args:
            enc_inputs: Tensor([B, T, D], tf.float32)
            hparams: DICT, {
                "hidden_size":
                "num_heads":
                "num_blocks":
                "filter_size":
                "dropout": (optional)
            }
            enc_paddings: Tensor([B, T], tf.float)
            is_training: BOOL
            scope:
        return:
        """
        if "dropout" not in hparams:
            hparams["dropout"] = self._dropout
        print("[INFO] %s\n\tparams = %s" % (scope, str(hparams)))
        with tf.name_scope(scope):
            if enc_paddings is None:
                enc_paddings = tf_utils.get_padding_mask(enc_inputs)

            # positional encoding
            batch_size, time_steps, dim = tf.unstack(tf.shape(enc_inputs))
            enc_inputs += self.position_encoding(batch_size, time_steps, dim,
                                                 scope="encoder_position")

            block_in = tf.nn.dropout(enc_inputs, 1.0-hparams["dropout"])
            # encoder blocks stack
            for i in range(hparams["num_blocks"]):
                block_out, _ = self.encoder_block(
                    block_in, hparams, enc_paddings,
                    is_training=is_training,
                    scope="encoder_block_%d" % i)
                block_in = block_out
            enc_outputs = layer_norm(block_in, scope="encoder_out_norm")
        return enc_outputs

    def decode(self, targets, enc_outputs, hparams, enc_paddings=None,
               is_training=True, scope="decode"):
        """ attention_decoder + out_layer
        args:
            targets: Tensor([B, T, 1], tf.int32), with EOS in tail
            enc_outputs: Tensor([B, enc_seq_len, enc_hidden_size])
            hparams: DICT, {
                "num_classes": num_classes
                "max_seq_len": maximun of sequence length
                "hidden_size": hidden_size for decoder
                "num_blocks": number of attention blocks
                "num_heads": number of attention heads
                "filter_size": num_units for feed-forword networks
                "dropout":
            }
            scope:
        return:
            Tensor([B, T, vocab_size])
        """
        if "dropout" not in hparams:
            hparams["dropout"] = self._dropout
        max_len = hparams["max_seq_len"]
        num_classes = hparams["num_classes"]
        hidden_size = hparams["hidden_size"]
        print("[INFO] %s: num_classes = %d, max_seq_len = %d" %
              (scope, num_classes, max_len))
        with tf.name_scope(scope):
            batch_size = tf.shape(targets)[0]
            length = tf.shape(targets)[1]

            position_enc = self.position_encoding_v3(max_len, hidden_size)
            pos_enc_table = tf.get_variable("position_embedding", dtype=tf.float32,
                                            initializer=tf.to_float(position_enc))

            dec_inputs = self.embedding_lookup(
                targets, "label_embedding", [num_classes, hidden_size])

            dec_paddings = tf_utils.get_padding_mask(targets)

            with tf.name_scope("preprocess"):
                # zero-padding mask
                _mask = tf.tile(tf.expand_dims(dec_paddings, axis=-1),
                                [1, 1, hidden_size])
                dec_inputs = dec_inputs * _mask

                # Shift targets to the right
                dec_inputs = tf.pad(dec_inputs, [[0, 0], [1, 0], [0, 0]])
                dec_paddings = tf.pad(dec_paddings, [[0, 0], [1, 0]],
                                      constant_values=1.)

            # add positional encoding
            with tf.name_scope("add_pos_encoding"):
                dec_inputs += tf.tile(
                    tf.expand_dims(pos_enc_table[:length+1, :], axis=0),
                    [batch_size, 1, 1])

            block_in = tf.nn.dropout(dec_inputs, 1.0-hparams["dropout"])
            # decoder block stack
            for i in range(hparams["num_blocks"]):
                block_out, _, _ = self.decoder_block(
                    block_in, enc_outputs, hparams,
                    dec_paddings, enc_paddings,
                    is_training=is_training,
                    scope="decoder_block_%d" % i)
                block_in = block_out
            dec_outputs = layer_norm(block_in, scope="decoder_out_norm")
            print("[INFO] %s_decoder\n\tparams = %s" % (scope, str(hparams)))
            # output layer
            _params = {"num_units": num_classes, "activation": "linear"}
            logits, _ = tf_layers.apply_dense(dec_outputs, _params,
                is_training=is_training,
                dropout=hparams["dropout"],
                scope="output_layer")
            print("[INFO] %s_output_layer\n\tparams = %s" % (scope, str(_params)))
        return logits

    def predict(self):
        pass

    def encoder_block(self, inputs, hparams, paddings,
                      is_training=True, scope=None, reuse=False):
        """ multihead self-attention + feed-forward networks
        add & norm for each modules according the official implementation
        args:
            inputs: Tensor([B, T, D], tf.float32)
            hparams: DICT, {
                "hidden_size": INT, should be same with last dimension of inputs
                "num_heads": number of attention heads
                "filter_size": num_units for feed-forward networks
                "dropout":
                "causal": False
            }
            scope:
            reuse:
        return:
            Tensor([B, T, hidden_size], tf.float32)
            Tensor([B, T, hidden_size, tf.float32)
        """
        hidden_size = int(hparams["hidden_size"])
        num_heads = int(hparams["num_heads"])
        causal = "causal" in hparams and hparams["causal"]
        keep_prob = 1.0 - float(hparams["dropout"])
        filter_size = int(hparams["filter_size"])
        scope = scope or "attention_block"
        with tf.variable_scope(scope, reuse=reuse):
            # norm
            atten_in = layer_norm(inputs, scope="norm_0")
            # multi-head self-attention
            atten_out, attention_matrix = self.multihead_attention(
                atten_in, atten_in, atten_in, paddings,
                hidden_size=hidden_size,
                num_heads=num_heads,
                causal=False,
                scope="attention")
            self.plot_attention_matrix("scope/attentions", attention_matrix, num_heads)
            '''
            # extra the loss for keep head to learn different
            attention_matrix = tf.split(attention_matrix, num_heads, axis=0)
            constrain_matrix = tf.multiply(attention_matrix[0], attention_matrix[1])
            for i in range(2, len(attention_matrix)):
                constrain_matrix = tf.multiply(attention_matrix[i], constrain_matrix)
            self._att_constrain.append(constrain_matrix)
            '''
            out1 = tf.nn.dropout(atten_out, keep_prob)
            # add
            out1 += inputs

            # norm
            fnn_in = layer_norm(out1, scope="norm_1")
            # feed-forward networks
            ffn_out = tf.layers.dense(fnn_in, filter_size, tf.nn.relu, name="fnn_0")
            ffn_out = tf.layers.dense(ffn_out, hidden_size, name="fnn_1")
            outputs = tf.nn.dropout(ffn_out, keep_prob)
            # add
            outputs += out1
            # mask the padding element
            _mask = tf.tile(tf.expand_dims(tf.to_float(paddings), axis=-1),
                            [1, 1, hidden_size])
            outputs *= _mask
        return outputs, atten_out

    def decoder_block(self, decoder_in, encoder_out, hparams,
                      dec_paddings, enc_paddings,
                      is_training=True, scope=None, reuse=None):
        """ decoder block, including:
            1. multihead self-attention for targets
            2. multihead encoder-decoder attention
            3. feed-forward networks
        add & norm for each modules according the official implementation
        args:
            decoder_in: Tensor([B, T, D], tf.float32)
            encoder_out: Tensor([B, T, d_enc], tf.float32)
            hparams: DICT, {
                "hidden_size":
                "num_heads":
                "filter_size":
                "causal": BOOL
                "dropout":
            }
            scope:
        return:
        """
        hidden_size = int(hparams["hidden_size"])
        num_heads = int(hparams["num_heads"])
        causal = "causal" in hparams and hparams["causal"]
        keep_prob = 1.0 - float(hparams["dropout"])
        filter_size = int(hparams["filter_size"])

        scope = scope or "decoder_block"
        with tf.variable_scope(scope, reuse=reuse):
            # norm
            atten_in = layer_norm(decoder_in, scope="norm_0")
            # multiheaded self-attention for decoder
            atten_out1, _ = self.multihead_attention(
                atten_in, atten_in, atten_in, dec_paddings,
                hidden_size=hidden_size,
                num_heads=num_heads,
                causal=True,
                scope="attention_0")
            out1 = tf.nn.dropout(atten_out1, keep_prob)
            # add
            out1 += decoder_in

            # norm
            atten_in = layer_norm(out1, scope="norm_1")
            # encoder-decoder attention
            atten_out2, _ = self.multihead_attention(
                atten_in, encoder_out, encoder_out, enc_paddings,
                hidden_size=hidden_size,
                num_heads=num_heads,
                causal=False,
                scope="attention_1")
            out2 = tf.nn.dropout(atten_out2, keep_prob)
            # add
            out2 += out1

            # norm
            ffn_in = layer_norm(out2, scope="norm_2")
            # feed-forward networks
            ffn_out = tf.layers.dense(ffn_in, filter_size, tf.nn.relu, name="fnn_0")
            ffn_out = tf.layers.dense(ffn_out, hidden_size, name="fnn_1")
            outputs = tf.nn.dropout(ffn_out, keep_prob)
            # add
            outputs += out2
            # blind the padding location
            _mask = tf.tile(tf.expand_dims(tf.to_float(dec_paddings), axis=-1),
                            [1, 1, hidden_size])
            outputs *= _mask
        return outputs, atten_out1, atten_out2

    def multihead_attention(self, query, key, value,
                            key_padding,
                            hidden_size=None,
                            num_heads=8,
                            causal=False,
                            scope=None,
                            reuse=None):
        """ multihead attention, support causal for decoder_self_attention
        args:
            query:  Tensor([B, T_q, D_q], tf.float32)
            key:    Tensor([B, T_k, D_q], tf.float32)
            value:  Tensor([B, T_v, D_v], tf.float32), T_v = T_k
            hidden_size: INT, num_units for projection of query, key, value
            num_heads: INT, num heads for attention
            causal: True, mask out (setting to a small value) the values of Q*K^T;
                    False, default
            scope:
            reuse: False(default)
        return:
            attention_value: Tensor([B, T_q, num_units], tf.float32),
            attention_matrix:Tensor(B, T_q, T_k), tf.float32)
        """
        scope = scope or "multihead_attention"
        with tf.variable_scope(scope, reuse=reuse):
            if hidden_size is None:
                hidden_size = query.get_shape().as_list[-1]
            depth = hidden_size // num_heads
            # multihead parallel computing
            Q = tf.layers.dense(query, hidden_size, use_bias=False, name="Q")  # (B, T_q, C)
            K = tf.layers.dense(key, hidden_size, use_bias=False, name="K")    # (B, T_k, C)
            V = tf.layers.dense(value, hidden_size, use_bias=False, name="V")  # (B, T_v, C)

            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*B, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*B, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*B, T_v, C/h)

            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))   # (h*B, T_q, T_k)
            # scale
            outputs *= depth ** -0.5

            # mask the padding part according key_mask
            _mask = tf.tile(tf.to_float(key_padding), [num_heads, 1])  # (h*B, T_k)
            # (h*B, T_q, T_k)
            _mask = tf.tile(tf.expand_dims(_mask, 1), [1, tf.shape(query)[1], 1])

            if causal:
                # print("[INFO] %s | causal=True" % scope)
                # mask the future location
                tril = tf.matrix_band_part(tf.ones_like(outputs), -1, 0)
                _mask *= tril

            # a NEG_INF value(-2^32+1) at padding location
            padding_bias = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(_mask, 0), padding_bias, outputs)
            '''
            # TODO: optimize for blinding the future elements when decoding
            # mask to keep the causality, future blinding
            if causal:
                print("[INFO] Using the causal.....")
                diag_vals = tf.ones_like(outputs[0, :, :])
                tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
                # (h*N, T_q, T_k)
                masking = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])
                # a NEG_INF at future location
                masking_bias = tf.ones_like(masking) * (-2 ** 32 + 1)
                # (h*N, T_q, T_k)
                outputs = tf.where(tf.equal(masking, 0), masking_bias, outputs)
            '''
            attention_matrix = tf.nn.softmax(outputs)  # (h*B, T_q, T_k)

            # weighted sum, calculate the attention_value
            outputs = tf.matmul(attention_matrix, V_)  # (h*B, T_q, c/h), T_v = T_k
            # reshape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
            # output_transform
            # outputs = tf.layers.dense(outputs, hidden_size, use_bias=False,
            #                           name="out_transform")
        return outputs, attention_matrix

    @staticmethod
    def position_encoding_v1(batch_size, time_steps, dim,
                            scale=None,
                            scope="add_pos_encoding_v1"):
        """ gen position encoding for sequences
        args:
            batch_size: INT, batch size
            length: INT, sequence length
            dim: INT,
            scale: FLOAT,
            scope:
        return:
            Tensor([batch_size, length, dim], tf.float32)
        """
        with tf.name_scope(scope):
            position_ind = tf.tile(tf.expand_dims(tf.range(time_steps), 0),
                                   [batch_size, 1])

            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(10000, 2.0*i / dim) for i in range(dim)]
                for pos in range(time_steps)])

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

            # Convert to a tensor
            lookup_table = tf.convert_to_tensor(position_enc, tf.float32)

            outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

            if scale is None:
                outputs *= tf.sqrt(tf.to_float(dim))
            else:
                outputs *= float(scale)
        return tf.to_float(outputs)

    @staticmethod
    def position_encoding(batch_size, length, channels,
                          scale=False,
                          start_index=0,
                          min_timescale=1.0,
                          max_timescale=1.0e4,
                          scope="add_pos_encoding"):
        """ Return positional encoding for sequences
        Calculates the position encoding as a mix of sine and cosine functions with
        geometrically increasing wavelengths.
        Based on the implementation in
        https://github.com/tensorflow/models/tree/master/official/transformer/model
        Defined and formulized in Attention is All You Need, section 3.5.
        Args:
            batch_size: INT, batch size
            time_steps: INT, Sequence length.
            dim: INT, Size of the
            scale: BOOL, True, multiply results by sqrt(dim)
            start_index: INT, the start position, < length
            min_timescale: Minimum scale that will be applied at each position
            max_timescale: Maximum scale that will be applied at each position
            scope: STR, "position_encoding"
        Returns:
          Tensor([batch_size, time_steps, dim], tf.float)
        """
        with tf.name_scope(scope):
            position = tf.to_float(tf.range(length) + start_index)
            num_timescales = channels // 2
            log_timescale_increment = (
                np.log(float(max_timescale) / float(min_timescale)) /
                tf.maximum(tf.to_float(num_timescales) - 1, 1))
            inv_timescales = min_timescale * tf.exp(
                tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
            scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
            outputs = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
            outputs = tf.pad(outputs, [[0, 0], [0, tf.mod(channels, 2)]])
            if scale:
                outputs *= tf.sqrt(tf.to_float(channels))
            outputs = tf.tile(tf.expand_dims(outputs, axis=0), [batch_size, 1, 1])
        return outputs

    @staticmethod
    def position_encoding_v3(length, channels,
                          scale=False,
                          min_timescale=1.0,
                          max_timescale=1.0e4):
        """ Return positional encoding for sequences
        Calculates the position encoding as a mix of sine and cosine functions with
        geometrically increasing wavelengths.

        Args:
            batch_size: INT, batch size
            length: INT, Sequence length.
            channels: INT, Size of the
            scale: BOOL, True, multiply results by sqrt(dim)
            start_index: the start position, < length
            min_timescale: Minimum scale that will be applied at each position
            max_timescale: Maximum scale that will be applied at each position
            scope: STR, "position_encoding"
        Returns:
          array([batch_size, time_steps, dim], tf.float32)
        """
        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.0 * (i // 2) / channels)
             for i in range(channels)] for pos in range(length)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        if scale:
            position_enc *= channels ** 0.5
        return position_enc

    @staticmethod
    def embedding_lookup(inputs, name,
                         shape=None,
                         initializer=None,
                         scale=True,
                         trainble=True,
                         scope="embedding",
                         reuse=None):
        """ lookup the embedding of inputs,
        args:
            inputs: Tensor([B,..., 1], tf.int32)
            name: STR, the name of embedding_table
            shape: LIST, [size_vocab, embedding_dim]
            initializer: initializer methods or Tensor
            scale: True, embedded results * sqrt(embedding_dim)
            trainble: True, the learned embedding table
            scope: "embedding"
            reuse: BOOL or None
        return:
            Tensor([B,..., embedding_dim], tf.float32)
        """
        with tf.variable_scope(scope, reuse=reuse):
            if initializer is None:
                initializer = tf.random_normal_initializer(
                    0., shape[-1]**-0.5)
            lookup_table = tf.get_variable(name, shape=shape,
                                           dtype=tf.float32,
                                           initializer=initializer,
                                           trainable=trainble)
            tf.summary.histogram(name, lookup_table)
            # inputs = tf.convert_to_tensor(inputs, tf.int32)
            # outputs = tf.gather(lookup_table, inputs)
            outputs = tf.nn.embedding_lookup(lookup_table, inputs)
            if scale:
                outputs *= shape[-1] ** 0.5
        return outputs

    def _embedding_feature_lookup(self, features):
        """ translate feature ids to embedding vector
        first ids is wordvec, others should be one-hot
            word ids --> embedding --> word2vec_layers -|--concat -->
            onehot ids --------------> onehot_layers ---|
        zero-padding
        args:
            features: Tensor([B, T, num_features], tf.int32)
            wordvec_size: INT, the size of word_vocab
            wordvec_initializer: initializer method or Tensor
            trainble: True, the learned word_embedding
        return:
            outputs: embedding features, Tensor([B, T, feature_dim], tf.float32)
        """
        # vec_fn = os.path.abspath("../../dict/word2vec_decompress.feat")
        # word2vec = TextProcessBox.read_wordvec(vec_fn, is_compress=False)
        # get padding mask of feature ids, [B, T]
        padding_mask = tf.not_equal(tf.reduce_max(features, axis=-1), PAD_ID)
        embedding_inputs = list()
        wordvec_dim = self._feature_dim_list[0]
        if self._wordvec is None:
            _initializer = tf.random_normal_initializer(
                0, stddev=wordvec_dim**-0.5)
            word_table = tf.get_variable(
                "word_embedding", shape=(self._vocab_size, wordvec_dim),
                initializer=_initializer, trainable=True)
        else:
            word_table = tf.get_variable(
                "word_embedding", initializer=self._wordvec, trainable=False)

        word_embedding = tf.nn.embedding_lookup(word_table, features[..., 0])

        try:
            word2vec_layers = self._cfg["word2vec_layer"]
        except KeyError:
            word2vec_layers = []
        layer_out = word_embedding
        for i, params in enumerate(word2vec_layers):
            layer_type = params["layer_type"].lower()
            scope = "word2vec_layer_%d_%s" % (i, layer_type)
            layer_out = tf_layers.apply_layer(
                layer_out,
                params,
                dropout=0.,
                is_training=self._is_training,
                scope=scope
            )
            print("[DEBUG] %s\n\tparams:%s" % (scope, str(params)))

        embedding_inputs.append(layer_out)

        onehot_inputs = list()
        for i in range(1, len(self._feature_dim_list)):
            if self._feature_dim_list[i] == 1:
                onehot_inputs.append(tf.expand_dims(features[..., i], -1))
            else:
                onehot_inputs.append(
                    tf.one_hot(tf.cast(features[..., i], tf.int32),
                               depth=self._feature_dim_list[i],
                               on_value=1., off_value=0., dtype=tf.float32))
        onehot_inputs = tf.concat(onehot_inputs, axis=2)

        try:
            onehot_layers = self._cfg["onehot_layer"]
        except KeyError:
            onehot_layers = []
        layer_out = onehot_inputs
        for i, params in enumerate(onehot_layers):
            layer_type = params["layer_type"].lower()
            scope = "onehot_layer_%d_%s" % (i, layer_type)
            layer_out = tf_layers.apply_layer(
                layer_out,
                params,
                dropout=0.,
                is_training=self._is_training,
                scope=scope
            )
            print("[DEBUG] %s\n\tparams:%s" % (scope, str(params)))
        embedding_inputs.append(layer_out)
        outputs = tf.concat(embedding_inputs, axis=2)
        # inputs = tf.nn.dropout(inputs, keep_prob=1.0-self._dropout)
        # zero-padding mask
        _mask = tf.expand_dims(tf.to_float(padding_mask), axis=-1)
        # print(_mask)
        outputs = outputs * _mask
        return outputs

    @staticmethod
    def plot_attention_matrix(name, attention_matrix, num_heads):
        batch_size, time_steps, _ = tf.unstack(tf.shape(attention_matrix))
        # draw attention
        atten_images = tf.expand_dims(
            tf.reshape(attention_matrix, [batch_size, -1, time_steps]), -1)
        for i in range(num_heads):
            tf.summary.image("%s_%d" % (name, i),
                             atten_images[:, time_steps * i: time_steps * (i + 1), :],
                             max_outputs=1)

