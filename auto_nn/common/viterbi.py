"""
An implementation of Viterbi decoding as a TensorFlow graph.

This can be used for CRF decoding without relying on NumPy.
"""
import tensorflow as tf


def batch_gather_3d(values, indices):
    """Do a batched gather on a 3D tensor."""
    return tf.gather(tf.reshape(values, [-1, tf.shape(values)[2]]),
                     tf.range(0, tf.shape(values)[0]) * tf.shape(values)[1] +
                     indices)


def batch_gather_2d(values, indices):
    """Do a batched gather on a 2D tensor."""
    return tf.gather(tf.reshape(values, [-1]),
                     tf.range(0, tf.shape(values)[0]) * tf.shape(values)[1] +
                     indices)


def viterbi_decode(score, transition_params, sequence_lengths, scope=None):
    """
    Decode the highest scoring sequence of tags as a compute graph.
    Args:
        score: A [batch, seq_len, num_tags] matrix of unary potentials.
        transition_params: A [num_tags, num_tags] matrix of binary potentials.
        sequence_lengths: A [batch] int32 vector of the length of each score
            sequence.
        scope: A scope for the decoding operations.

    Returns a tuple (viterbi, viterbi_score):
        viterbi: A [batch, seq_len] list of integers containing the highest
            scoring tag indices.
        viterbi_score: A vector of float containing the score for the Viterbi
            sequence.
    """
    with tf.name_scope(scope or "ViterbiDecode"):
        sequence_lengths = tf.cast(tf.convert_to_tensor(
            sequence_lengths, name="SequenceLengths"), tf.int32)
        score = tf.convert_to_tensor(score, name="Score")
        transition_params = tf.convert_to_tensor(
            transition_params, name="TransitionParams")

        def condition(timestep, *_):
            """Stop when full score sequence has been read in."""
            return tf.less(timestep, tf.shape(score)[1])

        def body_fwd(timestep, trellis, backpointers, trellis_val):
            """Perform forward viterbi pass."""
            v = (tf.expand_dims(trellis_val, 2)
                 + tf.expand_dims(transition_params, 0))
            new_trellis_val = score[:, timestep, :] + tf.reduce_max(v, axis=1)
            new_trellis = trellis.write(timestep, new_trellis_val)
            new_backpointers = backpointers.write(
                timestep, tf.cast(tf.argmax(v, axis=1), tf.int32))
            return timestep + 1, new_trellis, new_backpointers, new_trellis_val

        trellis_arr = tf.TensorArray(score.dtype, size=0, dynamic_size=True,
                                     clear_after_read=False, infer_shape=False)
        first_trellis_val = score[:, 0, :]
        trellis_arr = trellis_arr.write(0, first_trellis_val)

        backpointers_arr = tf.TensorArray(tf.int32, size=0, dynamic_size=True,
                                          clear_after_read=False,
                                          infer_shape=False)
        backpointers_arr = backpointers_arr.write(
            0, tf.zeros_like(score[:, 0, :], dtype=tf.int32))

        start_index = tf.constant(1, name="StartIndex", dtype=tf.int32)
        _, trellis_out, backpointers_out, _ = tf.while_loop(
            condition, body_fwd,
            (start_index, trellis_arr, backpointers_arr, first_trellis_val),
            parallel_iterations=1, back_prop=False)

        # Flip from time-major to batch major representation.
        trellis_out = tf.transpose(trellis_out.stack(), [1, 0, 2])
        backpointers_out = tf.transpose(backpointers_out.stack(), [1, 0, 2])

        def body_bwd(timestep, viterbi, last_decision):
            """Loop body for the backward pass."""
            backpointers_timestep = batch_gather_3d(
                backpointers_out, tf.maximum(sequence_lengths - timestep, 0))
            new_last_decision = batch_gather_2d(
                backpointers_timestep, last_decision)
            new_viterbi = viterbi.write(timestep, new_last_decision)
            return timestep + 1, new_viterbi, new_last_decision

        last_timestep = batch_gather_3d(trellis_out, sequence_lengths - 1)
        # get scores for last timestep of each batch element inside
        # trellis:
        scores = tf.reduce_max(last_timestep, axis=1)
        # get choice index for last timestep:
        last_decision = tf.cast(tf.argmax(last_timestep, axis=1), tf.int32)

        # Decode backwards using backpointers.
        viterbi = tf.TensorArray(tf.int32, size=0, dynamic_size=True,
                                 clear_after_read=False, infer_shape=False)
        viterbi = viterbi.write(0, last_decision)
        _, viterbi_out, _ = tf.while_loop(
            condition, body_bwd,
            (start_index, viterbi, last_decision),
            parallel_iterations=1, back_prop=False)
        viterbi_out = viterbi_out

        # Make outputs batch major.
        viterbi_out = tf.transpose(viterbi_out.stack(), [1, 0])
        viterbi_out_fwd = tf.reverse_sequence(
            viterbi_out, sequence_lengths, seq_dim=1)
        return viterbi_out_fwd, scores
