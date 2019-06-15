import io
import logging
import os
import pickle
import typing
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Text, Tuple
from random import shuffle

import numpy as np
from scipy.sparse import issparse, csr_matrix
from tensor2tensor.models.transformer import transformer_small, transformer_prepare_encoder, transformer_encoder
from tensor2tensor.layers.common_attention import add_timing_signal_1d, large_compatible_negative

from rasa.nlu.classifiers import INTENT_RANKING_LENGTH, NUM_INTENT_CANDIDATES
from rasa.nlu.components import Component
from rasa.utils.common import is_logging_disabled

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import tensorflow as tf
    from rasa.nlu.config import RasaNLUModelConfig
    from rasa.nlu.training_data import TrainingData
    from rasa.nlu.model import Metadata
    from rasa.nlu.training_data import Message

try:
    import tensorflow as tf
except ImportError:
    tf = None


class EmbeddingIntentClassifier(Component):
    """Intent classifier using supervised embeddings.

    The embedding intent classifier embeds user inputs
    and intent labels into the same space.
    Supervised embeddings are trained by maximizing similarity between them.
    It also provides rankings of the labels that did not "win".

    The embedding intent classifier needs to be preceded by
    a featurizer in the pipeline.
    This featurizer creates the features used for the embeddings.
    It is recommended to use ``CountVectorsFeaturizer`` that
    can be optionally preceded by ``SpacyNLP`` and ``SpacyTokenizer``.

    Based on the starspace idea from: https://arxiv.org/abs/1709.03856.
    However, in this implementation the `mu` parameter is treated differently
    and additional hidden layers are added together with dropout.
    """

    provides = ["intent", "intent_ranking"]

    requires = ["text_features"]

    defaults = {
        # nn architecture
        # sizes of hidden layers before the embedding layer for input words
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_a": [256, 128],
        # sizes of hidden layers before the embedding layer for intent labels
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_b": [],

        "share_embedding": False,
        "bidirectional": False,
        "fused_lstm": False,
        "gpu_lstm": False,
        "transformer": False,
        "pos_encoding": "timing",  # {"timing", "emb", "custom_timing"}
        # introduce phase shift in time encodings between transformers
        # 0.5 - 0.8 works on small dataset
        "pos_max_timescale": 1.0e2,
        "max_seq_length": 256,
        "num_heads": 4,
        "use_last": False,

        # training parameters
        "layer_norm": True,
        # initial and final batch sizes - batch size will be
        # linearly increased for each epoch
        "batch_size": [64, 256],
        # number of epochs
        "epochs": 300,

        # embedding parameters
        # dimension size of embedding vectors
        "embed_dim": 20,
        # how similar the algorithm should try
        # to make embedding vectors for correct intent labels
        "mu_pos": 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect intent labels
        "mu_neg": -0.4,  # should be -1.0 < ... < 1.0 for 'cosine'
        # the type of the similarity
        "similarity_type": 'cosine',  # string 'cosine' or 'inner'
        "loss_type": 'margin',  # string 'softmax' or 'margin'
        # the number of incorrect intents, the algorithm will minimize
        # their similarity to the input words during training
        "num_neg": 20,
        "iou_threshold": 1.0,
        # flag: if true, only minimize the maximum similarity for
        # incorrect intent labels
        "use_max_sim_neg": True,
        # set random seed to any int to get reproducible results
        # try to change to another int if you are not getting good results
        "random_seed": None,

        # regularization parameters
        # the scale of L2 regularization
        "C2": 0.002,
        # the scale of how critical the algorithm should be of minimizing the
        # maximum similarity between embeddings of different intent labels
        "C_emb": 0.8,
        # dropout rate for rnn
        "droprate": 0.2,

        # flag: if true, the algorithm will split the intent labels into tokens
        #       and use bag-of-words representations for them
        "intent_tokenization_flag": False,
        # delimiter string to split the intent labels
        "intent_split_symbol": '_',

        # visualization of accuracy
        # how often to calculate training accuracy
        "evaluate_every_num_epochs": 10,  # small values may hurt performance
        # how many examples to use for calculation of training accuracy
        "evaluate_on_num_examples": 1000  # large values may hurt performance
    }

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 inv_intent_dict: Optional[Dict[int, Text]] = None,
                 encoded_all_intents: Optional[np.ndarray] = None,
                 all_intents_embed_values: Optional[np.ndarray] = None,
                 session: Optional['tf.Session'] = None,
                 graph: Optional['tf.Graph'] = None,
                 message_placeholder: Optional['tf.Tensor'] = None,
                 intent_placeholder: Optional['tf.Tensor'] = None,
                 similarity_op: Optional['tf.Tensor'] = None,
                 all_intents_embed_in: Optional['tf.Tensor'] = None,
                 sim_all: Optional['tf.Tensor'] = None,
                 word_embed: Optional['tf.Tensor'] = None,
                 intent_embed: Optional['tf.Tensor'] = None,
                 ) -> None:
        """Declare instant variables with default values"""

        self._check_tensorflow()
        super(EmbeddingIntentClassifier, self).__init__(component_config)

        self._load_params()

        # transform numbers to intents
        self.inv_intent_dict = inv_intent_dict

        # encode all intents with numbers
        self.encoded_all_intents = encoded_all_intents
        self.all_intents_embed_values = all_intents_embed_values
        self.iou = None

        # tf related instances
        self.session = session
        self.graph = graph
        self.a_in = message_placeholder
        self.b_in = intent_placeholder
        self.sim_op = similarity_op

        self.all_intents_embed_in = all_intents_embed_in
        self.sim_all = sim_all

        self.sequence = len(self.a_in.shape) == 3 if self.a_in is not None else None

        # persisted embeddings
        self.word_embed = word_embed
        self.intent_embed = intent_embed

        # Flags to ensure test data is featurized only once
        self.is_test_data_featurized = False


    # init helpers
    def _load_nn_architecture_params(self, config: Dict[Text, Any]) -> None:
        self.hidden_layer_sizes = {'a': config['hidden_layers_sizes_a'],
                                   'b': config['hidden_layers_sizes_b']}

        self.share_embedding = config['share_embedding']
        if self.share_embedding:
            if self.hidden_layer_sizes['a'] != self.hidden_layer_sizes['b']:
                raise ValueError("If embeddings are shared "
                                 "hidden_layer_sizes must coincide")

        self.bidirectional = config['bidirectional']
        self.fused_lstm = config['fused_lstm']
        self.gpu_lstm = config['gpu_lstm']
        self.transformer = config['transformer']
        if (self.gpu_lstm and self.fused_lstm) or (self.transformer and self.fused_lstm) or (self.gpu_lstm and self.transformer):
            raise ValueError("Either `gpu_lstm` or `fused_lstm` or `transformer` should be specified")
        if self.gpu_lstm or self.transformer:
            if any(self.hidden_layer_sizes['a'][0] != size
                   for size in self.hidden_layer_sizes['a']):
                raise ValueError("GPU training only supports identical sizes among layers a")
            if any(self.hidden_layer_sizes['b'][0] != size
                   for size in self.hidden_layer_sizes['b']):
                raise ValueError("GPU training only supports identical sizes among layers b")

        self.pos_encoding = config['pos_encoding']
        self.pos_max_timescale = config['pos_max_timescale']
        self.max_seq_length = config['max_seq_length']
        self.num_heads = config['num_heads']
        self.use_last = config['use_last']

        self.batch_size = config['batch_size']
        self.epochs = config['epochs']

    def _load_embedding_params(self, config: Dict[Text, Any]) -> None:
        self.layer_norm = config['layer_norm']
        self.embed_dim = config['embed_dim']
        self.mu_pos = config['mu_pos']
        self.mu_neg = config['mu_neg']
        self.similarity_type = config['similarity_type']
        self.loss_type = config['loss_type']
        self.num_neg = config['num_neg']
        self.iou_threshold = config['iou_threshold']
        self.use_max_sim_neg = config['use_max_sim_neg']
        self.random_seed = self.component_config['random_seed']

    def _load_regularization_params(self, config: Dict[Text, Any]) -> None:
        self.C2 = config['C2']
        self.C_emb = config['C_emb']
        self.droprate = config['droprate']

    def _load_flag_if_tokenize_intents(self, config: Dict[Text, Any]) -> None:
        self.intent_tokenization_flag = config['intent_tokenization_flag']
        self.intent_split_symbol = config['intent_split_symbol']
        if self.intent_tokenization_flag and not self.intent_split_symbol:
            logger.warning("intent_split_symbol was not specified, "
                           "so intent tokenization will be ignored")
            self.intent_tokenization_flag = False

    def _load_visual_params(self, config: Dict[Text, Any]) -> None:
        self.evaluate_every_num_epochs = config['evaluate_every_num_epochs']
        if self.evaluate_every_num_epochs < 1:
            self.evaluate_every_num_epochs = self.epochs

        self.evaluate_on_num_examples = config['evaluate_on_num_examples']

    def _load_params(self) -> None:

        self._load_nn_architecture_params(self.component_config)
        self._load_embedding_params(self.component_config)
        self._load_regularization_params(self.component_config)
        self._load_flag_if_tokenize_intents(self.component_config)
        self._load_visual_params(self.component_config)

    # package safety checks
    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["tensorflow"]

    @staticmethod
    def _check_tensorflow():
        if tf is None:
            raise ImportError(
                'Failed to import `tensorflow`. '
                'Please install `tensorflow`. '
                'For example with `pip install tensorflow`.')

    # training data helpers:
    @staticmethod
    def _create_intent_dict(training_data: 'TrainingData') -> Dict[Text, int]:
        """Create intent dictionary"""

        distinct_intents = set([example.get("intent")
                                for example in training_data.intent_examples])
        return {intent: idx
                for idx, intent in enumerate(sorted(distinct_intents))}

    @staticmethod
    def _find_example_for_intent(intent, examples):
        for ex in examples:
            if ex.get("intent") == intent:
                return ex

    # @staticmethod
    def _create_encoded_intents(self,
                                intent_dict: Dict[Text, int],
                                training_data: 'TrainingData') -> np.ndarray:
        """Create matrix with intents encoded in rows as bag of words.

        If intent_tokenization_flag is off, returns identity matrix.
        """

        if self.intent_tokenization_flag:
            encoded_all_intents = []

            for key, idx in intent_dict.items():
                encoded_all_intents.insert(
                    idx,
                    self._find_example_for_intent(
                        key,
                        training_data.intent_examples
                    ).get("intent_features")
                )

            return np.array(encoded_all_intents)
        else:
            return np.eye(len(intent_dict))

    # noinspection PyPep8Naming
    def _prepare_data_for_training(
        self,
        training_data: 'TrainingData',
        intent_dict: Dict[Text, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""

        X = np.stack([e.get("text_features")
                      for e in training_data.intent_examples])

        intents_for_X = np.array([intent_dict[e.get("intent")]
                                  for e in training_data.intent_examples])

        if self.intent_tokenization_flag:
            Y = np.stack([e.get("intent_features")
                          for e in training_data.intent_examples])
        else:
            Y = np.stack([self.encoded_all_intents[intent_idx]
                          for intent_idx in intents_for_X])

        return X, Y, intents_for_X

    # tf helpers:
    def _create_tf_embed_nn(self, x_in: 'tf.Tensor', is_training: 'tf.Tensor',
                            layer_sizes: List[int], name: Text) -> 'tf.Tensor':
        """Create nn with hidden layers and name"""

        reg = tf.contrib.layers.l2_regularizer(self.C2)
        x = x_in
        for i, layer_size in enumerate(layer_sizes):
            x = tf.layers.dense(inputs=x,
                                units=layer_size,
                                activation=tf.nn.relu,
                                kernel_regularizer=reg,
                                name='hidden_layer_{}_{}'.format(name, i),
                                reuse=tf.AUTO_REUSE)
            x = tf.layers.dropout(x, rate=self.droprate, training=is_training)

        return x

    def _create_rnn_cell(self,
                         is_training: 'tf.Tensor',
                         rnn_size: int,
                         real_length) -> 'tf.contrib.rnn.RNNCell':
        """Create one rnn cell."""

        # chrono initialization for forget bias
        # assuming that characteristic time is max dialogue length
        # left border that initializes forget gate close to 0
        bias_0 = -1.0
        characteristic_time = tf.reduce_mean(tf.cast(real_length, tf.float32))
        # right border that initializes forget gate close to 1
        bias_1 = tf.log(characteristic_time - 1.)
        fbias = (bias_1 - bias_0) * np.random.random(rnn_size) + bias_0

        keep_prob = 1.0 - (self.droprate *
                           tf.cast(is_training, tf.float32))

        return ChronoBiasLayerNormBasicLSTMCell(
            num_units=rnn_size,
            layer_norm=self.layer_norm,
            forget_bias=fbias,
            input_bias=-fbias,
            dropout_keep_prob=keep_prob,
            reuse=tf.AUTO_REUSE
        )

    def _create_tf_rnn_embed(self, x_in: 'tf.Tensor', is_training: 'tf.Tensor',
                             layer_sizes: List[int], name: Text) -> 'tf.Tensor':
        """Create rnn for dialogue level embedding."""

        reg = tf.contrib.layers.l2_regularizer(self.C2)
        # mask different length sequences
        mask = tf.sign(tf.reduce_max(x_in, -1))
        last = mask * tf.cumprod(1 - mask, axis=1, exclusive=True, reverse=True)
        mask = tf.cumsum(last, axis=1, reverse=True)
        real_length = tf.cast(tf.reduce_sum(mask, 1), tf.int32)

        last = tf.expand_dims(last, -1)

        x = tf.nn.relu(x_in)

        if len(layer_sizes) == 0:
            # return simple bag of words
            return tf.reduce_sum(x, 1)

        if self.fused_lstm:
            x = tf.transpose(x, [1, 0, 2])

            for i, layer_size in enumerate(layer_sizes):
                if self.bidirectional:
                    cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(layer_size,
                                                                reuse=tf.AUTO_REUSE,
                                                                name='rnn_fw_encoder_{}_{}'.format(name, i))
                    x_fw, _ = cell_fw(x, dtype=tf.float32, sequence_length=real_length)

                    cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(layer_size,
                                                                reuse=tf.AUTO_REUSE,
                                                                name='rnn_bw_encoder_{}_{}'.format(name, i))
                    cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(cell_bw)
                    x_bw, _ = cell_bw(x, dtype=tf.float32, sequence_length=real_length)

                    x = tf.concat([x_fw, x_bw], -1)

                else:
                    cell = tf.contrib.rnn.LSTMBlockFusedCell(layer_size,
                                                             reuse=tf.AUTO_REUSE,
                                                             name='rnn_encoder_{}_{}'.format(name, i))
                    x, _ = cell(x, dtype=tf.float32, sequence_length=real_length)

            x = tf.transpose(x, [1, 0, 2])
            x = tf.reduce_sum(x * last, 1)

        elif self.gpu_lstm:
            # only trains and predicts on gpu_lstm
            x = tf.transpose(x, [1, 0, 2])

            if self.bidirectional:
                direction = 'bidirectional'
            else:
                direction = 'unidirectional'

            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(len(layer_sizes),
                                                  layer_sizes[0],
                                                  direction=direction,
                                                  name='rnn_encoder_{}'.format(name))

            x, _ = lstm(x, training=True)
            # prediction graph is created separately

            x = tf.transpose(x, [1, 0, 2])
            x = tf.reduce_sum(x * last, 1)

        elif self.transformer:
            hparams = transformer_small()

            hparams.num_hidden_layers = len(layer_sizes)
            hparams.hidden_size = layer_sizes[0]
            # it seems to be factor of 4 for transformer architectures in t2t
            hparams.filter_size = layer_sizes[0] * 4
            hparams.num_heads = self.num_heads
            # hparams.relu_dropout = self.droprate
            hparams.pos = self.pos_encoding

            hparams.max_length = self.max_seq_length
            if not self.bidirectional:
                hparams.unidirectional_encoder = True

            # When not in training mode, set all forms of dropout to zero.
            for key, value in hparams.values().items():
                if key.endswith("dropout") or key == "label_smoothing":
                    setattr(hparams, key, value * tf.cast(is_training, tf.float32))

            x = tf.layers.dense(inputs=x,
                                units=hparams.hidden_size,
                                use_bias=False,
                                kernel_initializer=tf.random_normal_initializer(0.0, hparams.hidden_size**-0.5),
                                kernel_regularizer=reg,
                                name='transformer_embed_layer_{}'.format(name),
                                reuse=tf.AUTO_REUSE)
            x = tf.layers.dropout(x, rate=hparams.layer_prepostprocess_dropout, training=is_training)

            if hparams.multiply_embedding_mode == "sqrt_depth":
                x *= hparams.hidden_size**0.5

            x *= tf.expand_dims(mask, -1)

            with tf.variable_scope('transformer_{}'.format(name), reuse=tf.AUTO_REUSE):
                (x,
                 self_attention_bias,
                 encoder_decoder_attention_bias
                 ) = transformer_prepare_encoder(x, None, hparams)

                if hparams.pos == 'custom_timing':
                    x = add_timing_signal_1d(x, max_timescale=self.pos_max_timescale)

                x *= tf.expand_dims(mask, -1)

                x = tf.nn.dropout(x, 1.0 - hparams.layer_prepostprocess_dropout)

                attn_bias_for_padding = None
                # Otherwise the encoder will just use encoder_self_attention_bias.
                if hparams.unidirectional_encoder:
                    attn_bias_for_padding = encoder_decoder_attention_bias

                x = transformer_encoder(
                    x,
                    self_attention_bias,
                    hparams,
                    nonpadding=mask,
                    attn_bias_for_padding=attn_bias_for_padding)

            if self.use_last:
                x = tf.reduce_sum(x * last, 1)
            else:
                x *= tf.expand_dims(mask, -1)
                sum_mask = tf.reduce_sum(tf.expand_dims(mask, -1), 1)
                # fix for zero length sequences
                sum_mask = tf.where(sum_mask < 1, tf.ones_like(sum_mask), sum_mask)
                x = tf.reduce_sum(x, 1) / sum_mask

        else:
            for i, layer_size in enumerate(layer_sizes):
                if self.bidirectional:
                    cell_fw = self._create_rnn_cell(is_training, layer_size, real_length)
                    cell_bw = self._create_rnn_cell(is_training, layer_size, real_length)

                    x, _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, x,
                        dtype=tf.float32,
                        sequence_length=real_length,
                        scope='rnn_encoder_{}_{}'.format(name, i)
                    )
                    x = tf.concat(x, 2)

                else:
                    cell = self._create_rnn_cell(is_training, layer_size, real_length)

                    x, _ = tf.nn.dynamic_rnn(
                        cell, x,
                        dtype=tf.float32,
                        sequence_length=real_length,
                        scope='rnn_encoder_{}_{}'.format(name, i)
                    )

            x = tf.reduce_sum(x * last, 1)

        return x

    def _create_tf_embed_a(self,
                           a_in: 'tf.Tensor',
                           is_training: 'tf.Tensor',
                           ) -> 'tf.Tensor':
        """Create tf graph for training"""

        if len(a_in.shape) == 2:
            a = self._create_tf_embed_nn(a_in, is_training,
                                         self.hidden_layer_sizes['a'],
                                         name='a_and_b' if self.share_embedding else 'a')
        else:
            a = self._create_tf_rnn_embed(a_in, is_training,
                                          self.hidden_layer_sizes['a'],
                                          name='a_and_b' if self.share_embedding else 'a')

        reg = tf.contrib.layers.l2_regularizer(self.C2)
        emb_a = tf.layers.dense(inputs=a,
                                units=self.embed_dim,
                                kernel_regularizer=reg,
                                name='embed_layer_{}'.format('a'),
                                reuse=tf.AUTO_REUSE)

        return emb_a

    def _create_tf_embed_b(self,
                           b_in: 'tf.Tensor',
                           is_training: 'tf.Tensor',
                           ) -> 'tf.Tensor':
        """Create tf graph for training"""

        if len(b_in.shape) == 2:
            b = self._create_tf_embed_nn(b_in, is_training,
                                         self.hidden_layer_sizes['b'],
                                         name='a_and_b' if self.share_embedding else 'b')

        else:
            b = self._create_tf_rnn_embed(b_in, is_training,
                                          self.hidden_layer_sizes['b'],
                                          name='a_and_b' if self.share_embedding else 'b')

        reg = tf.contrib.layers.l2_regularizer(self.C2)
        emb_b = tf.layers.dense(inputs=b,
                                units=self.embed_dim,
                                kernel_regularizer=reg,
                                name='embed_layer_{}'.format('b'),
                                reuse=tf.AUTO_REUSE)
        return emb_b

    @staticmethod
    def _tf_sample_neg(b,
                       is_training: 'tf.Tensor',
                       neg_ids=None,
                       batch_size=None,
                       first_only=False
                       ) -> 'tf.Tensor':

        all_b = b[tf.newaxis, :, :]
        if batch_size is None:
            batch_size = tf.shape(b)[0]
        all_b = tf.tile(all_b, [batch_size, 1, 1])
        if neg_ids is None:
            return all_b

        def sample_neg_b():
            neg_b = tf.batch_gather(all_b, neg_ids)
            return tf.concat([b[:, tf.newaxis, :], neg_b], 1)

        if first_only:
            out_b = b[:, tf.newaxis, :]
        else:
            out_b = all_b

        return tf.cond(tf.logical_and(is_training, tf.shape(neg_ids)[0] > 1), sample_neg_b, lambda: out_b)

    def _tf_calc_iou(self,
                     b_raw,
                     is_training: 'tf.Tensor',
                     neg_ids,
                     ) -> 'tf.Tensor':

        if len(b_raw.shape) == 3:
            b_raw = tf.reduce_sum(b_raw, 1)
        tiled_intent_raw = self._tf_sample_neg(b_raw, is_training, neg_ids)
        pos_b_raw = tiled_intent_raw[:, :1, :]
        neg_b_raw = tiled_intent_raw[:, 1:, :]
        intersection_b_raw = tf.minimum(neg_b_raw, pos_b_raw)
        union_b_raw = tf.maximum(neg_b_raw, pos_b_raw)

        return tf.reduce_sum(intersection_b_raw, -1) / tf.reduce_sum(union_b_raw, -1)

    def _tf_sim(self,
                a: 'tf.Tensor',
                b: 'tf.Tensor') -> Tuple['tf.Tensor', 'tf.Tensor', 'tf.Tensor']:
        """Define similarity

        in two cases:
            sim: between embedded words and embedded intent labels
            sim_emb: between individual embedded intent labels only
        """

        if self.similarity_type == 'cosine':
            # normalize embedding vectors for cosine similarity
            a = tf.nn.l2_normalize(a, -1)
            b = tf.nn.l2_normalize(b, -1)

        if len(a.shape) == 3:
            a_pos = a[:, :1, :]
        else:
            a_pos = tf.expand_dims(a, 1)

        if self.similarity_type in {'cosine', 'inner'}:
            sim = tf.reduce_sum(a_pos * b, -1)
            sim_intent_emb = tf.reduce_sum(b[:, :1, :] * b[:, 1:, :], -1)
            if len(a.shape) == 3:
                sim_input_emb = tf.reduce_sum(a[:, :1, :] * a[:, 1:, :], -1)
            else:
                sim_input_emb = None

            return sim, sim_intent_emb, sim_input_emb

        else:
            raise ValueError("Wrong similarity type {}, "
                             "should be 'cosine' or 'inner'"
                             "".format(self.similarity_type))

    def _tf_loss_margin(self, sim: 'tf.Tensor', sim_intent_emb: 'tf.Tensor', sim_input_emb: 'tf.Tensor', bad_negs) -> 'tf.Tensor':
        """Define loss"""

        # loss for maximizing similarity with correct action
        loss = tf.maximum(0., self.mu_pos - sim[:, 0])

        sim_neg = sim[:, 1:] + large_compatible_negative(bad_negs.dtype) * bad_negs
        if self.use_max_sim_neg:
            # minimize only maximum similarity over incorrect actions
            max_sim_neg = tf.reduce_max(sim_neg, -1)
            loss += tf.maximum(0., self.mu_neg + max_sim_neg)
        else:
            # minimize all similarities with incorrect actions
            max_margin = tf.maximum(0., self.mu_neg + sim_neg)
            loss += tf.reduce_sum(max_margin, -1)

        # penalize max similarity between intent embeddings
        sim_intent_emb += large_compatible_negative(bad_negs.dtype) * bad_negs
        max_sim_intent_emb = tf.maximum(0., tf.reduce_max(sim_intent_emb, -1))
        loss += max_sim_intent_emb * self.C_emb

        # penalize max similarity between input embeddings

        sim_input_emb = tf.cond(tf.math.equal(tf.shape(sim_input_emb)[1], 0), lambda: sim_input_emb,
                                lambda: sim_input_emb + large_compatible_negative(bad_negs.dtype) * bad_negs)
        # sim_input_emb += large_compatible_negative(bad_negs.dtype) * bad_negs
        max_sim_input_emb = tf.maximum(0., tf.reduce_max(sim_input_emb, -1))
        loss += max_sim_input_emb * self.C_emb

        # average the loss over the batch and add regularization losses
        loss = tf.reduce_mean(loss) + tf.losses.get_regularization_loss()
        return loss

    def _tf_loss_softmax(self, sim: 'tf.Tensor', sim_intent_emb: 'tf.Tensor', sim_input_emb: 'tf.Tensor', bad_negs) -> 'tf.Tensor':
        """Define loss."""

        logits = tf.concat([sim[:, :1],
                            sim[:, 1:] + large_compatible_negative(bad_negs.dtype) * bad_negs,
                            sim_intent_emb + large_compatible_negative(bad_negs.dtype) * bad_negs,
                            sim_input_emb + large_compatible_negative(bad_negs.dtype) * bad_negs
                            ], -1)
        pos_labels = tf.ones_like(logits[:, :1])
        neg_labels = tf.zeros_like(logits[:, 1:])
        labels = tf.concat([pos_labels, neg_labels], -1)

        loss = tf.losses.softmax_cross_entropy(labels,
                                               logits)
        # add regularization losses
        loss += tf.losses.get_regularization_loss()
        return loss

    def tf_acc_op(self,sim_mat,is_training):

        max_sim_id = tf.reduce_max(sim_mat,axis=-1)
        acc_op = tf.cond(is_training, lambda: tf.reduce_mean(tf.cast(tf.math.equal(max_sim_id,tf.zeros_like(max_sim_id)),
                                                                     dtype=tf.float32)),
                         lambda: tf.reduce_mean(tf.cast(tf.math.equal(max_sim_id, tf.linalg.tensor_diag_part(sim_mat)),
                                                                     dtype=tf.float32)))

        return acc_op

    # training helpers:
    def _linearly_increasing_batch_size(self, epoch: int) -> int:
        """Linearly increase batch size with every epoch.

        The idea comes from https://arxiv.org/abs/1711.00489
        """

        if not isinstance(self.batch_size, list):
            return int(self.batch_size)

        if self.epochs > 1:
            batch_size = int(self.batch_size[0] +
                             epoch * (self.batch_size[1] -
                                      self.batch_size[0]) / (self.epochs - 1))

            return batch_size if batch_size % 2 == 0 else batch_size + 1

        else:
            return int(self.batch_size[0])

    @staticmethod
    def _to_sparse_tensor(array_of_sparse, auto2d=True):
        seq_len = max([x.shape[0] for x in array_of_sparse])
        coo = [x.tocoo() for x in array_of_sparse]
        data = [v for x in array_of_sparse for v in x.data]
        if seq_len == 1 and auto2d:
            indices = [ids for i, x in enumerate(coo) for ids in zip([i] * len(x.row), x.col)]
            return tf.SparseTensor(indices, data, (len(array_of_sparse), array_of_sparse[0].shape[-1]))
        else:
            indices = [ids for i, x in enumerate(coo) for ids in zip([i] * len(x.row), x.row, x.col)]
            return tf.SparseTensor(indices, data, (len(array_of_sparse), seq_len, array_of_sparse[0].shape[-1]))

    @staticmethod
    def _sparse_tensor_to_dense(sparse, units=None, shape=None):
        if shape is None:
            if len(sparse.shape) == 2:
                shape = (tf.shape(sparse)[0], units)
            else:
                shape = (tf.shape(sparse)[0], tf.shape(sparse)[1], units)

        return tf.cast(tf.reshape(tf.sparse_tensor_to_dense(sparse), shape), tf.float32)

    # noinspection PyPep8Naming
    def train(self,
              training_data: 'TrainingData',
              cfg: Optional['RasaNLUModelConfig'] = None,
              **kwargs: Any) -> None:
        """Train the embedding intent classifier on a data set."""

        tb_sum_dir = '/tmp/tb_logs'

        intent_dict = self._create_intent_dict(training_data)
        if len(intent_dict) < 2:
            logger.error("Can not train an intent classifier. "
                         "Need at least 2 different classes. "
                         "Skipping training of intent classifier.")
            return

        self.inv_intent_dict = {v: k for k, v in intent_dict.items()}
        self.encoded_all_intents = self._create_encoded_intents(
            intent_dict, training_data)

        X, Y, intents_for_X = self._prepare_data_for_training(
            training_data, intent_dict)

        if self.share_embedding:
            if X[0].shape[-1] != Y[0].shape[-1]:
                raise ValueError("If embeddings are shared "
                                 "text features and intent features "
                                 "must coincide")

        # check if number of negatives is less than number of intents
        logger.debug("Check if num_neg {} is smaller than "
                     "number of intents {}, "
                     "else set num_neg to the number of intents - 1"
                     "".format(self.num_neg,
                               self.encoded_all_intents.shape[0]))
        self.num_neg = min(self.num_neg,
                           self.encoded_all_intents.shape[0] - 1)

        self.graph = tf.Graph()
        with self.graph.as_default():
            # set random seed
            np.random.seed(self.random_seed)
            tf.set_random_seed(self.random_seed)

            X_tensor = self._to_sparse_tensor(X)
            Y_tensor = self._to_sparse_tensor(Y)

            batch_size_in = tf.placeholder(tf.int64)
            train_dataset = tf.data.Dataset.from_tensor_slices((X_tensor, Y_tensor))
            train_dataset = train_dataset.shuffle(buffer_size=len(X))
            train_dataset = train_dataset.batch(batch_size_in, drop_remainder=self.fused_lstm)

            # tf.summary.scalar("batch_size", batch_size_in)

            if self.evaluate_on_num_examples:
                ids = np.random.permutation(len(X))[:self.evaluate_on_num_examples]
                # ids = [0, 1, 2]
                # [print(self.inv_intent_dict[intent]) for intent in intents_for_X[ids]]
                # exit()
                X_tensor_val = self._to_sparse_tensor(X[ids])
                Y_tensor_val = self._to_sparse_tensor(Y[ids])

                val_dataset = tf.data.Dataset.from_tensor_slices((X_tensor_val, Y_tensor_val)).batch(self.evaluate_on_num_examples)
            else:
                val_dataset = None

            if len(train_dataset.output_shapes[0]) == 2:
                train_dataset_output_shapes_X = train_dataset.output_shapes[0]
            else:
                train_dataset_output_shapes_X = (None, None, train_dataset.output_shapes[0][-1])

            if len(train_dataset.output_shapes[1]) == 2:
                train_dataset_output_shapes_Y = train_dataset.output_shapes[1]
            else:
                train_dataset_output_shapes_Y = (None, None, train_dataset.output_shapes[1][-1])

            iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                       (train_dataset_output_shapes_X, train_dataset_output_shapes_Y),
                                                       output_classes=train_dataset.output_classes)

            # iterator = train_dataset.make_initializable_iterator()
            a_sparse, b_sparse = iterator.get_next()

            self.a_raw = self._sparse_tensor_to_dense(a_sparse, X[0].shape[-1])
            self.b_raw = self._sparse_tensor_to_dense(b_sparse, Y[0].shape[-1])

            is_training = tf.placeholder_with_default(False, shape=())

            self.word_embed = self._create_tf_embed_a(self.a_raw, is_training)
            self.intent_embed = self._create_tf_embed_b(self.b_raw, is_training)

            neg_ids = tf.random.categorical(tf.log(1. - tf.eye(tf.shape(self.b_raw)[0])), self.num_neg)

            iou_intent = self._tf_calc_iou(self.b_raw, is_training, neg_ids)
            self.bad_negs = 1. - tf.nn.relu(tf.sign(self.iou_threshold - iou_intent))

            self.tiled_word_embed = self._tf_sample_neg(self.word_embed, is_training, neg_ids, first_only=True)
            self.tiled_intent_embed = self._tf_sample_neg(self.intent_embed, is_training, neg_ids)

            self.sim_op, self.sim_intent_emb, self.sim_input_emb = self._tf_sim(self.tiled_word_embed, self.tiled_intent_embed)
            if self.loss_type == 'margin':
                loss = self._tf_loss_margin(self.sim_op, self.sim_intent_emb, self.sim_input_emb, self.bad_negs)
            elif self.loss_type == 'softmax':
                loss = self._tf_loss_softmax(self.sim_op, self.sim_intent_emb, self.sim_input_emb, self.bad_negs)
            else:
                raise

            self.acc_op = self.tf_acc_op(self.sim_op,is_training)

            tf.summary.scalar('loss',loss)
            tf.summary.scalar('accuracy',self.acc_op)

            train_op = tf.train.AdamOptimizer().minimize(loss)

            train_init_op = iterator.make_initializer(train_dataset)
            if self.evaluate_on_num_examples:
                val_init_op = iterator.make_initializer(val_dataset)
            else:
                val_init_op = None

            self.summary_merged_op = tf.summary.merge_all()

            # [print(v.name, v.shape) for v in tf.trainable_variables()]
            # exit()
            # train tensorflow graph
            self.session = tf.Session()

            # self._train_tf(X, Y, intents_for_X, negs_in,
            #                loss, is_training, train_op)
            self._train_tf_dataset(train_init_op, val_init_op, batch_size_in, loss, is_training, train_op, tb_sum_dir)

            self.all_intents_embed_values = self._create_all_intents_embed(self.encoded_all_intents, iterator)

            # prediction graph
            self.all_intents_embed_in = tf.placeholder(tf.float32, (None, None, self.embed_dim),
                                                       name='all_intents_embed')

            self.a_in = tf.placeholder(self.a_raw.dtype, self.a_raw.shape, name='a')
            self.b_in = tf.placeholder(self.b_raw.dtype, self.b_raw.shape, name='b')

            if not self.gpu_lstm:
                self.word_embed = self._create_tf_embed_a(self.a_in, is_training)
                self.intent_embed = self._create_tf_embed_b(self.b_in, is_training)

                tiled_intent_embed = self._tf_sample_neg(self.intent_embed, is_training, None, tf.shape(self.word_embed)[0])

                self.sim_op, _, _ = self._tf_sim(self.word_embed, tiled_intent_embed)

                self.sim_all, _, _ = self._tf_sim(self.word_embed, self.all_intents_embed_in)

    def _train_tf_dataset(self,
                          train_init_op,
                          val_init_op,
                          batch_size_in,
                          loss: 'tf.Tensor',
                          is_training: 'tf.Tensor',
                          train_op: 'tf.Tensor',
                          tb_sum_dir: Text,
                          ) -> None:
        """Train tf graph"""

        self.session.run(tf.global_variables_initializer())

        train_tb_writer = tf.summary.FileWriter(os.path.join(tb_sum_dir,'train'),
                                      self.session.graph)
        test_tb_writer = tf.summary.FileWriter(os.path.join(tb_sum_dir,'test'))

        if self.evaluate_on_num_examples:
            logger.info("Accuracy is updated every {} epochs"
                        "".format(self.evaluate_every_num_epochs))

        pbar = tqdm(range(self.epochs), desc="Epochs", disable=is_logging_disabled())
        train_acc = 0
        last_loss = 0
        iter_num = 0
        for ep in pbar:

            batch_size = self._linearly_increasing_batch_size(ep)

            self.session.run(train_init_op, feed_dict={batch_size_in: batch_size})

            ep_loss = 0
            batches_per_epoch = 0

            while True:
                try:

                    fetch_list = (self.a_raw, self.b_raw, self.intent_embed, self.word_embed,
                                                                        self.tiled_intent_embed, self.tiled_word_embed,
                                                                        self.sim_op, self.bad_negs, self.sim_input_emb,
                                                                        self.sim_intent_emb,
                                                                        train_op, loss,
                                                                        self.summary_merged_op)

                    return_vals = self.session.run(fetch_list, feed_dict={is_training: True})

                    batch_loss = return_vals[-2]
                    #
                    # for i in range(10):
                    #     print(return_vals[i].shape)
                    #
                    # print(return_vals[7][0])

                    train_tb_writer.add_summary(return_vals[-1], iter_num)

                except tf.errors.OutOfRangeError:
                    break

                batches_per_epoch += 1
                iter_num += 1
                ep_loss += batch_loss

            # logger.info('Epoch ended with {0} number of mini-batch iterations'.format(batches_per_epoch))

            ep_loss /= batches_per_epoch

            if self.evaluate_on_num_examples and val_init_op is not None:
                if (ep == 0 or
                        (ep + 1) % self.evaluate_every_num_epochs == 0 or
                        (ep + 1) == self.epochs):

                    self.session.run(val_init_op)

                    fetch_list = (self.a_raw, self.b_raw, self.intent_embed, self.word_embed,
                                                                        self.tiled_intent_embed, self.tiled_word_embed,
                                                                        self.sim_op, self.bad_negs,self.sim_input_emb,
                                                                        self.sim_intent_emb,
                                                                        loss, self.acc_op, self.summary_merged_op)

                    return_vals = self.session.run(fetch_list)

                    summary = return_vals[-1]
                    # print('Evaluation round')
                    # for i in range(12):
                    #     print(return_vals[i].shape)

                    train_acc = return_vals[-2]

                    last_loss = ep_loss
                    test_tb_writer.add_summary(summary, iter_num)

                pbar.set_postfix({
                    "loss": "{:.3f}".format(ep_loss),
                    "acc": "{:.3f}".format(train_acc)
                })
            else:
                pbar.set_postfix({
                    "loss": "{:.3f}".format(ep_loss)
                })

        if self.evaluate_on_num_examples:
            logger.info("Finished training embedding classifier, "
                        "loss={:.3f}, train accuracy={:.3f}"
                        "".format(last_loss, train_acc))

    def _output_training_stat_dataset(self, val_init_op) -> np.ndarray:
        """Output training statistics"""

        self.session.run(val_init_op)
        train_sim = self.session.run(self.sim_op)

        sim_maxs = np.max(train_sim, -1)
        sim_diags = train_sim.diagonal()
        train_acc = np.mean(sim_maxs == sim_diags)
        return train_acc

    def _create_all_intents_embed(self, encoded_all_intents, iterator=None):

        all_intents_embed = []
        batch_size = self._linearly_increasing_batch_size(0)

        if iterator is None:
            batches_per_epoch = (len(encoded_all_intents) // batch_size +
                                 int(len(encoded_all_intents) % batch_size > 0))

            for i in range(batches_per_epoch):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size

                # batch_b = self._to_sparse_tensor(encoded_all_intents[start_idx:end_idx])
                batch_b = self._toarray(encoded_all_intents[start_idx:end_idx])

                all_intents_embed.append(self.session.run(self.intent_embed, feed_dict={self.b_in: batch_b}))
        else:
            if len(iterator.output_shapes[0]) == 2:
                shape_X = (len(encoded_all_intents), iterator.output_shapes[0][-1])
            else:
                shape_X = (len(encoded_all_intents), 1, iterator.output_shapes[0][-1])

            X_tensor = tf.SparseTensor(tf.zeros((0, len(iterator.output_shapes[0])), tf.int64),
                                       tf.zeros((0,), tf.int32), shape_X)
            Y_tensor = self._to_sparse_tensor(encoded_all_intents)

            all_intents_dataset = tf.data.Dataset.from_tensor_slices((X_tensor, Y_tensor)).batch(batch_size)
            self.session.run(iterator.make_initializer(all_intents_dataset))

            while True:
                try:
                    all_intents_embed.append(self.session.run(self.intent_embed))
                except tf.errors.OutOfRangeError:
                    break

        all_intents_embed = np.expand_dims(np.concatenate(all_intents_embed, 0), 0)

        return all_intents_embed

    # process helpers
    # noinspection PyPep8Naming
    def _calculate_message_sim(self,
                               X: np.ndarray,
                               all_Y: np.ndarray
                               ) -> Tuple[np.ndarray, List[float]]:
        """Load tf graph and calculate message similarities"""

        message_sim = self.session.run(self.sim_op,
                                       feed_dict={self.a_in: X,
                                                  self.b_in: all_Y})
        message_sim = message_sim.flatten()  # sim is a matrix

        intent_ids = message_sim.argsort()[::-1]
        message_sim[::-1].sort()

        if self.similarity_type == 'cosine':
            # clip negative values to zero
            message_sim[message_sim < 0] = 0
        elif self.similarity_type == 'inner':
            # normalize result to [0, 1] with softmax
            message_sim = np.exp(message_sim)
            message_sim /= np.sum(message_sim)

        # transform sim to python list for JSON serializing
        return intent_ids, message_sim.tolist()

    # noinspection PyPep8Naming
    def _calculate_message_sim_all(self,
                                   X: np.ndarray,
                                   target_intent_id: int,
                                   ) -> Tuple[np.ndarray, List[float]]:
        """Load tf graph and calculate message similarities"""

        num_unique_intents = self.all_intents_embed_values.shape[1]

        all_cand_ids = list(range(num_unique_intents))

        all_cand_ids.pop(target_intent_id)

        filtered_neg_intent_ids = np.random.choice(np.array(all_cand_ids),NUM_INTENT_CANDIDATES-1)

        filtered_cand_ids = np.append(filtered_neg_intent_ids,[target_intent_id])

        filtered_embed_values = self.all_intents_embed_values[:,filtered_cand_ids]

        # print(filtered_cand_ids, filtered_embed_values.shape)

        message_sim = self.session.run(
            self.sim_all,
            feed_dict={self.a_in: X,
                       self.all_intents_embed_in: filtered_embed_values}
        )

        # print('Shapes in message sim func', message_sim.shape, X.shape, self.all_intents_embed_values.shape)

        message_sim = message_sim.flatten().tolist()  # sim is a matrix

        # print(message_sim)

        message_sim = list(zip(filtered_cand_ids, message_sim))

        # print(intent_sim_pairs)

        # target_intent_id_sim = intent_sim_pairs.pop(target_intent_id)
        #
        # shuffle(intent_sim_pairs)
        #
        # negative_intent_sample = intent_sim_pairs[:NUM_INTENT_CANDIDATES-1]
        #
        # message_sim = negative_intent_sample + [target_intent_id_sim]

        sorted_intents = sorted(message_sim, key=lambda x: x[1])[::-1]

        intent_ids, message_sim = list(zip(*sorted_intents))

        intent_ids = list(intent_ids)
        message_sim = np.array(message_sim)

        # print(intent_ids,message_sim)

        # intent_ids = message_sim.argsort()[::-1]
        # # print(intent_ids)
        # message_sim[::-1].sort()

        if self.similarity_type == 'cosine':
            # clip negative values to zero
            message_sim[message_sim < 0] = 0
        elif self.similarity_type == 'inner':
            # normalize result to [0, 1] with softmax but only over 3*num_neg+1 values
            # message_sim[3*self.num_neg+1:] += -np.inf
            message_sim = np.exp(message_sim)
            message_sim /= np.sum(message_sim)

        # transform sim to python list for JSON serializing
        return np.array(intent_ids), message_sim.tolist()

    def _toarray(self, array_of_sparse):
        if issparse(array_of_sparse):
            return array_of_sparse.toarray()
        elif issparse(array_of_sparse[0]):
            if not self.sequence:
                return np.array([x.toarray() for x in array_of_sparse]).squeeze()
            else:
                seq_len = max([x.shape[0] for x in array_of_sparse])
                X = np.ones([len(array_of_sparse), seq_len, array_of_sparse[0].shape[-1]], dtype=np.int32) * -1
                for i, x in enumerate(array_of_sparse):
                    X[i, :x.shape[0], :] = x.toarray()

                return X
        else:
            return array_of_sparse

    # noinspection PyPep8Naming
    def process(self, message: 'Message', **kwargs: Any) -> None:
        """Return the most likely intent and its similarity to the input."""

        intent = {"name": None, "confidence": 0.0}
        intent_ranking = []

        if self.session is None:
            logger.error("There is no trained tf.session: "
                         "component is either not trained or "
                         "didn't receive enough training data")

        else:
            # get features (bag of words) for a message
            X = message.get("text_features")
            X = self._toarray(X)
            intent_target = message.get("intent_target")

            # Add test intents to existing intents
            if "test_data" in kwargs and not self.is_test_data_featurized:

                logger.info("Embedding test intents and adding to intent list for the first time")
                test_data = kwargs["test_data"]
                new_test_intents = list(set([example.get("intent")
                                                    for example in test_data.intent_examples
                                                    if example.get("intent") not in self.inv_intent_dict.keys()]))

                self.test_intent_dict = {intent: idx + len(self.inv_intent_dict)
                                                         for idx, intent in enumerate(sorted(new_test_intents))}

                self.test_inv_intent_dict = {v: k for k, v in self.test_intent_dict.items()}

                encoded_new_intents = self._create_encoded_intents(self.test_intent_dict, test_data)

                # self.inv_intent_dict.update(self.test_inv_intent_dict)

                # Reindex the intents from 0
                self.test_inv_intent_dict = {i:val for i,(key,val) in enumerate(self.test_inv_intent_dict.items())}
                self.inv_intent_dict = self.test_inv_intent_dict

                self.test_intent_dict = {v: k for k, v in self.inv_intent_dict.items()}

                self.encoded_all_intents = np.append(self.encoded_all_intents, encoded_new_intents, axis=0)

                new_intents_embed_values = self._create_all_intents_embed(encoded_new_intents)
                # print(self.all_intents_embed_values.shape,new_intents_embed_values.shape)
                # self.all_intents_embed_values = np.append(self.all_intents_embed_values, new_intents_embed_values, axis=1)
                self.all_intents_embed_values = new_intents_embed_values

                self.is_test_data_featurized = True

                # print(self.intent_dict)

            # with self.graph.as_default():
            #     if issparse(X):
            #         a_sparse = self._to_sparse_tensor([X])
            #     else:
            #         a_sparse = self._to_sparse_tensor(X, auto2d=False)
            #
            #     a_raw = self._sparse_tensor_to_dense(a_sparse, X[0].shape[-1])
            #
            # X = self.session.run(a_raw)

            # stack encoded_all_intents on top of each other
            # to create candidates for test examples
            # all_Y = self._create_all_Y(X.shape[0])

            # load tf graph and session
            # intent_ids, message_sim = self._calculate_message_sim(X, all_Y)

            intent_target_id = self.test_intent_dict[intent_target]
            # print(intent_target,intent_target_id)

            intent_ids, message_sim = self._calculate_message_sim_all(X, intent_target_id)

            # if X contains all zeros do not predict some label
            if X.any() and intent_ids.size > 0:
                intent = {"name": self.inv_intent_dict[intent_ids[0]],
                          "confidence": message_sim[0]}

                ranking = list(zip(list(intent_ids), message_sim))
                ranking = ranking[:INTENT_RANKING_LENGTH]
                intent_ranking = [{"name": self.inv_intent_dict[intent_idx],
                                   "confidence": score}
                                  for intent_idx, score in ranking]

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    def persist(self,
                file_name: Text,
                model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """

        if self.session is None:
            return {"file": None}

        checkpoint = os.path.join(model_dir, file_name + ".ckpt")

        try:
            os.makedirs(os.path.dirname(checkpoint))
        except OSError as e:
            # be happy if someone already created the path
            import errno
            if e.errno != errno.EEXIST:
                raise
        with self.graph.as_default():
            if not self.gpu_lstm:
                self.graph.clear_collection('message_placeholder')
                self.graph.add_to_collection('message_placeholder',
                                             self.a_in)

                self.graph.clear_collection('intent_placeholder')
                self.graph.add_to_collection('intent_placeholder',
                                             self.b_in)

                self.graph.clear_collection('similarity_op')
                self.graph.add_to_collection('similarity_op',
                                             self.sim_op)

                self.graph.clear_collection('all_intents_embed_in')
                self.graph.add_to_collection('all_intents_embed_in',
                                             self.all_intents_embed_in)
                self.graph.clear_collection('sim_all')
                self.graph.add_to_collection('sim_all',
                                             self.sim_all)

                self.graph.clear_collection('word_embed')
                self.graph.add_to_collection('word_embed',
                                             self.word_embed)
                self.graph.clear_collection('intent_embed')
                self.graph.add_to_collection('intent_embed',
                                             self.intent_embed)

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        placeholder_dims = {'a_in': np.int(self.a_in.shape[-1]),
                            'b_in': np.int(self.b_in.shape[-1])}
        with io.open(os.path.join(
                model_dir,
                file_name + "_placeholder_dims.pkl"), 'wb') as f:
            pickle.dump(placeholder_dims, f)
        with io.open(os.path.join(
                model_dir,
                file_name + "_inv_intent_dict.pkl"), 'wb') as f:
            pickle.dump(self.inv_intent_dict, f)
        with io.open(os.path.join(
                model_dir,
                file_name + "_encoded_all_intents.pkl"), 'wb') as f:
            pickle.dump(self.encoded_all_intents, f)
        with io.open(os.path.join(
                model_dir,
                file_name + "_all_intents_embed_values.pkl"), 'wb') as f:
            pickle.dump(self.all_intents_embed_values, f)

        return {"file": file_name}

    @staticmethod
    def _create_tf_gpu_predict_embed(meta, x_in: 'tf.Tensor',
                                     layer_sizes: List[int], name: Text) -> 'tf.Tensor':
        """Used for prediction if gpu_lstm is true"""

        # mask different length sequences
        mask = tf.sign(tf.reduce_max(x_in, -1) + 1)
        last = mask * tf.cumprod(1 - mask, axis=1, exclusive=True, reverse=True)
        mask = tf.cumsum(last, axis=1, reverse=True)
        real_length = tf.cast(tf.reduce_sum(mask, 1), tf.int32)

        last = tf.expand_dims(last, -1)

        x = tf.nn.relu(x_in)

        if meta['bidirectional']:
            with tf.variable_scope('rnn_encoder_{}'.format(name)):
                single_cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(layer_sizes[0], reuse=tf.AUTO_REUSE)
                cells_fw = [single_cell() for _ in range(len(layer_sizes))]
                cells_bw = [single_cell() for _ in range(len(layer_sizes))]
                x, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, x,
                                                                         dtype=tf.float32,
                                                                         sequence_length=real_length)
        else:
            with tf.variable_scope('rnn_encoder_{}'.format(name)):
                single_cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(layer_sizes[0], reuse=tf.AUTO_REUSE)
                # NOTE: Even if there's only one layer, the cell needs to be wrapped in
                # MultiRNNCell.
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell() for _ in range(len(layer_sizes))])
                # Leave the scope arg unset.
                x, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, sequence_length=real_length)

        x = tf.reduce_sum(x * last, 1)

        return x

    @staticmethod
    def _tf_gpu_sim(meta,
                    a: 'tf.Tensor',
                    b: 'tf.Tensor') -> Tuple['tf.Tensor']:
        """Define similarity

        in two cases:
            sim: between embedded words and embedded intent labels
            sim_emb: between individual embedded intent labels only
        """

        if meta['similarity_type'] == 'cosine':
            # normalize embedding vectors for cosine similarity
            a = tf.nn.l2_normalize(a, -1)
            b = tf.nn.l2_normalize(b, -1)

        if meta['similarity_type'] in {'cosine', 'inner'}:
            sim = tf.reduce_sum(tf.expand_dims(a, 1) * b, -1)

            return sim

        else:
            raise ValueError("Wrong similarity type {}, "
                             "should be 'cosine' or 'inner'"
                             "".format(meta['similarity_type']))

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Text = None,
             model_metadata: 'Metadata' = None,
             cached_component: Optional['EmbeddingIntentClassifier'] = None,
             **kwargs: Any
             ) -> 'EmbeddingIntentClassifier':

        if model_dir and meta.get("file"):
            file_name = meta.get("file")
            checkpoint = os.path.join(model_dir, file_name + ".ckpt")

            graph = tf.Graph()
            with graph.as_default():
                sess = tf.Session()
                if meta['gpu_lstm']:
                    # rebuild tf graph for prediction
                    with io.open(os.path.join(
                            model_dir,
                            file_name + "_placeholder_dims.pkl"), 'rb') as f:
                        placeholder_dims = pickle.load(f)
                    reg = tf.contrib.layers.l2_regularizer(meta['C2'])

                    a_in = tf.placeholder(tf.float32, (None, None, placeholder_dims['a_in']),
                                          name='a')
                    b_in = tf.placeholder(tf.float32, (None, None, placeholder_dims['b_in']),
                                          name='b')
                    a = cls._create_tf_gpu_predict_embed(meta, a_in,
                                                         meta['hidden_layers_sizes_a'],
                                                         name='a_and_b' if meta['share_embedding'] else 'a')
                    word_embed = tf.layers.dense(inputs=a,
                                                 units=meta['embed_dim'],
                                                 kernel_regularizer=reg,
                                                 name='embed_layer_{}'.format('a'),
                                                 reuse=tf.AUTO_REUSE)

                    b = cls._create_tf_gpu_predict_embed(meta, b_in,
                                                         meta['hidden_layers_sizes_b'],
                                                         name='a_and_b' if meta['share_embedding'] else 'b')
                    intent_embed = tf.layers.dense(inputs=b,
                                                   units=meta['embed_dim'],
                                                   kernel_regularizer=reg,
                                                   name='embed_layer_{}'.format('b'),
                                                   reuse=tf.AUTO_REUSE)

                    tiled_intent_embed = cls._tf_sample_neg(intent_embed, None, None,
                                                            tf.shape(word_embed)[0])

                    sim_op = cls._tf_gpu_sim(meta, word_embed, tiled_intent_embed)

                    all_intents_embed_in = tf.placeholder(tf.float32, (None, None, meta['embed_dim']),
                                                          name='all_intents_embed')
                    sim_all = cls._tf_gpu_sim(meta, word_embed, all_intents_embed_in)

                    saver = tf.train.Saver()

                else:
                    saver = tf.train.import_meta_graph(checkpoint + '.meta')

                    # iterator = tf.get_collection('data_iterator')[0]

                    a_in = tf.get_collection('message_placeholder')[0]
                    b_in = tf.get_collection('intent_placeholder')[0]

                    sim_op = tf.get_collection('similarity_op')[0]

                    all_intents_embed_in = tf.get_collection('all_intents_embed_in')[0]
                    sim_all = tf.get_collection('sim_all')[0]

                    word_embed = tf.get_collection('word_embed')[0]
                    intent_embed = tf.get_collection('intent_embed')[0]

                saver.restore(sess, checkpoint)

            with io.open(os.path.join(
                    model_dir,
                    file_name + "_inv_intent_dict.pkl"), 'rb') as f:
                inv_intent_dict = pickle.load(f)
            with io.open(os.path.join(
                    model_dir,
                    file_name + "_encoded_all_intents.pkl"), 'rb') as f:
                encoded_all_intents = pickle.load(f)
            with io.open(os.path.join(
                    model_dir,
                    file_name + "_all_intents_embed_values.pkl"), 'rb') as f:
                all_intents_embed_values = pickle.load(f)


            return cls(
                component_config=meta,
                inv_intent_dict=inv_intent_dict,
                encoded_all_intents=encoded_all_intents,
                all_intents_embed_values=all_intents_embed_values,
                session=sess,
                graph=graph,
                message_placeholder=a_in,
                intent_placeholder=b_in,
                similarity_op=sim_op,
                all_intents_embed_in=all_intents_embed_in,
                sim_all=sim_all,
                word_embed=word_embed,
                intent_embed=intent_embed
            )

        else:
            logger.warning("Failed to load nlu model. Maybe path {} "
                           "doesn't exist"
                           "".format(os.path.abspath(model_dir)))
            return cls(component_config=meta)


class ChronoBiasLayerNormBasicLSTMCell(tf.contrib.rnn.LayerNormBasicLSTMCell):
    """Custom LayerNormBasicLSTMCell that allows chrono initialization
        of gate biases.

        See super class for description.

        See https://arxiv.org/abs/1804.11188
        for details about chrono initialization
    """

    def __init__(self,
                 num_units,
                 forget_bias=1.0,
                 input_bias=0.0,
                 activation=tf.tanh,
                 layer_norm=True,
                 norm_gain=1.0,
                 norm_shift=0.0,
                 dropout_keep_prob=1.0,
                 dropout_prob_seed=None,
                 out_layer_size=None,
                 reuse=None):
        """Initializes the basic LSTM cell

        Additional args:
            input_bias: float, The bias added to input gates.
            out_layer_size: (optional) integer, The number of units in
                the optional additional output layer.
        """
        super(ChronoBiasLayerNormBasicLSTMCell, self).__init__(
            num_units,
            forget_bias=forget_bias,
            activation=activation,
            layer_norm=layer_norm,
            norm_gain=norm_gain,
            norm_shift=norm_shift,
            dropout_keep_prob=dropout_keep_prob,
            dropout_prob_seed=dropout_prob_seed,
            reuse=reuse
        )
        self._input_bias = input_bias
        self._out_layer_size = out_layer_size

    @property
    def output_size(self):
        return self._out_layer_size or self._num_units

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._num_units,
                                             self.output_size)

    @staticmethod
    def _dense_layer(args, layer_size):
        """Optional out projection layer"""
        proj_size = args.get_shape()[-1]
        dtype = args.dtype
        weights = tf.get_variable("kernel",
                                  [proj_size, layer_size],
                                  dtype=dtype)
        bias = tf.get_variable("bias",
                               [layer_size],
                               dtype=dtype)
        out = tf.nn.bias_add(tf.matmul(args, weights), bias)
        return out

    def call(self, inputs, state):
        """LSTM cell with layer normalization and recurrent dropout."""
        c, h = state
        args = tf.concat([inputs, h], 1)
        concat = self._linear(args)
        dtype = args.dtype

        i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)
        if self._layer_norm:
            i = self._norm(i, "input", dtype=dtype)
            j = self._norm(j, "transform", dtype=dtype)
            f = self._norm(f, "forget", dtype=dtype)
            o = self._norm(o, "output", dtype=dtype)

        g = self._activation(j)
        if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
            g = tf.nn.dropout(g, self._keep_prob, seed=self._seed)

        new_c = (c * tf.sigmoid(f + self._forget_bias) +
                 g * tf.sigmoid(i + self._input_bias))  # added input_bias

        # do not do layer normalization on the new c,
        # because there are no trainable weights
        # if self._layer_norm:
        #     new_c = self._norm(new_c, "state", dtype=dtype)

        new_h = self._activation(new_c) * tf.sigmoid(o)

        # added dropout to the hidden state h
        if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
            new_h = tf.nn.dropout(new_h, self._keep_prob, seed=self._seed)

        # add postprocessing of the output
        if self._out_layer_size is not None:
            with tf.variable_scope('out_layer'):
                new_h = self._dense_layer(new_h, self._out_layer_size)

        new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        return new_h, new_state