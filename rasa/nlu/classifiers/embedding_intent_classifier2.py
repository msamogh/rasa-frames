import logging

import numpy as np
import os
import typing

from tensorflow.python.keras.utils import Sequence
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa.nlu.classifiers import INTENT_RANKING_LENGTH
from rasa.nlu.components import Component

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import tensorflow as tf
    from rasa.nlu.config import RasaNLUModelConfig
    from rasa.nlu.training_data import TrainingData
    from rasa.nlu.model import Metadata
    from rasa.nlu.training_data import Message

try:
    import tensorflow as tf
    import tensorflow.keras.layers as layers
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
        # training parameters
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
        "similarity_type": "cosine",  # string 'cosine' or 'inner'
        # the number of incorrect intents, the algorithm will minimize
        # their similarity to the input words during training
        "num_neg": 20,
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
        "intent_split_symbol": "_",
        # visualization of accuracy
        # how often to calculate training accuracy
        "evaluate_every_num_epochs": 10,  # small values may hurt performance
        # how many examples to use for calculation of training accuracy
        "evaluate_on_num_examples": 1000,  # large values may hurt performance
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        model: Any = None,  # TODO: fix type
    ) -> None:
        """Declare instant variables with default values"""

        self._check_tensorflow()
        super(EmbeddingIntentClassifier, self).__init__(component_config)

        self._load_params()
        self.model = model

    # init helpers
    def _load_nn_architecture_params(self, config: Dict[Text, Any]) -> None:
        self.hidden_layer_sizes = {
            "a": config["hidden_layers_sizes_a"],
            "b": config["hidden_layers_sizes_b"],
        }

        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]

    def _load_embedding_params(self, config: Dict[Text, Any]) -> None:
        self.embed_dim = config["embed_dim"]
        self.mu_pos = config["mu_pos"]
        self.mu_neg = config["mu_neg"]
        self.similarity_type = config["similarity_type"]
        self.num_neg = config["num_neg"]
        self.use_max_sim_neg = config["use_max_sim_neg"]
        self.random_seed = self.component_config["random_seed"]

    def _load_regularization_params(self, config: Dict[Text, Any]) -> None:
        self.C2 = config["C2"]
        self.C_emb = config["C_emb"]
        self.droprate = config["droprate"]

    def _load_flag_if_tokenize_intents(self, config: Dict[Text, Any]) -> None:
        self.intent_tokenization_flag = config["intent_tokenization_flag"]
        self.intent_split_symbol = config["intent_split_symbol"]
        if self.intent_tokenization_flag and not self.intent_split_symbol:
            logger.warning(
                "intent_split_symbol was not specified, "
                "so intent tokenization will be ignored"
            )
            self.intent_tokenization_flag = False

    def _load_visual_params(self, config: Dict[Text, Any]) -> None:
        self.evaluate_every_num_epochs = config["evaluate_every_num_epochs"]
        if self.evaluate_every_num_epochs < 1:
            self.evaluate_every_num_epochs = self.epochs

        self.evaluate_on_num_examples = config["evaluate_on_num_examples"]

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
                "Failed to import `tensorflow`. "
                "Please install `tensorflow`. "
                "For example with `pip install tensorflow`."
            )

    # training data helpers:
    @staticmethod
    def _create_intent_dict(training_data: "TrainingData") -> Dict[Text, int]:
        """Create intent dictionary"""

        distinct_intents = set(
            [example.get("intent") for example in training_data.intent_examples]
        )
        return {intent: idx for idx, intent in enumerate(sorted(distinct_intents))}

    @staticmethod
    def _create_intent_token_dict(
        intents: List[Text], intent_split_symbol: Text
    ) -> Dict[Text, int]:
        """Create intent token dictionary"""

        distinct_tokens = set(
            [token for intent in intents for token in intent.split(intent_split_symbol)]
        )
        return {token: idx for idx, token in enumerate(sorted(distinct_tokens))}

    def _create_encoded_intents(self, intent_dict: Dict[Text, int]) -> np.ndarray:
        """Create matrix with intents encoded in rows as bag of words.

        If intent_tokenization_flag is off, returns identity matrix.
        """

        if self.intent_tokenization_flag:
            intent_token_dict = self._create_intent_token_dict(
                list(intent_dict.keys()), self.intent_split_symbol
            )

            encoded_all_intents = np.zeros((len(intent_dict), len(intent_token_dict)))
            for key, idx in intent_dict.items():
                for t in key.split(self.intent_split_symbol):
                    encoded_all_intents[idx, intent_token_dict[t]] = 1

            return encoded_all_intents
        else:
            return np.eye(len(intent_dict))

    # noinspection PyPep8Naming
    def _create_all_Y(self, size: int) -> np.ndarray:
        """Stack encoded_all_intents on top of each other

        to create candidates for training examples and
        to calculate training accuracy
        """

        return np.stack([self.encoded_all_intents] * size)

    # noinspection PyPep8Naming
    def _prepare_data_for_training(
        self, training_data: "TrainingData", intent_dict: Dict[Text, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""

        # Matrix n_examples x n_text_features
        X = np.stack([e.get("text_features") for e in training_data.intent_examples])

        # intent names
        intents_for_X = np.array(
            [intent_dict[e.get("intent")] for e in training_data.intent_examples]
        )

        # Matrix n_examples x n_intents
        Y = np.stack(
            [self.encoded_all_intents[intent_idx] for intent_idx in intents_for_X]
        )

        return X, Y, intents_for_X

    def create_model(self, num_text_features, num_labels):
        features_input = tf.keras.Input(shape=(num_text_features,))
        label_input = tf.keras.Input(shape=(None, num_labels))

        embedded_features = self.embed(features_input, "text_features")
        embedded_features = layers.Lambda(lambda x: tf.expand_dims(x, 1))(
            embedded_features
        )
        embedded_labels = self.embed(label_input, "labels")

        output = layers.Concatenate(axis=1)([embedded_features, embedded_labels])

        return tf.keras.Model(inputs=[features_input, label_input], outputs=output)

    def embed(self, layer, name):
        for i, layer_size in enumerate(self.hidden_layer_sizes["a"]):
            layer = layers.Dense(
                layer_size,
                activation="relu",
                kernel_regularizer="l2",
                name="Hidden_Layer_{}_{}".format(name, i),
            )(layer)
            layer = layers.Dropout(
                noise_shape=None,
                rate=self.droprate,
                name="Dropout_Layer_{}_{}".format(name, i),
            )(layer)

        return layers.Dense(
            self.embed_dim,
            kernel_regularizer="l2",
            name="Embedding_Layer_{}".format(name),
        )(layer)

    def _linearly_increasing_batch_size(self, epoch: int) -> int:
        """Linearly increase batch size with every epoch.

        The idea comes from https://arxiv.org/abs/1711.00489
        """

        if not isinstance(self.batch_size, list):
            return int(self.batch_size)

        if self.epochs > 1:
            return int(
                self.batch_size[0]
                + epoch * (self.batch_size[1] - self.batch_size[0]) / (self.epochs - 1)
            )
        else:
            return int(self.batch_size[0])

    def loss(self) -> typing.Callable:
        """Define loss"""

        mu_pos = self.mu_pos
        mu_neg = self.mu_neg

        similarity_type = self.similarity_type
        use_max_sim_neg = self.use_max_sim_neg
        C_emb = self.C_emb

        def calculate_similarities(
            a: "tf.Tensor", b: "tf.Tensor"
        ) -> Tuple["tf.Tensor", "tf.Tensor"]:
            """Define similarity

            in two cases:
                sim: between embedded words and embedded intent labels
                sim_emb: between individual embedded intent labels only
            """

            if similarity_type == "cosine":
                # normalize embedding vectors for cosine similarity
                a = tf.nn.l2_normalize(a, -1)
                b = tf.nn.l2_normalize(b, -1)

            sim = tf.reduce_sum(a * b, -1)
            sim_emb = tf.reduce_sum(b[:, 0:1, :] * b[:, 1:, :], -1)

            return sim, sim_emb

        def _loss(_, embeddings):
            embedded_features = embeddings[:, 0:1, :]
            embedded_labels = embeddings[:, 1:, :]

            similiaries = calculate_similarities(embedded_features, embedded_labels)
            sim = similiaries[0]
            sim_emb = similiaries[1]

            # loss for maximizing similarity with correct action
            loss = tf.maximum(0.0, mu_pos - sim[:, 0])

            if use_max_sim_neg:
                # minimize only maximum similarity over incorrect actions
                max_sim_neg = tf.reduce_max(sim[:, 1:], -1)
                loss += tf.maximum(0.0, mu_neg + max_sim_neg)
            else:
                # minimize all similarities with incorrect actions
                max_margin = tf.maximum(0.0, mu_neg + sim[:, 1:])
                loss += tf.reduce_sum(max_margin, -1)

            # penalize max similarity between intent embeddings
            max_sim_emb = tf.maximum(0.0, tf.reduce_max(sim_emb, -1))
            loss += max_sim_emb * C_emb

            # average the loss over the batch and add regularization losses
            loss = tf.reduce_mean(loss) + tf.losses.get_regularization_loss()

            return loss

        return _loss

    # noinspection PyPep8Naming
    def _train_tf(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        intents_for_X: np.ndarray,
        loss: "tf.Tensor",
        is_training: "tf.Tensor",
        train_op: "tf.Tensor",
    ) -> None:
        """Train tf graph"""

        self.session.run(tf.global_variables_initializer())

        if self.evaluate_on_num_examples:
            logger.info(
                "Accuracy is updated every {} epochs"
                "".format(self.evaluate_every_num_epochs)
            )

        pbar = tqdm(range(self.epochs), desc="Epochs")
        train_acc = 0
        last_loss = 0
        for ep in pbar:
            indices = np.random.permutation(len(X))

            batch_size = self._linearly_increasing_batch_size(ep)
            batches_per_epoch = len(X) // batch_size + int(len(X) % batch_size > 0)

            ep_loss = 0
            for i in range(batches_per_epoch):
                end_idx = (i + 1) * batch_size
                start_idx = i * batch_size
                # batch of text features
                batch_a = X[indices[start_idx:end_idx]]

                # batch of intent_labels
                batch_pos_b = Y[indices[start_idx:end_idx]]
                intents_for_b = intents_for_X[indices[start_idx:end_idx]]
                # add negatives
                batch_b = self._create_batch_b(batch_pos_b, intents_for_b)

                sess_out = self.session.run(
                    {"loss": loss, "train_op": train_op},
                    feed_dict={
                        self.a_in: batch_a,
                        self.b_in: batch_b,
                        is_training: True,
                    },
                )
                ep_loss += sess_out.get("loss") / batches_per_epoch

            # TODO: We still need this
            if self.evaluate_on_num_examples:
                if (
                    ep == 0
                    or (ep + 1) % self.evaluate_every_num_epochs == 0
                    or (ep + 1) == self.epochs
                ):
                    train_acc = self._output_training_stat(
                        X, intents_for_X, is_training
                    )
                    last_loss = ep_loss

                pbar.set_postfix(
                    {
                        "loss": "{:.3f}".format(ep_loss),
                        "acc": "{:.3f}".format(train_acc),
                    }
                )
            else:
                pbar.set_postfix({"loss": "{:.3f}".format(ep_loss)})

        if self.evaluate_on_num_examples:
            logger.info(
                "Finished training embedding classifier, "
                "loss={:.3f}, train accuracy={:.3f}"
                "".format(last_loss, train_acc)
            )

    # noinspection PyPep8Naming
    def _output_training_stat(
        self, X: np.ndarray, intents_for_X: np.ndarray, is_training: "tf.Tensor"
    ) -> np.ndarray:
        """Output training statistics"""

        n = self.evaluate_on_num_examples
        ids = np.random.permutation(len(X))[:n]
        all_Y = self._create_all_Y(X[ids].shape[0])

        train_sim = self.session.run(
            self.sim_op,
            feed_dict={self.a_in: X[ids], self.b_in: all_Y, is_training: False},
        )

        train_acc = np.mean(np.argmax(train_sim, -1) == intents_for_X[ids])
        return train_acc

    def train(
        self,
        training_data: "TrainingData",
        cfg: Optional["RasaNLUModelConfig"] = None,
        **kwargs: Any
    ) -> None:
        """Train the embedding intent classifier on a data set."""

        # {"intent_name": intent_id}
        intent_dict = self._create_intent_dict(training_data)
        if len(intent_dict) < 2:
            logger.error(
                "Can not train an intent classifier. "
                "Need at least 2 different classes. "
                "Skipping training of intent classifier."
            )
            return

        # {intent_id: "intent_name"}
        self.inv_intent_dict = {v: k for k, v in intent_dict.items()}

        # if not tokenization: identity matrix
        self.encoded_all_intents = self._create_encoded_intents(intent_dict)

        # noinspection PyPep8Naming
        X, Y, intents_for_X = self._prepare_data_for_training(
            training_data, intent_dict
        )

        # check if number of negatives is less than number of intents
        logger.debug(
            "Check if num_neg {} is smaller than "
            "number of intents {}, "
            "else set num_neg to the number of intents - 1"
            "".format(self.num_neg, self.encoded_all_intents.shape[0])
        )
        num_neg = min(self.num_neg, self.encoded_all_intents.shape[0] - 1)

        self.model = self.create_model(X.shape[-1], Y.shape[-1])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(), loss=self.loss(), metrics=[]
        )

        sequence = FeatureSequence(
            X, Y, intents_for_X, self.encoded_all_intents, 0, num_negatives=num_neg
        )

        for i in range(self.epochs):
            sequence.on_epoch_end(self._linearly_increasing_batch_size(i))
            self.model.fit(sequence, epochs=1, verbose=1, use_multiprocessing=True)

    def process(self, message: "Message", **kwargs: Any) -> None:
        """Return the most likely intent and its similarity to the input."""

        intent = {"name": None, "confidence": 0.0}
        intent_ranking = []

        if self.model is None:
            logger.error(
                "There is no trained tf.session: "
                "component is either not trained or "
                "didn't receive enough training data"
            )

        else:
            X = message.get("text_features").reshape(1, -1)
            all_Y = self._create_all_Y(X.shape[0])
            predictions = self.model.predict([X, all_Y])

            with tf.Session() as sess:
                message_sim = process_predictions(predictions, self.similarity_type)
                message_sim = sess.run(message_sim)

            message_sim = message_sim[0:1]
            message_sim = message_sim.flatten()  # sim is a matrix

            intent_ids = message_sim.argsort()[::-1]
            message_sim[::-1].sort()

            if self.similarity_type == "cosine":
                # clip negative values to zero
                message_sim[message_sim < 0] = 0
            elif self.similarity_type == "inner":
                # normalize result to [0, 1] with softmax
                message_sim = np.exp(message_sim)
                message_sim /= np.sum(message_sim)

            # if X contains all zeros do not predict some label
            if X.any() and intent_ids.size > 0:
                intent = {
                    "name": self.inv_intent_dict[intent_ids[0]],
                    "confidence": message_sim[0],
                }

                ranking = list(zip(list(intent_ids), message_sim))
                ranking = ranking[:INTENT_RANKING_LENGTH]
                intent_ranking = [
                    {"name": self.inv_intent_dict[intent_idx], "confidence": score}
                    for intent_idx, score in ranking
                ]

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """
        file_name = file_name + ".h5"
        target_path = os.path.join(model_dir, file_name)
        import tensorflow.keras.models as keras_models

        keras_models.save_model(self.model, target_path)

        return {"file": file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: "Metadata" = None,
        cached_component: Optional["EmbeddingIntentClassifier"] = None,
        **kwargs: Any
    ) -> "EmbeddingIntentClassifier":
        # TODO finish this

        if model_dir and meta.get("file"):
            file_name = meta.get("file")
            path = os.path.join(model_dir, file_name)

            import tensorflow.keras.models as keras_models

            model = keras_models.load_model(path)

            return cls(meta, model)


def process_predictions(predictions, similarity_type):
    predictions = tf.convert_to_tensor(predictions)
    embedded_features = predictions[:, 0:1, :]
    embedded_labels = predictions[:, 1:, :]

    if similarity_type == "cosine":
        # normalize embedding vectors for cosine similarity
        a = tf.nn.l2_normalize(embedded_features, -1)
        b = tf.nn.l2_normalize(embedded_labels, -1)

        return tf.reduce_sum(a * b, -1)


class FeatureSequence(Sequence):
    def __init__(
        self,
        feature_x,
        Y,
        intents_for_X,
        encoded_all_intents,
        batch_size,
        num_negatives,
    ):
        self.feature_x, self.Y = feature_x, Y
        self.intents_for_X = intents_for_X
        self.encoded_all_intents = encoded_all_intents
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.indices = np.random.permutation(len(self.feature_x))

    def __len__(self):
        return len(self.feature_x) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        slice = self.indices[start:end]
        batch_features = self.feature_x[slice]
        batch_labels = self.Y[slice]
        intents_for_b = self.intents_for_X[slice]
        batch_labels = self._create_batch_b(batch_labels, intents_for_b)

        return [batch_features, batch_labels], self.empty_y

    def on_epoch_end(self, batch_size=10):
        self.indices = np.random.permutation(len(self.feature_x))
        self.batch_size = batch_size
        self.empty_y = np.empty((batch_size, 0))

    def _create_batch_b(
        self, batch_pos_b: np.ndarray, intent_ids: np.ndarray
    ) -> np.ndarray:
        """Create batch of intents.

        Where the first is correct intent
        and the rest are wrong intents sampled randomly
        """

        # new dimensions: batch_size x 1 x n_features
        batch_pos_b = batch_pos_b[:, np.newaxis, :]

        # sample negatives: batch_size x num_negatives x n_features
        batch_neg_b = np.zeros(
            (batch_pos_b.shape[0], self.num_negatives, batch_pos_b.shape[-1])
        )
        # per entry in batch
        for b in range(batch_pos_b.shape[0]):
            # create negative indexes out of possible ones
            # except for correct index of b
            negative_indexes = [
                i
                for i in range(self.encoded_all_intents.shape[0])
                if i != intent_ids[b]
            ]

            # get num_neg
            negs = np.random.choice(negative_indexes, size=self.num_negatives)

            # get_encoded
            batch_neg_b[b] = self.encoded_all_intents[negs]
        # batch_size x (1 + num_negatives) x featurzes
        return np.concatenate([batch_pos_b, batch_neg_b], 1)
