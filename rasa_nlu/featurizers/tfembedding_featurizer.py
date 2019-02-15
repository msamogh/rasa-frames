import numpy as np
import typing
from typing import Any

from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.featurizers import Featurizer
from rasa_nlu.training_data import Message, TrainingData
from rasa_nlu.model import Interpreter

if typing.TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc


def ndim(spacy_nlp: 'Language') -> int:
    """Number of features used to represent a document / sentence."""
    return spacy_nlp.vocab.vectors_length


def features_for_doc(doc: 'Doc') -> np.ndarray:
    """Feature vector for a single document / sentence."""
    return doc.vector


class TfEmbeddingFeaturizer(Featurizer):
    name = "tfembedding_featurizer"

    provides = ["text_features"]

    def __init__(self, component_config=None):
        super(TfEmbeddingFeaturizer, self).__init__(component_config)
        self.pretrained_model = Interpreter.load(self.component_config['model'])

    def train(self,
              training_data: TrainingData,
              config: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        i=0
        for example in training_data.intent_examples:
            i+=1
            self._set_features(example)

    def process(self, message: Message, test_data=None, **kwargs: Any) -> None:

        self._set_features(message)
        return test_data

    def _set_features(self, message):
        """Adds the spacy word vectors to the messages text features."""

        for component in self.pretrained_model.pipeline[:-1]:
            component.process(message)
        X = np.expand_dims(message.get("text_features"), axis=0)
        classifier = self.pretrained_model.pipeline[-1]
        if classifier.all_Y is None:
            classifier.all_Y = classifier._create_all_Y(X.shape[0])
        word_embed_values = classifier.session.run(classifier.word_embed,
                                                   feed_dict={classifier.a_in: X,
                                                              classifier.b_in: classifier.all_Y})
        # print(word_embed_values)
        message.set("text_features", word_embed_values[0])
