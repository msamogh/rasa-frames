import re
from typing import Any, List, Text

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers import Token, Tokenizer
from rasa.nlu.training_data import Message, TrainingData


class WhitespaceTokenizer(Tokenizer, Component):

    provides = ["tokens"]

    defaults = {
        "intent_split_symbol": ' ',
        "add_class_label": False
    }

    def __init__(self, component_config=None):
        super(WhitespaceTokenizer, self).__init__(component_config)
        self.intent_split_symbol = self.component_config['intent_split_symbol']
        self.add_class_label = self.component_config['add_class_label']
        self.is_test_data_featurized = False

    def train(self, training_data: TrainingData, config: RasaNLUModelConfig,
              **kwargs: Any) -> None:

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))
            if example.get("intent"):
                example.set("intent_tokens",
                            self.tokenize(example.get("intent"),
                                          self.intent_split_symbol))

    def process(self, message: Message, **kwargs: Any) -> None:

        if "test_data" in kwargs and not self.is_test_data_featurized:

            test_data = kwargs["test_data"]

            for example in test_data.training_examples:
                if example.get("intent"):
                    example.set("intent_tokens",
                                self.tokenize(example.get("intent"),
                                              self.intent_split_symbol))

        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text: Text, split=' ') -> List[Token]:

        # remove 'not a word character' if
        words = re.sub(
            # there is a space or an end of a string after it
            r"[^\w#@&]+(?=\s|$)|"
            # there is a space or beginning of a string before it
            # not followed by a number
            r"(\s|^)[^\w#@&]+(?=[^0-9\s])|"
            # not in between numbers and not . or @ or & or - or #
            # e.g. 10'000.00 or blabla@gmail.com
            # and not url characters
            r"(?<=[^0-9\s])[^\w._~:/?#\[\]()@!$&*+,;=-]+(?=[^0-9\s])",
            split, text
        ).split(split)

        running_offset = 0
        tokens = []
        for word in words:
            word_offset = text.index(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len
            tokens.append(Token(word, word_offset))

        if self.add_class_label:
            # using BERT logic, add `[CLS]` class label token
            tokens.append(Token('__CLS__', len(text)))

        return tokens
