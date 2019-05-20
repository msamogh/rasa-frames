import re
from typing import Any, List, Text

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers import Token, Tokenizer
from rasa.nlu.training_data import Message, TrainingData
import numpy as np
import copy


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

    @staticmethod
    def _find_example_for_text(text, examples):
        out = []
        for ex in examples:
            if ex.text == text:
                out.append(ex)
        return out

    def train(self, training_data: TrainingData, config: RasaNLUModelConfig,
              **kwargs: Any) -> None:

        max_es = 0
        new_es = []
        for example in training_data.intent_examples:
            es = self._find_example_for_text(example.text, training_data.intent_examples)
            for i, e in enumerate(es):
                if len(es) == 1:
                    # e1 = copy.deepcopy(e)
                    # e2 = copy.deepcopy(e)
                    # e3 = copy.deepcopy(e)
                    e.set("random_number", np.random.randint(7))
                    # e1.set("random_number", np.random.randint(7))
                    # e2.set("random_number", np.random.randint(7))
                    # e3.set("random_number", np.random.randint(7))
                    # new_es.extend([e1, e2, e3])
                else:
                    e.set("random_number", i)
                    # print(e.get("intent"))
            # if len(es) > max_es:
            #     max_es = len(es)
            # print("-----")
            # for i, e in enumerate(es):
            #     tokens =
            #     e.set("tokens") = 'rand_{}'.format(i) + e.text
        # print(max_es)
        # exit()
        # training_data.training_examples.extend(new_es)

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text, random_number=example.get("random_number")))
            if example.get("intent"):
                example.set("intent_tokens",
                            self.tokenize(example.get("intent"),
                                          self.intent_split_symbol))

    def process(self, message: Message, **kwargs: Any) -> None:
        message.set("random_number", np.random.randint(7))
        message.set("tokens", self.tokenize(message.text, random_number=message.get("random_number")))

    def tokenize(self, text: Text, split=' ', random_number=None) -> List[Token]:

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
            cls_string = '__CLS__'
            if random_number is not None:
                cls_string += str(random_number)
            # using BERT logic, add `[CLS]` class label token
            tokens.append(Token(cls_string, len(text)))

        return tokens
