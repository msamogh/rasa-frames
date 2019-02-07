import re
from typing import Any, List, Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Token, Tokenizer
from rasa_nlu.training_data import Message, TrainingData


class WhitespaceTokenizer(Tokenizer, Component):
    name = "tokenizer_whitespace"

    provides = ["tokens"]

    defaults = {
        "intent_split_symbol": ' ',
    }

    def __init__(self, component_config=None):
        super(WhitespaceTokenizer, self).__init__(component_config)
        self.intent_split_symbol = self.component_config['intent_split_symbol']

    def train(self, training_data: TrainingData, config: RasaNLUModelConfig,
              **kwargs: Any) -> None:

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))
            if example.get("intent"):
                example.set("intent_tokens",
                            self.tokenize(example.get("intent"),
                                          self.intent_split_symbol))

    def process(self, message: Message, **kwargs: Any) -> None:

        message.set("tokens", self.tokenize(message.text))

    @staticmethod
    def tokenize(text: Text, split=' ') -> List[Token]:

        # there is space or end of string after punctuation
        # because we do not want to replace 10.000 with 10 000
        punctuations = re.findall(r'([.,!?]+(?:\s|$))', text)
        for punctuation in punctuations:
            text = text.replace(punctuation, ' ' + punctuation)

        # words = re.sub(r'[.,!?]+(\s|$)', ' ', text).split()
        words = text.split(split)

        running_offset = 0
        tokens = []
        for word in words:
            word_offset = text.index(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len
            tokens.append(Token(word, word_offset))
        return tokens
