import spacy
from spacy.tokens import Doc

class MentionGenerator():
    pass


class EntityEmbedder():
    pass


def get_empty_candidates():
    """
    The mention generators always return at least one candidate, but signal
    it with this special candidate
    """
    return {
        "candidate_spans": [[-1, -1]],
        "candidate_entities": [["@@PADDING@@"]],
        "candidate_entity_priors": [[1.0]]
    }


# from https://spacy.io/usage/linguistic-features#custom-tokenizer-example
class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)