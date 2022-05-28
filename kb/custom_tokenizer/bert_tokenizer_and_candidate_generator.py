from typing import Dict, List
import copy

import numpy as np

from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer

from kb.custom_tokenizer.common import MentionGenerator, get_empty_candidates

start_token = "[CLS]"
sep_token = "[SEP]"


def truncate_sequence_pair(word_piece_tokens_a, word_piece_tokens_b, max_word_piece_sequence_length):
    length_a = sum([len(x) for x in word_piece_tokens_a])
    length_b = sum([len(x) for x in word_piece_tokens_b])
    while max_word_piece_sequence_length < length_a + length_b:
        if length_a < length_b:
            discarded = word_piece_tokens_b.pop()
            length_b -= len(discarded)
        else:
            discarded = word_piece_tokens_a.pop()
            length_a -= len(discarded)


class TokenizerAndCandidateGenerator():
    pass

class BertTokenizerAndCandidateGenerator(TokenizerAndCandidateGenerator):
    def __init__(self,
                 entity_candidate_generators: Dict[str, MentionGenerator],
                 bert_model_type: str,
                 do_lower_case: bool,
                 whitespace_tokenize: bool = True,
                 max_word_piece_sequence_length: int = 512) -> None:
        """
        Note: the fields need to be used with a pre-generated vocabulary
        that contains the entity id namespaces and the bert name space.
        entity_indexers = {'wordnet': indexer for wordnet entities,
                          'wiki': indexer for wiki entities}
        """

        # load BertTokenizer from huggingface
        self.candidate_generators = entity_candidate_generators
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            bert_model_type, do_lower_case=do_lower_case
        )
        self.bert_word_tokenizer = BasicTokenizer(do_lower_case=False)
        # Target length should include start and end token
        self.max_word_piece_sequence_length = max_word_piece_sequence_length

        self.do_lowercase = do_lower_case
        self.whitespace_tokenize = whitespace_tokenize
        self.dtype = np.float32

    def _word_to_word_pieces(self, word):
        if self.do_lowercase and word not in self.bert_tokenizer.basic_tokenizer.never_split:
            word = word.lower()
        return self.bert_tokenizer.wordpiece_tokenizer.tokenize(word)

    def tokenize_and_generate_candidates(self, text_a: str, text_b: str = None):
        """
        # run BertTokenizer.basic_tokenizer.tokenize on sentence1 and sentence2 to word tokenization
        # generate candidate mentions for each of the generators and for each of sentence1 and 2 from word tokenized text
        # run BertTokenizer.wordpiece_tokenizer on sentence1 and sentence2
        # truncate length, add [CLS] and [SEP] to word pieces
        # compute token offsets
        # combine candidate mention spans from sentence1 and sentence2 and remap to word piece indices

        returns:

        {'tokens': List[str], the word piece strings with [CLS] [SEP]
         'segment_ids': List[int] the same length as 'tokens' with 0/1 for sentence1 vs 2
         'candidates': Dict[str, Dict[str, Any]],
            {'wordnet': {'candidate_spans': List[List[int]],
                         'candidate_entities': List[List[str]],
                         'candidate_entity_prior': List[List[float]],
                         'segment_ids': List[int]},
             'wiki': ...}
        }
        """
        offsets_a, grouped_wp_a, tokens_a = self._tokenize_text(text_a)

        if text_b is not None:
            offsets_b, grouped_wp_b, tokens_b = self._tokenize_text(text_b)
            truncate_sequence_pair(grouped_wp_a, grouped_wp_b, self.max_word_piece_sequence_length - 3)
            offsets_b = offsets_b[:len(grouped_wp_b)]
            tokens_b = tokens_b[:len(grouped_wp_b)]
            instance_b = self._generate_sentence_entity_candidates(tokens_b, offsets_b)
            word_piece_tokens_b = [word_piece for word in grouped_wp_b for word_piece in word]
        else:
            length_a = sum([len(x) for x in grouped_wp_a])
            while self.max_word_piece_sequence_length - 2 < length_a:
                discarded = grouped_wp_a.pop()
                length_a -= len(discarded)

        word_piece_tokens_a = [word_piece for word in grouped_wp_a for word_piece in word]
        offsets_a = offsets_a[:len(grouped_wp_a)]
        tokens_a = tokens_a[:len(grouped_wp_a)]
        instance_a = self._generate_sentence_entity_candidates(tokens_a, offsets_a)

        # If we got 2 sentences.
        if text_b is not None:
            # Target length should include start and two end tokens, and then be divided equally between both sentences
            # Note that this will result in potentially shorter documents than original target length,
            # if one (or both) of the sentences are shorter than half the target length.
            tokens = [start_token] + word_piece_tokens_a + [sep_token] + word_piece_tokens_b + [sep_token]
            segment_ids = (len(word_piece_tokens_a) + 2) * [0] + (len(word_piece_tokens_b) + 1) * [1]
            offsets_a = [x + 1 for x in offsets_a]
            offsets_b = [x + 2 + len(word_piece_tokens_a) for x in offsets_b]
        # Single sentence
        else:
            tokens = [start_token] + word_piece_tokens_a + [sep_token]
            segment_ids = len(tokens) * [0]
            offsets_a = [x + 1 for x in offsets_a]
            offsets_b = None

        for name in instance_a.keys():
            for span in instance_a[name]['candidate_spans']:
                span[0] += 1
                span[1] += 1


        # Dict[str, Sequence] 
        fields= {}

        # concatanating both sentences (for both tokens and ids)
        if text_b is None:
            candidates = instance_a
        else:
            #: Dict[str, Field] 
            candidates= {}

            # Merging candidate lists for both sentences.
            for entity_type in instance_b:
                candidate_instance_a = instance_a[entity_type]
                candidate_instance_b = instance_b[entity_type]

                candidates[entity_type] = {}

                for span in candidate_instance_b['candidate_spans']:
                    span[0] += len(word_piece_tokens_a) + 2
                    span[1] += len(word_piece_tokens_a) + 2

                # Merging each of the fields.
                for key in ['candidate_entities', 'candidate_spans', 'candidate_entity_priors']:
                    candidates[entity_type][key] = candidate_instance_a[key] + candidate_instance_b[key]


        for entity_type in candidates.keys():
            # deal with @@PADDING@@
            if len(candidates[entity_type]['candidate_entities']) == 0:
                candidates[entity_type] = get_empty_candidates()
            else:
                padding_indices = []
                has_entity = False
                for cand_i, candidate_list in enumerate(candidates[entity_type]['candidate_entities']):
                    if candidate_list == ["@@PADDING@@"]:
                        padding_indices.append(cand_i)
                        candidates[entity_type]["candidate_spans"][cand_i] = [-1, -1]
                    else:
                        has_entity = True
                indices_to_remove = []
                if has_entity and len(padding_indices) > 0:
                    # remove all the padding entities since have some valid
                    indices_to_remove = padding_indices
                elif len(padding_indices) > 0:
                    assert len(padding_indices) == len(candidates[entity_type]['candidate_entities'])
                    indices_to_remove = padding_indices[1:]
                for ind in reversed(indices_to_remove):
                    del candidates[entity_type]["candidate_spans"][ind]
                    del candidates[entity_type]["candidate_entities"][ind]
                    del candidates[entity_type]["candidate_entity_priors"][ind]

        # get the segment ids for the spans
        for key, cands in candidates.items():
            span_segment_ids = []
            for candidate_span in cands['candidate_spans']:
                span_segment_ids.append(segment_ids[candidate_span[0]])
            candidates[key]['candidate_segment_ids'] = span_segment_ids

        fields['tokens'] = tokens
        fields['segment_ids'] = segment_ids
        fields['candidates'] = candidates
        fields['offsets_a'] = offsets_a
        fields['offsets_b'] = offsets_b
        return fields

    def _tokenize_text(self, text):
        if self.whitespace_tokenize:
            tokens = text.split()
        else:
            tokens = self.bert_word_tokenizer.tokenize(text)

        word_piece_tokens = []
        offsets = [0]
        for token in tokens:
            word_pieces = self._word_to_word_pieces(token)
            offsets.append(offsets[-1] + len(word_pieces))
            word_piece_tokens.append(word_pieces)
        del offsets[0]
        return offsets, word_piece_tokens, tokens

    def _generate_sentence_entity_candidates(self, tokens, offsets):
        """
        Tokenize sentence, trim it to the target length, and generate entity candidates.
        :param sentence
        :param target_length: The length of the output sentence in terms of word pieces.
        :return: Dict[str, Dict[str, Any]],
            {'wordnet': {'candidate_spans': List[List[int]],
                         'candidate_entities': List[List[str]],
                         'candidate_entity_priors': List[List[float]]},
             'wiki': ...}

        """
        assert len(tokens) == len(offsets), f'Length of tokens {len(tokens)} must equal that of offsets {len(offsets)}.'
        entity_instances = {}
        for name, mention_generator in self.candidate_generators.items():
            entity_instances[name] = mention_generator.get_mentions_raw_text(' '.join(tokens), whitespace_tokenize=True)

        for name, entities in entity_instances.items():
            candidate_spans = entities["candidate_spans"]
            adjusted_spans = []
            for start, end in candidate_spans:
                if 0 < start:
                    adjusted_span = [offsets[start - 1], offsets[end] - 1]
                else:
                    adjusted_span = [0, offsets[end] - 1]
                adjusted_spans.append(adjusted_span)
            entities['candidate_spans'] = adjusted_spans
            entity_instances[name] = entities
        return entity_instances

    #custom
    def convert_tokens_candidates_to_array(self, tokens_and_candidates, entity_vocabulary):
        """
        tokens_and_candidates is the return from a previous call to
        generate_sentence_entity_candidates. entity_vocabulary is the vocabulary used to convert from
        text entiy to ids. Converts the dict to a dict of numpy array
        """
        fields = {}

        fields['tokens'] = np.array([self.bert_tokenizer.vocab[t] for t in tokens_and_candidates['tokens']])

        fields['segment_ids'] = np.array(tokens_and_candidates['segment_ids'])
        
        all_candidates = {}
        for key, entity_candidates in tokens_and_candidates['candidates'].items():
            # pad the ids and prior to create 
            candidate_entity_prior = copy.deepcopy(entity_candidates['candidate_entity_priors'])
            max_cands = max(len(p) for p in candidate_entity_prior)
            for p in candidate_entity_prior:
                len_diff = max_cands - len(p)
                if len_diff>0:
                    p.extend([0.0] * len_diff)
            
            np_prior = np.array(candidate_entity_prior)
            
            candidate_ids = []
            for mention_candidate_entities in entity_candidates['candidate_entities']:
                mention_candidate_ids= [entity_vocabulary.get_token_index(entity,namespace='entity') for entity in mention_candidate_entities]
                candidate_ids.append(mention_candidate_ids)
            
            for mention_candidate_ids in candidate_ids:
                len_diff = max_cands - len(mention_candidate_ids)
                if len_diff>0:
                    mention_candidate_ids.extend([0] * len_diff)
            
            np_id = np.array(candidate_ids)

            #removed candidate entities text
            candidate_fields = {
                "candidate_entity_ids": np_id,
                "candidate_entity_priors": np_prior,
                "candidate_spans":np.array(entity_candidates['candidate_spans']),
                "candidate_segment_ids": np.array(entity_candidates['candidate_segment_ids'])
            }
            all_candidates[key] = candidate_fields

        fields["candidates"] = all_candidates

        return fields

class PretokenizedTokenizerAndCandidateGenerator(BertTokenizerAndCandidateGenerator):
    """
    Simple modification to the ``BertTokenizerAndCandidateGenerator``. We assume data comes
    pre-tokenized, so only wordpiece splitting is performed.

    # TODO: mypy is not going to like us calling ``tokenize_and_generate_candidates()`` on lists
    # instead of strings. Maybe update type annotations in ``BertTokenizerAndCandidateGenerator``?
    """
    def _tokenize_text(self, tokens: List[str]):
        word_piece_tokens = []
        offsets = [0]
        for token in tokens:
            # Stupid hack
            if token in ['[SEP]', '[MASK]']:
                word_pieces = [token]
            else:
                word_pieces = self._word_to_word_pieces(token)
            offsets.append(offsets[-1] + len(word_pieces))
            word_piece_tokens.append(word_pieces)
        del offsets[0]
        return offsets, word_piece_tokens, tokens