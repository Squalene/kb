
from typing import Union, List,Dict,Sequence
import numpy as np
from allennlp.common import Params
from allennlp.data import Instance, DataIterator

start_token = "[CLS]"
sep_token = "[SEP]"

# from kb.include_all import TokenizerAndCandidateGenerator
from kb.bert_pretraining_reader import replace_candidates_with_mask_entity

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

class CustomKnowBertBatchifier:
    """
    Takes a list of sentence strings and returns a tensor dict usable with
    a KnowBert model
    """
    def __init__(self, tokenizer_and_candidate_generator, entity_vocabulary, batch_size=32,
                       masking_strategy=None):

        #Given 
        self.tokenizer_and_candidate_generator = tokenizer_and_candidate_generator 
        self.tokenizer_and_candidate_generator.whitespace_tokenize = False

        assert masking_strategy is None or masking_strategy == 'full_mask'
        self.masking_strategy = masking_strategy

        #map entity -> embedding id
        self.entity_vocabulary = entity_vocabulary
        self.batch_size = batch_size
        self.iterator = DataIterator.from_params(
            Params({"type": "basic", "batch_size": batch_size})
        )
        self.iterator.index_with(self.entity_vocabulary)

    def _replace_mask(self, s):
        return s.replace('[MASK]', ' [MASK] ')

    def iter_batches(self, sentences_or_sentence_pairs: Union[List[str], List[List[str]]], verbose=True):
        # create instances
        instances = []
        for sentence_or_sentence_pair in sentences_or_sentence_pairs:
            if isinstance(sentence_or_sentence_pair, list):
                assert len(sentence_or_sentence_pair) == 2
                tokens_candidates = self.tokenizer_and_candidate_generator.\
                        tokenize_and_generate_candidates(
                                self._replace_mask(sentence_or_sentence_pair[0]),
                                self._replace_mask(sentence_or_sentence_pair[1]))
            else:
                tokens_candidates = self.tokenizer_and_candidate_generator.\
                        tokenize_and_generate_candidates(self._replace_mask(sentence_or_sentence_pair))

            if verbose:
                print(self._replace_mask(sentence_or_sentence_pair))
                print(tokens_candidates['tokens'])

            # now modify the masking if needed
            if self.masking_strategy == 'full_mask':
                # replace the mask span with a @@mask@@ span
                masked_indices = [index for index, token in enumerate(tokens_candidates['tokens']) if token == '[MASK]']

                spans_to_mask = set([(i, i) for i in masked_indices])
                replace_candidates_with_mask_entity(
                        tokens_candidates['candidates'], spans_to_mask
                )

                # now make sure the spans are actually masked
                for key in tokens_candidates['candidates'].keys():
                    for span_to_mask in spans_to_mask:
                        found = False
                        for span in tokens_candidates['candidates'][key]['candidate_spans']:
                            if tuple(span) == tuple(span_to_mask):
                                found = True
                        if not found:
                            tokens_candidates['candidates'][key]['candidate_spans'].append(list(span_to_mask))
                            tokens_candidates['candidates'][key]['candidate_entities'].append(['@@MASK@@'])
                            tokens_candidates['candidates'][key]['candidate_entity_priors'].append([1.0])
                            tokens_candidates['candidates'][key]['candidate_segment_ids'].append(0)
                            # hack, assume only one sentence
                            assert not isinstance(sentence_or_sentence_pair, list)

            #fields:
                #tokens: list(string),  
                # candidates['wiki']['candidate_entities']:list[list[string]], 
                # candidates['wiki']['candidate_spans']:
                # candidates['wiki']['candidate_entity_priors']:
                # candidates['wiki']['candidate_segment_ids']
                # candidates['wiki']['offset_a']
                # candidates['wiki']['offset_b']

            #print(tokens_candidates['tokens'])
            # print('\n')
            fields = self.tokenizer_and_candidate_generator.\
                convert_tokens_candidates_to_fields(tokens_candidates)

            #TODO: custom
            dicts = self.tokenizer_and_candidate_generator.convert_tokens_candidates_to_tensor(tokens_candidates)

            print(dicts['candidates'].keys())
            for key in tokens_candidates['candidates'].keys():
                #List of list of entities
                candidate_entities = dicts['candidates'][key]['candidate_entities']
                candidate_entities_ids = []
                for mention_candidate_entities in candidate_entities:
                    mention_candidate_ids= []
                    for candidate_entity in mention_candidate_entities:
                        id = self.entity_vocabulary.get_token_index(candidate_entity,namespace='entity')
                        #print(f"Candidate entity {candidate_entity}, corresponding id: {id}")
                        mention_candidate_ids.append(id)
                        #candidate_entities_ids.append(id)
                    candidate_entities_ids.append(mention_candidate_ids)
            

            dicts['candidates']['wiki']['candidate_ids']= np.array(candidate_entities_ids)

            #TODO: add padding if necessary => batcher function

            print(dicts['candidates']['wiki']['candidate_entities'])
            print(dicts['tokens'].shape)
            print(dicts['segment_ids'].shape)
            print(dicts['candidates']['wiki']['candidate_entity_priors'].shape)
            print(dicts['candidates']['wiki']['candidate_spans'].shape)
            print(dicts['candidates']['wiki']['candidate_segment_ids'].shape)
            print(dicts['candidates']['wiki']['candidate_ids'].shape)

            instances.append(Instance(fields))

            #iterator does: convert tokens to 

        for batch in self.iterator(instances, num_epochs=1, shuffle=False):
            yield batch
