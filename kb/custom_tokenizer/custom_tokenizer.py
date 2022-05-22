from collections import defaultdict
import copy
from typing import Union, List,Dict,Sequence
import numpy as np
from allennlp.common import Params
from allennlp.data import Instance, DataIterator
import torch

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
        instances_prev = []
        #Contains all the field of processing one sentence
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

        
            #TODO: custom
            arrays = self.tokenizer_and_candidate_generator.convert_tokens_candidates_to_array(tokens_candidates,self.entity_vocabulary)

            #TODO: add padding if necessary => batcher function

            # print(dicts['candidates']['wiki']['candidate_entities'])
            print(f"Tokens shape {arrays['tokens'].shape}")
            print(f"Segment ids shape {arrays['segment_ids'].shape}")
            print(f"Candidate spans shape {arrays['candidates']['wiki']['candidate_spans'].shape}")
            print(f"Candidate_segment_ids shape {arrays['candidates']['wiki']['candidate_segment_ids'].shape}")
            print(f"Candidate_entity_priors shape {arrays['candidates']['wiki']['candidate_entity_priors'].shape}")
            print(f"candidate_entity_ids shape {arrays['candidates']['wiki']['candidate_entity_ids'].shape}")

            instances.append(arrays)
            #iterator does: convert tokens to 

            fields = self.tokenizer_and_candidate_generator.convert_tokens_candidates_to_fields(tokens_candidates)

            instances_prev.append(Instance(fields))
            #fields:
                    #tokens: list(string),  
                    # candidates['wiki']['candidate_entities']:list[list[string]], 
                    # candidates['wiki']['candidate_spans']:
                    # candidates['wiki']['candidate_entity_priors']:
                    # candidates['wiki']['candidate_segment_ids']
                    # candidates['wiki']['offset_a']
                    # candidates['wiki']['offset_b']

        print(f"Number of instances is {len(instances)}")
        for batch in self.batchify(instances):
            convert_to_tensor(batch)
            yield batch

        
        # for batch in self.iterator(instances_prev, num_epochs=1, shuffle=False):
        #     yield batch

    #Takes a list of numpy array of varying length and must create batches
    def batchify(self,instances):
        #Must padd:
        #tokens ids with 0
        #segment ids with 0
        #candidate span with [-1,-1]
        #candidate entity_priors with 0 (on 2 axis)
        #candidate ids with 0 (on 2 axis)
        #candidate segments_ids with 0 (on 2 axis)
        
        
       for i in range(0, len(instances), self.batch_size):
            instance_batch = instances[i:i+self.batch_size]
            print(f"Number of instances is {len(instance_batch)}")
            max_tokens = max(len(instance["tokens"]) for instance in instance_batch)
            print(f"Max tokens {max_tokens}")
            batch = {}
            batch['tokens'] = {}
            batch['tokens']['tokens'] = []
            batch['segment_ids'] = []
            #Create batch for tokens segment ids
            for instance in instance_batch:
                padded_tokens= instance["tokens"]
                padded_segment_ids = instance["segment_ids"]
                padded_tokens = pad_to_shape(padded_tokens,(max_tokens),0)
                padded_segment_ids = pad_to_shape(padded_segment_ids,(max_tokens),0)
                # len_diff = max_tokens - len(padded_tokens)
                # if len_diff>0:
                #     padded_tokens = np.concatenate((padded_tokens,np.zeros(len_diff)))
                #     padded_segment_ids = np.concatenate((padded_segment_ids,np.zeros(len_diff)))

                batch['tokens']['tokens'].append(padded_tokens)
                batch['segment_ids'].append(padded_segment_ids)

            batch['tokens']['tokens']= np.array(batch['tokens']['tokens'],np.int64)
            batch['segment_ids']= np.array(batch['segment_ids'],np.int64)

            print(f"Batch tokens shape {batch['tokens']['tokens'].shape}")
            print(f"Batch segment_ids shape { batch['segment_ids'].shape}")

            #IMPORTANT: assume the number of candidate entities is already padded inside on instance(done by )
            #=> must padd across multiple instances
            kb_keys = instance_batch[0]['candidates'].keys()
            max_detected_entities = {}
            max_candidate_entities = {}

            for key in kb_keys:
                max_detected_entities[key] = max(instance['candidates'][key]['candidate_entity_ids'].shape[0] for instance in instance_batch)
                max_candidate_entities[key] = max(instance['candidates'][key]['candidate_entity_ids'].shape[1] for instance in instance_batch)

            print(f"Max number of detected entities {max_detected_entities}")
            print(f"Max number of candidate entities {max_candidate_entities}")

            batch['candidates'] = {}
            for key in kb_keys:
                batch['candidates'][key] = {}
                batch['candidates'][key]['candidate_spans'] = []
                batch['candidates'][key]['candidate_entities'] = {}
                batch['candidates'][key]['candidate_entities']['ids']=[]
                batch['candidates'][key]['candidate_entity_priors'] = []
                batch['candidates'][key]['candidate_segment_ids'] = []
                
            for instance in instance_batch:
                for key, entity_candidates in instance['candidates'].items():
                    max_entity = max_detected_entities[key]
                    max_candidate = max_candidate_entities[key]

                    padded_entity_ids = entity_candidates['candidate_entity_ids']
                    padded_spans = entity_candidates['candidate_spans']
                    padded_entity_priors = entity_candidates['candidate_entity_priors']
                    padded_entity_segment_ids = entity_candidates['candidate_segment_ids']

                    #Same for all here
                    # len_diff = max_entity-padded_entity_ids.shape[0]
                    
                    padded_spans = pad_to_shape(padded_spans,(max_entity,2),-1)
                    padded_entity_ids = pad_to_shape(padded_entity_ids,(max_entity,max_candidate),0)
                    padded_entity_priors = pad_to_shape(padded_entity_priors,(max_entity,max_candidate),0)
                    padded_entity_segment_ids = pad_to_shape(padded_entity_segment_ids,(max_entity),0)

                    batch['candidates'][key]['candidate_spans'].append(padded_spans)
                    batch['candidates'][key]['candidate_entities']['ids'].append(padded_entity_ids)
                    batch['candidates'][key]['candidate_entity_priors'].append(padded_entity_priors)
                    batch['candidates'][key]['candidate_segment_ids'].append(padded_entity_segment_ids)
            
            batch['candidates'][key]['candidate_spans'] = np.array(batch['candidates'][key]['candidate_spans'],np.int64)
            batch['candidates'][key]['candidate_entities']['ids'] = np.array(batch['candidates'][key]['candidate_entities']['ids'],np.int64)
            batch['candidates'][key]['candidate_entity_priors'] = np.array(batch['candidates'][key]['candidate_entity_priors'],np.float32)
            batch['candidates'][key]['candidate_segment_ids']= np.array(batch['candidates'][key]['candidate_segment_ids'],np.int64)
            
            print(f"Batch candidate_spans shape {batch['candidates'][key]['candidate_spans'].shape}")
            print(f"Batch candidate_entity_ids shape {batch['candidates'][key]['candidate_entities']['ids'].shape}")
            print(f"Batch candidate_entity_priors shape {batch['candidates'][key]['candidate_entity_priors'].shape}")
            print(f"Batch candidate_segment_ids shape {batch['candidates'][key]['candidate_segment_ids'].shape}")

            yield(batch)

                    
                    

                        
    
# #Assume 2D array
def pad_to_shape(arr,out_shape,value):

    out = np.ones(out_shape, dtype=arr.dtype)*value
    #1D case
    if(len(arr.shape)==1):
        out[:arr.shape[0]]=arr
    elif(len(arr.shape)==2):
        out[:arr.shape[0],:arr.shape[1]]=arr
    else:
        raise NotImplementedError

    return out

def convert_to_tensor(dict):
    "Can be a nested dict of numpy array, operates in place"
    for key,value in dict.items():
        if(isinstance(value,np.ndarray)):
            dict[key] = torch.from_numpy(value)
        elif(isinstance(value,Dict)):
            convert_to_tensor(value)



            
