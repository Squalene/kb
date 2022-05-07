from typing import Dict, List

import math
import torch
import torch.nn as nn
import numpy as np

from kb.custom_util import get_dtype_for_module, extend_attention_mask_for_bert, init_bert_weights
from kb.custom_knowledge import CustomWordNetAllEmbedding, EntityEmbedder
from kb.custom_metrics import F1Metric

from pytorch_pretrained_bert.modeling import BertForPreTraining, BertLayer, BertLayerNorm, BertConfig, BertEncoder

# KnowBert:
#   Combines bert with one or more SolderedKG
#
#   each SolderedKG is inserted at a particular level, given by an index,
#   such that we run Bert to the index, then the SolderedKG, then the rest
#   of bert.  Indices such that index 0 means run the first contextual layer,
#   then add KG, and index 11 means run to the top of Bert, then the KG
#   (for bert base with 12 layers).
#

#Do MLP(prior,span_representation @ entity_embedding) and generates weighted entity embedding from the obtained similarities

class CustomPreTrainedKnowbertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """
    def __init__(self):
        super().__init__()
        #self.config = config
    pass


    @classmethod
    def from_pretrained(cls, dir_path, state_dict=None,remapping_dict=None):

        pass


class CustomKnowBertForTraining(nn.Module):
    def __init__(self,knowbert_model):

        #The model should be initialized with the corresponding training mode
        #
        self.knowbert_model = knowbert_model

    def forward(self, tokens=None, segment_ids=None, candidates=None,
                lm_label_ids=None, next_sentence_label=None, **kwargs):

        output = self.knowbert_model(tokens, segment_ids, candidates,kwargs)

        #Can also do something about the loss obtained for the respective KG_Soldered
        loss = output['loss']
        contextual_embeddings = output['contextual_embeddings']
        pooled_output = output['pooled_output']
    
        if lm_label_ids is not None or next_sentence_label is not None:
                # compute MLM and NSP loss
                masked_lm_loss, next_sentence_loss = self._compute_loss(
                        contextual_embeddings,
                        pooled_output,
                        lm_label_ids,
                        next_sentence_label)

                loss = loss + masked_lm_loss + next_sentence_loss

        # if 'mask_indicator' in kwargs:
        #     self._compute_mrr(contextual_embeddings,
        #                     pooled_output,
        #                     lm_label_ids['lm_labels'],
        #                     kwargs['mask_indicator'])


        return loss


class CustomKnowBert(CustomPreTrainedKnowbertModel):
    def __init__(self,
                 soldered_kgs: Dict[str, nn.Module],
                 soldered_layers: Dict[str, int],
                 bert_model_name: str,
                 mode: str = None,
                 state_dict_file: str = None,
                 strict_load_archive: bool = True,
                 remap_segment_embeddings: int = None,
                 state_dict_map:Dict[str,str] = None):

        '''
        state_dict_map maps from string name in state_dict to new string name to fit
        '''
        super().__init__()

        self.remap_segment_embeddings = remap_segment_embeddings

        # get the LM + NSP parameters from BERT
        pretrained_bert = BertForPreTraining.from_pretrained(bert_model_name)
        self.pretrained_bert = pretrained_bert
        self.pretraining_heads = pretrained_bert.cls
        self.pooler = pretrained_bert.bert.pooler

        #NOTE: add the soldered layers as layer of this module
        self.soldered_kgs = soldered_kgs
        for key, skg in soldered_kgs.items():
            self.add_module(key + "_soldered_kg", skg)

        # list of (layer_number, soldered key) sorted in ascending order
        #eg: [(9, 'wordnet')]
        self.layer_to_soldered_kg = sorted(
                [(layer, key) for key, layer in soldered_layers.items()]
        )

        # the last layer
        # eg: 12
        num_bert_layers = len(self.pretrained_bert.bert.encoder.layer)

        # the first element of the list is the index
        #eg:[(9, 'wordnet'), [11, None]]
        self.layer_to_soldered_kg.append([num_bert_layers - 1, None])

        #Load the model's weights
        if state_dict_file is not None:
            if(torch.cuda.is_available()):
                state_dict = torch.load(state_dict_file)
            else:
                state_dict = torch.load(state_dict_file,map_location='cpu')
            
            #Does remapping
            if(state_dict_map!=None):
                state_dict = state_dict.copy()
                metadata = getattr(state_dict, '_metadata', None)
                if metadata is not None:
                    state_dict._metadata = metadata
                for old_key,new_key in state_dict_map.items():
                    state_dict[new_key] = state_dict.pop(old_key)
                        
            self.load_state_dict(state_dict, strict=strict_load_archive)

        #Token type embeddigns in bert <=> segment embeddings => originally: to which out of 2 sentence the token belongs to
        #Remapping allows to have more than 2 segment embeddings type
        if remap_segment_embeddings is not None:
            # will redefine the segment embeddings
            new_embeddings = self._remap_embeddings(self.pretrained_bert.bert.embeddings.token_type_embeddings.weight)
            if new_embeddings is not None:
                del self.pretrained_bert.bert.embeddings.token_type_embeddings
                self.pretrained_bert.bert.embeddings.token_type_embeddings = new_embeddings

        #entity_linking mode indicates that we are only training the entity linker and freezing the other parameters
        assert mode in (None, 'entity_linking')
        self.mode = mode
        #if mode = entity_linking,freeze all and then only unfreeze the SolderedKB layer, else, unfreeze the entire model
        self.unfreeze()

    def _remap_embeddings(self, token_type_embeddings):
        embed_dim = token_type_embeddings.shape[1]
        if list(token_type_embeddings.shape) == [self.remap_segment_embeddings, embed_dim]:
            # already remapped!
            return None
        new_embeddings = torch.nn.Embedding(self.remap_segment_embeddings, embed_dim)
        new_embeddings.weight.data.copy_(token_type_embeddings.data[0, :])
        return new_embeddings

    def load_state_dict(self, state_dict, strict=True):
        if self.remap_segment_embeddings:
            # hack the embeddings!
            new_embeddings = self._remap_embeddings(state_dict['pretrained_bert.bert.embeddings.token_type_embeddings.weight'])
            if new_embeddings is not None:
                state_dict['pretrained_bert.bert.embeddings.token_type_embeddings.weight'] = new_embeddings.weight
        super().load_state_dict(state_dict, strict=strict)

    def unfreeze(self):
        if self.mode == 'entity_linking':
            # all parameters in BERT are fixed, just training the linker
            # linker specific params set below when calling soldered_kg.unfreeze
            for p in self.parameters():
                p.requires_grad_(False)
        else:
            for p in self.parameters():
                p.requires_grad_(True)

        for key in self.soldered_kgs.keys():
            module = getattr(self, key + "_soldered_kg")
            module.unfreeze(self.mode)

    #NOTE: return for each soldered_kg, the loss it can compute??

    def forward(self, tokens=None, segment_ids=None, candidates=None, **kwargs):

        #Receives: 
        #tokens['tokens']: Tensor of tokens indices (used to idx an embedding) => because a batch contains multiple
         #sentences with varying # of tokens, all tokens tensors are padded with zeros 
         #shape: (batch_size (#sentences), max_seq_len)

        #segment_ids: Tenso of segments_ids for each token (0 for first segment and 1 for second), can be used for NSP
         #shape: (batch_size,max_seq_len)

        #candidates, for each SolderedKB contains:

         #candidates['wordnet']['candidate_entity_priors']: hape:(batch_size, max # detected entities, max # KB candidate entities)
          #Correctness probabilities estimated by the entity extractor (sum to 1 (or 0 if padding) on axis 2)
          #Adds 0 padding to axis 1 when there is less detected entities in the sentence than in the max sentence
          #Adds 0 padding to axis 2 when there is less detected KB entities for an entity in the sentence than in the max candidate KB entities entity

         #candidates['wordnet']['ids']: shape: (batch_size, max # detected entities, max # KB candidate entities)
          #Ids of the KB candidate entities + 0 padding on axis 1 or 2 if necessary

         #candidates['wordnet']['candidate_spans']: shape: (batch_size, max # detected entities, 2)
          #Spans of which sequence of tokens correspond to an entity in the sentence, eg: [1,2] for Michael Jackson (both bounds are included)
          #Padding with [-1,-1] when no more detected entities

         #candidates['wordnet']['candidate_segment_ids']: shape: (batch_size, max # detected entities)
          #For each sentence entity, indicate to which segment ids it corresponds to
        
        #lm_label_ids: suppose it is the lables of the masked token?

        #next_sentence_label: suppose it is the labels of the next sentence for NSP

        assert candidates.keys() == self.soldered_kgs.keys()

        #Mask correspond to token = -1
        mask = tokens['tokens'] > 0
        #0 for non masked tokens and -10000.0 for masked tokens
        attention_mask = extend_attention_mask_for_bert(mask, get_dtype_for_module(self))

        #Token embeddings extracted from their indices
        contextual_embeddings = self.pretrained_bert.bert.embeddings(tokens['tokens'], segment_ids)

        output = {}
        start_layer_index = 0
        loss = 0.0

        #NOTE: dictionnary that for each soldered kg layer contains a list of the ids of the correct entity in text => can be usd to train the entity linker
        gold_entities = kwargs.pop('gold_entities', None)

        for layer_num, soldered_kg_key in self.layer_to_soldered_kg:
            end_layer_index = layer_num + 1
            if end_layer_index > start_layer_index:
                # run bert layer in between previous and current SolderedKG 
                for layer in self.pretrained_bert.bert.encoder.layer[
                                start_layer_index:end_layer_index]:
                    contextual_embeddings = layer(contextual_embeddings, attention_mask)
            start_layer_index = end_layer_index

            # run the SolderedKG component
            if soldered_kg_key is not None:
                #Get soldered_kg module
                soldered_kg = getattr(self, soldered_kg_key + "_soldered_kg")
                #Gives for this soldered kg, the span, prior, ids and the segment_ids of the detected entities
                soldered_kwargs = candidates[soldered_kg_key]
                soldered_kwargs.update(kwargs)
                if gold_entities is not None and soldered_kg_key in gold_entities:
                    soldered_kwargs['gold_entities'] = gold_entities[soldered_kg_key]
                kg_output = soldered_kg(
                        contextual_embeddings=contextual_embeddings,
                        tokens_mask=mask,
                        **soldered_kwargs)

                #Add the soldered KG loss (entity linker loss) to the sum of loss of other soldered KG
                if 'loss' in kg_output:
                    loss = loss + kg_output['loss']
                    output[soldered_kg_key+'loss'] = kg_output['loss']

                contextual_embeddings = kg_output['contextual_embeddings']

                #Add the output of the soldered kg to the output of KnowBert
                output[soldered_kg_key] = {}
                for key in kg_output.keys():
                    if key != 'loss' and key != 'contextual_embeddings':
                        output[soldered_kg_key][key] = kg_output[key]

        # get the pooled CLS output
        pooled_output = self.pooler(contextual_embeddings)

        output['loss'] = loss
        output['pooled_output'] = pooled_output
        output['contextual_embeddings'] = contextual_embeddings

        return output