from typing import Dict, List

import math
import torch
import tarfile
import json
import torch.nn as nn
import numpy as np
import h5py

from kb.custom_metrics import Average, CategoricalAccuracy, WeightedAverage, ExponentialMovingAverage, F1Metric, MeanReciprocalRank
from kb.custom_util import get_dtype_for_module, extend_attention_mask_for_bert, init_bert_weights

from pytorch_pretrained_bert.modeling import BertForPreTraining, BertLayer, BertLayerNorm, BertConfig, BertEncoder

from kb.custom_span_extractor import SelfAttentiveSpanExtractor
from kb.custom_span_attention_layer import SpanAttentionLayer

def print_shapes(x, prefix='', raise_on_nan=False):
    if isinstance(x, torch.Tensor):
        print(prefix, x.shape)
        if x.dtype == torch.float32 or x.dtype == torch.float16:
            print(x.min(), x.max(), x.mean(), x.std())
        if raise_on_nan and torch.isnan(x).long().sum().item() > 0:
            print("GOT NAN!!")
            raise ValueError
    elif isinstance(x, (list, tuple)):
        for ele in x:
            print_shapes(ele, prefix + '-->')
    elif isinstance(x, dict):
        for k, v in x.items():
            print_shapes(v, prefix + ' ' + k + ':')
    else:
        print("COULDN'T get shape ", type(x))
            
def diagnose_backward_hook(module, m_input, m_output):
    print("------")
    print('Inside ' + module.__class__.__name__ + ' backward')
    print('Inside class:' + module.__class__.__name__)
    print("INPUT:")
    print_shapes(m_input)
    print("OUTPUT:")
    print_shapes(m_output)
    print("=======")

def diagnose_forward_hook(module, m_input, m_output):
    print("------")
    print('Inside ' + module.__class__.__name__ + ' forward')
    print('Inside class:' + module.__class__.__name__)
    print("INPUT:")
    print_shapes(m_input)
    print("OUTPUT:")
    print_shapes(m_output, raise_on_nan=True)
    print("=======")

#Similar to nn.Embedding but allow for more functionalities
class EntityEmbedder():
    pass

#Acts like a standard embedding but with pre-trained entity embeddings and trained POS embeddings
class CustomWordNetAllEmbedding(torch.nn.Module, EntityEmbedder):
    """
    Combines pretrained fixed embeddings with learned POS embeddings.

    Given entity candidate list:
        - get list of unique entity ids
        - look up
        - concat POS embedding
        - linear project to candidate embedding shape
    """
    POS_MAP = {
        '@@PADDING@@': 0,
        'n': 1,
        'v': 2,
        'a': 3,
        'r': 4,
        's': 5,
        # have special POS embeddings for mask / null, so model can learn
        # it's own representation for them
        '@@MASK@@': 6,
        '@@NULL@@': 7,
        '@@UNKNOWN@@': 8
    }

    def __init__(self,
                 embedding_file: str,
                 entity_dim: int,
                 entity_file: str = None,
                 vocab_file: str = None,
                 entity_h5_key: str = 'conve_tucker_infersent_bert',
                 dropout: float = 0.1,
                 pos_embedding_dim: int = 25,
                 include_null_embedding: bool = False):
        """
        pass pos_emedding_dim = None to skip POS embeddings and all the
            entity stuff, using this as a pretrained embedding file
            with feedforward
        """

        super().__init__()

        if pos_embedding_dim is not None:
            # entity_id -> pos abbreviation, e.g.
            # 'cat.n.01' -> 'n'
            # includes special, e.g. '@@PADDING@@' -> '@@PADDING@@'
            entity_to_pos = {}
            with open(entity_file, 'r') as fin:
                for node in fin:
                    node = json.loads(node)
                    if node['type'] == 'synset':
                        entity_to_pos[node['id']] = node['pos']
            for special in ['@@PADDING@@', '@@MASK@@', '@@NULL@@', '@@UNKNOWN@@']:
                entity_to_pos[special] = special
    
            # list of entity ids
            entities = ['@@PADDING@@']
            with open(vocab_file, 'r') as fin:
                for line in fin:
                    entities.append(line.strip())
    
            # the map from entity index id -> pos embedding id,
            # will use for POS embedding lookup
            entity_id_to_pos_index = [
                 self.POS_MAP[entity_to_pos[ent]] for ent in entities
            ]
            self.register_buffer('entity_id_to_pos_index', torch.tensor(entity_id_to_pos_index))
    
            self.pos_embeddings = torch.nn.Embedding(len(entities), pos_embedding_dim)
            init_bert_weights(self.pos_embeddings, 0.02)

            self.use_pos = True
        else:
            self.use_pos = False

        # load the embeddings
        with h5py.File(embedding_file, 'r') as fin:
            entity_embeddings = fin[entity_h5_key][...]
        self.entity_embeddings = torch.nn.Embedding(
                entity_embeddings.shape[0], entity_embeddings.shape[1],
                padding_idx=0
        )
        self.entity_embeddings.weight.data.copy_(torch.tensor(entity_embeddings).contiguous())

        if pos_embedding_dim is not None:
            assert entity_embeddings.shape[0] == len(entities)
            concat_dim = entity_embeddings.shape[1] + pos_embedding_dim
        else:
            concat_dim = entity_embeddings.shape[1]

        self.proj_feed_forward = torch.nn.Linear(concat_dim, entity_dim)
        init_bert_weights(self.proj_feed_forward, 0.02)

        self.dropout = torch.nn.Dropout(dropout)

        self.embedding_dim = entity_dim

        self.include_null_embedding = include_null_embedding
        self.null_embedding=None
        if include_null_embedding:
            # a special embedding for null
            entities = ['@@PADDING@@']
            with open(vocab_file, 'r') as fin:
                for line in fin:
                    entities.append(line.strip())
            self.null_id = entities.index("@@NULL@@")
            self.null_embedding = torch.nn.Parameter(torch.zeros(entity_dim))
            self.null_embedding.data.normal_(mean=0.0, std=0.02)

    def get_output_dim(self):
        return self.embedding_dim

    def get_null_embedding(self):
        return self.null_embedding

    def forward(self, entity_ids):
        """
        entity_ids = (batch_size, num_candidates, num_entities) array of entity
            ids

        returns (batch_size, num_candidates, num_entities, embed_dim)
            with entity embeddings
        """
        # get list of unique entity ids
        unique_ids, unique_ids_to_entity_ids = torch.unique(entity_ids, return_inverse=True)
        # unique_ids[unique_ids_to_entity_ids].reshape(entity_ids.shape)

        # look up (num_unique_embeddings, full_entity_dim)
        unique_entity_embeddings = self.entity_embeddings(unique_ids.contiguous()).contiguous()

        # get POS tags from entity ids (form entity id -> pos id embedding)
        # (num_unique_embeddings)
        if self.use_pos:
            unique_pos_ids = torch.nn.functional.embedding(unique_ids, self.entity_id_to_pos_index).contiguous()
            # (num_unique_embeddings, pos_dim)
            unique_pos_embeddings = self.pos_embeddings(unique_pos_ids).contiguous()
            # concat
            entity_and_pos = torch.cat([unique_entity_embeddings, unique_pos_embeddings], dim=-1)
        else:
            entity_and_pos = unique_entity_embeddings

        # run the ff
        # (num_embeddings, entity_dim)
        projected_entity_and_pos = self.dropout(self.proj_feed_forward(entity_and_pos.contiguous()))

        # replace null if needed
        if self.include_null_embedding:
            null_mask = unique_ids == self.null_id
            projected_entity_and_pos[null_mask] = self.null_embedding

        # remap to candidate embedding shape
        return projected_entity_and_pos[unique_ids_to_entity_ids].contiguous()

class CustomBertPretrainedMetricsLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.nsp_loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.lm_loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)

        self._metrics = {
            "total_loss_ema": ExponentialMovingAverage(alpha=0.5),
            "nsp_loss_ema": ExponentialMovingAverage(alpha=0.5),
            "lm_loss_ema": ExponentialMovingAverage(alpha=0.5),
            "total_loss": Average(),
            "nsp_loss": Average(),
            "lm_loss": Average(),
            "lm_loss_wgt": WeightedAverage(),
            "mrr": MeanReciprocalRank(),
        }
        self._accuracy = CategoricalAccuracy()

    def get_metrics(self, reset: bool = False):
        metrics = {k: v.get_metric(reset) for k, v in self._metrics.items()}
        metrics['nsp_accuracy'] = self._accuracy.get_metric(reset)
        return metrics

    def _compute_loss(self,
                      contextual_embeddings,
                      pooled_output,
                      lm_label_ids,
                      next_sentence_label,
                      update_metrics=True):

        # (batch_size, timesteps, vocab_size), (batch_size, 2)
        prediction_scores, seq_relationship_score = self.pretraining_heads(
                contextual_embeddings, pooled_output
        )

        loss_metrics = []
        if lm_label_ids is not None:
            # Loss
            vocab_size = prediction_scores.shape[-1]
            masked_lm_loss = self.lm_loss_function(
                prediction_scores.view(-1, vocab_size), lm_label_ids["lm_labels"].view(-1)
            )
            masked_lm_loss_item = masked_lm_loss.item()
            loss_metrics.append([["lm_loss_ema", "lm_loss"], masked_lm_loss_item])
            num_lm_predictions = (lm_label_ids["lm_labels"] > 0).long().sum().item()
            self._metrics['lm_loss_wgt'](masked_lm_loss_item, num_lm_predictions)
        else:
            masked_lm_loss = 0.0

        if next_sentence_label is not None:
            next_sentence_loss = self.nsp_loss_function(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            loss_metrics.append([["nsp_loss_ema", "nsp_loss"], next_sentence_loss.item()])
            if update_metrics:
                self._accuracy(
                    seq_relationship_score.detach(), next_sentence_label.view(-1).detach()
                )
        else:
            next_sentence_loss = 0.0

        # update metrics
        if update_metrics:
            total_loss = masked_lm_loss + next_sentence_loss
            for keys, v in [[["total_loss_ema", "total_loss"], total_loss.item()]] + loss_metrics:
                for key in keys:
                    self._metrics[key](v)

        return masked_lm_loss, next_sentence_loss

    def _compute_mrr(self,
                     contextual_embeddings,
                     pooled_output,
                     lm_labels_ids,
                     mask_indicator):
        prediction_scores, seq_relationship_score = self.pretraining_heads(
                contextual_embeddings, pooled_output
        )
        self._metrics['mrr'](prediction_scores, lm_labels_ids, mask_indicator)


#@Model.register("bert_pretrained_masked_lm")
class CustomBertPretrainedMaskedLM(CustomBertPretrainedMetricsLoss):
    """
    So we can evaluate and compute the loss of the pretrained bert model
    """
    def __init__(self,
                 bert_model_name: str,
                 remap_segment_embeddings: int = None):
        super().__init__()

        pretrained_bert = BertForPreTraining.from_pretrained(bert_model_name)
        self.pretraining_heads = pretrained_bert.cls
        self.bert = pretrained_bert

        self.remap_segment_embeddings = remap_segment_embeddings
        if remap_segment_embeddings is not None:
            new_embeddings = self._remap_embeddings(self.bert.bert.embeddings.token_type_embeddings.weight)
            if new_embeddings is not None:
                del self.bert.bert.embeddings.token_type_embeddings
                self.bert.bert.embeddings.token_type_embeddings = new_embeddings

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
            new_embeddings = self._remap_embeddings(state_dict['bert.bert.embeddings.token_type_embeddings.weight'])
            if new_embeddings is not None:
                state_dict['bert.bert.embeddings.token_type_embeddings.weight'] = new_embeddings.weight
        super().load_state_dict(state_dict, strict=strict)

    def forward(self,
                tokens,
                segment_ids,
                lm_label_ids=None,
                next_sentence_label=None,
                **kwargs):
        mask = tokens['tokens'] > 0
        contextual_embeddings, pooled_output = self.bert.bert(
            tokens['tokens'], segment_ids,
            mask, output_all_encoded_layers=False
        )
        if lm_label_ids is not None or next_sentence_label is not None:
            masked_lm_loss, next_sentence_loss = self._compute_loss(
                contextual_embeddings, pooled_output, lm_label_ids, next_sentence_label
            )
            loss = masked_lm_loss + next_sentence_loss
        else:
            loss = 0.0

        if 'mask_indicator' in kwargs:
            self._compute_mrr(contextual_embeddings,
                              pooled_output,
                              lm_label_ids['lm_labels'],
                              kwargs['mask_indicator'])
        return {'loss': loss,
                'contextual_embeddings': contextual_embeddings,
                'pooled_output': pooled_output}

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
class DotAttentionWithPrior(nn.Module):
    def __init__(self,
                 output_feed_forward_hidden_dim: int = 100,
                 weighted_entity_threshold: float = None,
                 null_embedding: torch.Tensor = None,
                 initializer_range: float = 0.02):

        super().__init__()

        # layers for the dot product attention
        self.out_layer_1 = torch.nn.Linear(2, output_feed_forward_hidden_dim)
        self.out_layer_2 = torch.nn.Linear(output_feed_forward_hidden_dim, 1)
        init_bert_weights(self.out_layer_1, initializer_range)
        init_bert_weights(self.out_layer_2, initializer_range)

        self.weighted_entity_threshold = weighted_entity_threshold

        #Used to represent entities with all similarity values under threshold => cannot weighted sum => represent them with null entity
        if null_embedding is not None:
            self.register_buffer('null_embedding', null_embedding)

    def forward(self,
            projected_span_representations,
            candidate_entity_embeddings,
            candidate_entity_prior,
            entity_mask):
        """
        projected_span_representations = (batch_size, num_spans, entity_dim)
        candidate_entity_embeddings = (batch_size, num_spans, num_candidates, entity_embedding_dim)
        candidate_entity_prior = (batch_size, num_spans, num_candidates)
            with prior probability of each candidate entity.
            0 <= candidate_entity_prior <= 1 and candidate_entity_prior.sum(dim=-1) == 1
        entity_mask = (batch_size, num_spans, num_candidates)
            with 0/1 bool of whether it is a valid candidate

        returns dict with:
            linking_scores: linking sccore to each entity in each span
                (batch_size, num_spans, num_candidates)
                masked with -10000 for invalid links
            weighted_entity_embeddings: weighted entity embedding
                (batch_size, num_spans, entity_dim)
        """
        # dot product between span embedding and entity embeddings, scaled
        # by sqrt(dimension) as in Transformer
        # (batch_size, num_spans, num_candidates)
        scores = torch.sum(
            projected_span_representations.unsqueeze(-2) * candidate_entity_embeddings,
            dim=-1
        ) / math.sqrt(candidate_entity_embeddings.shape[-1])

        # compute the final score
        # the prior needs to be input as float32 due to half not supported on
        # cpu.  so need to cast it here.
        dtype = list(self.parameters())[0].dtype

        scores_with_prior = torch.cat(
            [scores.unsqueeze(-1), candidate_entity_prior.unsqueeze(-1).to(dtype)],
            dim=-1
        )

        # (batch_size, num_spans, num_candidates)
        #NOTE: applies MLP
        linking_score = self.out_layer_2(
            torch.nn.functional.relu(self.out_layer_1(scores_with_prior))
        ).squeeze(-1)

        # mask out the invalid candidates
        invalid_candidate_mask = ~entity_mask

        linking_scores = linking_score.masked_fill(invalid_candidate_mask, -10000.0)
        return_dict = {'linking_scores': linking_scores}
        
        #Represent entity by weghted sum of entity embeddings
        weighted_entity_embeddings = self._get_weighted_entity_embeddings(
                linking_scores, candidate_entity_embeddings
        )
        return_dict['weighted_entity_embeddings'] = weighted_entity_embeddings

        return return_dict

    #Do weighted sum of embeddings
    def _get_weighted_entity_embeddings(self, linking_scores, candidate_entity_embeddings):
        """
        Get the entity linking weighted entity embedding

        linking_scores = (batch_size, num_spans, num_candidates)
             with unnormalized scores and masked with very small value
            (-10000) for invalid candidates.
        candidate_entity_embeddings = (batch_size, num_spans, num_candidates, entity_embedding_dim)

        returns weighted_entity_embeddings = (batch_size, num_spans, entity_dim)
        """
        # compute softmax of linking scores
        # if we are using the decode threshold, set all scores less then
        # the threshold to small values so they aren't given any weight
        if self.weighted_entity_threshold is not None:
            below_threshold = linking_scores < self.weighted_entity_threshold
            linking_scores = linking_scores.masked_fill(below_threshold, -10000.0)

        # (batch_size, num_spans, num_candidates)
        normed_linking_scores = torch.nn.functional.softmax(linking_scores, dim=-1)

        # use softmax to get weighted entity embedding from candidates
        # (batch_size, num_spans, entity_dim)
        weighted_entity_embeddings = (normed_linking_scores.unsqueeze(-1) * candidate_entity_embeddings).sum(dim=2)

        # if we have a decode threshold, some spans won't have a single
        # predicted entity above the threshold, need to replace them with
        # NULL
        if self.weighted_entity_threshold is not None:
            num_candidates = linking_scores.shape[-1]
            # (batch_size, num_spans)
            all_below_threshold = (below_threshold == 1).long().sum(dim=-1) == num_candidates
            weighted_entity_embeddings[all_below_threshold] = self.null_embedding

        return weighted_entity_embeddings

class EntityDisambiguator(torch.nn.Module):
    def __init__(self,
                 contextual_embedding_dim: int,
                 entity_embedding_dim: int,
                 entity_embeddings: EntityEmbedder,
                 max_sequence_length: int = 512,
                 span_encoder_config: Dict[str, int] = None,
                 dropout: float = 0.1,
                 output_feed_forward_hidden_dim: int = 100,
                 initializer_range: float = 0.02,
                 weighted_entity_threshold: float = None,
                 null_embedding: torch.Tensor = None):
        """
        Idea: Align the bert and KG vector space by learning a mapping between
            them.
        """
        """
        Extra added:
        contextual_embedding_dim: dimension of token embeddding
        entity_embedding_dim: dimension of entity embeddings
        entity_embeddings: contains embeddings of entity id to vector
        max_sequence_length: length of max sequence => unused
        span_encoder_config: configuraton of span encoder (hidden_size,num_hidden_layers,num_attention_heads,intermediate_size)
        dropout: dropout probability in training 
        output_feed_forward_hidden_dim: #hidden dim of 1 hidden layer MLP to compute similarity between mention and KB entity 
                                        MLP(prior,mention_embedding @ entity_embedding)

        initializer_range: std of normal weight initialization
        weighted_entity_threshold: similarity threshold (computed using MLP (DotAttentionWithPrior)) 
                                    under which an entity is not considered for the weighted sum representation of entity
        null_embedding: enbedding of null entity

        """

        #Not 0???
        #print(entity_embeddings.entity_embeddings.weight[null_entity_id, :])

        #TODO: see if null entity embedding id is used somewhere, else: just provide null entity embedding

        super().__init__()

        #self-attentive span pooling descirbed in paper
        self.span_extractor = SelfAttentiveSpanExtractor(entity_embedding_dim)
        init_bert_weights(self.span_extractor._global_attention,
                          initializer_range)

        self.dropout = torch.nn.Dropout(dropout)

        self.bert_to_kg_projector = torch.nn.Linear(
                contextual_embedding_dim, entity_embedding_dim)
        init_bert_weights(self.bert_to_kg_projector, initializer_range)

        self.projected_span_layer_norm = BertLayerNorm(entity_embedding_dim, eps=1e-5)
        init_bert_weights(self.projected_span_layer_norm, initializer_range)

        self.kg_layer_norm = BertLayerNorm(entity_embedding_dim, eps=1e-5)
        init_bert_weights(self.kg_layer_norm, initializer_range)

        # already pretrained, don't init
        self.entity_embeddings = entity_embeddings
        self.entity_embedding_dim = entity_embedding_dim

        self.dot_attention_with_prior = DotAttentionWithPrior(
                 output_feed_forward_hidden_dim,
                 weighted_entity_threshold,
                 null_embedding,
                 initializer_range
        )
        self.contextual_embedding_dim = contextual_embedding_dim

        if span_encoder_config is None:
            self.span_encoder = None
        else:
            # create BertConfig
            assert len(span_encoder_config) == 4
            config = BertConfig(
                0, # vocab size, not used
                hidden_size=span_encoder_config['hidden_size'],
                num_hidden_layers=span_encoder_config['num_hidden_layers'],
                num_attention_heads=span_encoder_config['num_attention_heads'],
                intermediate_size=span_encoder_config['intermediate_size']
            )
            self.span_encoder = BertEncoder(config)
            init_bert_weights(self.span_encoder, initializer_range)

    def unfreeze(self, mode):
        def _is_in_alignment(n):
            if 'bert_to_kg_projector' in n:
                return True
            elif 'projected_span_layer_norm' in n:
                return True
            elif 'kg_position_embeddings.embedding_projection' in n:
                return True
            elif 'kg_position_embeddings.position_layer_norm' in n:
                return True
            elif 'kg_layer_norm' in n:
                return True
            elif 'span_extractor' in n:
                return True
            else:
                return False

        if mode == 'entity_linking':
            # learning the entity linker
            for n, p in self.named_parameters():
                if _is_in_alignment(n):
                    p.requires_grad_(True)
                elif 'entity_embeddings.weight' in n:
                    p.requires_grad_(False)
                elif 'kg_position_embeddings' in n:
                    p.requires_grad_(False)
                else:
                    p.requires_grad_(True)
        elif mode == 'freeze':
            for p in self.parameters():
                p.requires_grad_(False)
        else:
            for n, p in self.named_parameters():
                if 'entity_embeddings.weight' in n:
                    p.requires_grad_(False)
                else:
                    p.requires_grad_(True)

    def _run_span_encoders(self, x, span_mask):
        # run the transformer
        attention_mask = extend_attention_mask_for_bert(span_mask, get_dtype_for_module(self))
        return self.span_encoder(
            x, attention_mask,
            output_all_encoded_layers=False
        )

    def forward(self,
                contextual_embeddings: torch.Tensor,
                mask: torch.Tensor,
                candidate_spans: torch.Tensor,
                candidate_entities: torch.Tensor,
                candidate_entity_priors: torch.Tensor,
                candidate_segment_ids: torch.Tensor,
                **kwargs
        ):
        """
        contextual_embeddings = (batch_size, timesteps, dim) output
            from language model
        mask = (batch_size, num_times)
        candidate_spans = (batch_size, max_num_spans, 2) with candidate
            mention spans. This gives the start / end location for each
            span such span i in row k has:
                start, end = candidate_spans[k, i, :]
                span_embeddings = contextual_embeddings[k, start:end, :]
            it is padded with -1
        candidate_entities = (batch_size, max_num_spans, max_entity_ids)
            padded with 0
        candidate_entity_prior = (batch_size, max_num_spans, max_entity_ids)
            with prior probability of each candidate entity.
            0 <= candidate_entity_prior <= 1 and candidate_entity_prior.sum(dim=-1) == 1

        Returns:
            linking sccore to each entity in each span
                (batch_size, max_num_spans, max_entity_ids)
            masked with -10000 for invalid links
        """
        # get the candidate entity embeddings
        # (batch_size, num_spans, num_candidates, entity_embedding_dim)
        candidate_entity_embeddings = self.entity_embeddings(candidate_entities)
        candidate_entity_embeddings = self.kg_layer_norm(candidate_entity_embeddings.contiguous())

        # project to entity embedding dim
        # (batch_size, timesteps, entity_dim)
        projected_bert_representations = self.bert_to_kg_projector(contextual_embeddings)

        # compute span representations
        span_mask = (candidate_spans[:, :, 0] > -1).long()
        # (batch_size, num_spans, embedding_dim)
        projected_span_representations = self.span_extractor(
            projected_bert_representations,
            candidate_spans,
            mask,
            span_mask
        )
        projected_span_representations = self.projected_span_layer_norm(projected_span_representations.contiguous())

        # run the span transformer encoders
        #Apply transformer on spans
        if self.span_encoder is not None:
            projected_span_representations = self._run_span_encoders(
                projected_span_representations, span_mask
            )[-1]

        entity_mask = candidate_entities > 0
        #Compute similarity with KB entititeis and do weighted sum
        return_dict = self.dot_attention_with_prior(
                    projected_span_representations,
                    candidate_entity_embeddings,
                    candidate_entity_priors,
                    entity_mask)

        return_dict['projected_span_representations'] = projected_span_representations
        return_dict['projected_bert_representations'] = projected_bert_representations

        return return_dict

class CustomEntityLinkingBase(nn.Module):
    def __init__(self,
                 null_entity_id: int,
                 margin: float = 0.2,
                 decode_threshold: float = 0.0,
                 loss_type: str = 'margin'):

        super().__init__()

        if loss_type == 'margin':
            self.loss = torch.nn.MarginRankingLoss(margin=margin)
            self.decode_threshold = decode_threshold
        elif loss_type == 'softmax':
            self.loss = torch.nn.NLLLoss(ignore_index=-100)
            # set threshold to small value so we just take argmax
            self._log_softmax = torch.nn.LogSoftmax(dim=-1)
            self.decode_threshold = -990
        else:
            raise ValueError("invalid loss type, got {}".format(loss_type))
        self.loss_type = loss_type

        self.null_entity_id = null_entity_id

        self._f1_metric = F1Metric()
        self._f1_metric_untyped = F1Metric()

    def _compute_f1(self, linking_scores, candidate_spans, candidate_entities,
                          gold_entities):
        # will call F1Metric with predicted and gold entities encoded as
        # [(start, end), entity_id]

        predicted_entities = self._decode(
                    linking_scores, candidate_spans, candidate_entities
        )

        # make a mask of valid predictions and non-null entities to select
        # ids and spans
        # (batch_size, num_spans, 1)
        gold_mask = (gold_entities > 0) & (gold_entities != self.null_entity_id)

        valid_gold_entity_spans = candidate_spans[
                torch.cat([gold_mask, gold_mask], dim=-1)
        ].view(-1, 2).tolist()
        valid_gold_entity_id = gold_entities[gold_mask].tolist()

        batch_size, num_spans, _ = linking_scores.shape
        batch_indices = torch.arange(batch_size).unsqueeze(-1).repeat([1, num_spans])[gold_mask.squeeze(-1).cpu()]

        gold_entities_for_f1 = []
        predicted_entities_for_f1 = []
        gold_spans_for_f1 = []
        predicted_spans_for_f1 = []
        for k in range(batch_size):
            gold_entities_for_f1.append([])
            predicted_entities_for_f1.append([])
            gold_spans_for_f1.append([])
            predicted_spans_for_f1.append([])

        for gi, gs, g_batch_index in zip(valid_gold_entity_id,
                              valid_gold_entity_spans,
                              batch_indices.tolist()):
            gold_entities_for_f1[g_batch_index].append((tuple(gs), gi))
            gold_spans_for_f1[g_batch_index].append((tuple(gs), "ENT"))

        for p_batch_index, ps, pi in predicted_entities:
            span = tuple(ps)
            predicted_entities_for_f1[p_batch_index].append((span, pi))
            predicted_spans_for_f1[p_batch_index].append((span, "ENT"))

        self._f1_metric_untyped(predicted_spans_for_f1, gold_spans_for_f1)
        self._f1_metric(predicted_entities_for_f1, gold_entities_for_f1)

    def _decode(self, linking_scores, candidate_spans, candidate_entities):
        # returns [[batch_index1, (start1, end1), eid1],
        #          [batch_index2, (start2, end2), eid2], ...]

        # Note: We assume that linking_scores has already had the mask
        # applied such that invalid candidates have very low score. As a result,
        # we don't need to worry about masking the valid candidate spans
        # here, since their score will be very low, and won't exceed
        # the threshold.

        # find maximum candidate entity score in each valid span
        # (batch_size, num_spans), (batch_size, num_spans)
        max_candidate_score, max_candidate_indices = linking_scores.max(dim=-1)

        # get those above the threshold
        above_threshold_mask = max_candidate_score > self.decode_threshold

        # for entities with score > threshold:
        #       get original candidate span
        #       get original entity id
        # (num_extracted_spans, 2)
        extracted_candidates = candidate_spans[above_threshold_mask]
        # (num_extracted_spans, num_candidates)
        candidate_entities_for_extracted_spans = candidate_entities[above_threshold_mask]
        extracted_indices = max_candidate_indices[above_threshold_mask]
        # the batch number (num_extracted_spans, )
        batch_size, num_spans, _ = linking_scores.shape
        batch_indices = torch.arange(batch_size).unsqueeze(-1).repeat([1, num_spans])[above_threshold_mask.cpu()]

        extracted_entity_ids = []
        for k, ind in enumerate(extracted_indices):
            extracted_entity_ids.append(candidate_entities_for_extracted_spans[k, ind])

        # make tuples [(span start, span end), id], ignoring the null entity
        ret = []
        for start_end, eid, batch_index in zip(
                    extracted_candidates.tolist(),
                    extracted_entity_ids,
                    batch_indices.tolist()
        ):
            entity_id = eid.item()
            if entity_id != self.null_entity_id:
                ret.append((batch_index, tuple(start_end), entity_id))

        return ret

    def get_metrics(self, reset: bool = False):
        precision, recall, f1_measure = self._f1_metric.get_metric(reset)
        precision_span, recall_span, f1_measure_span = self._f1_metric_untyped.get_metric(reset)
        metrics = {
            'el_precision': precision,
            'el_recall': recall,
            'el_f1': f1_measure,
            'span_precision': precision_span,
            'span_recall': recall_span,
            'span_f1': f1_measure_span
        }

        return metrics

    def _compute_loss(self,
                      candidate_entities,
                      candidate_spans,
                      linking_scores,
                      gold_entities):

        if self.loss_type == 'margin':
            return self._compute_margin_loss(
                    candidate_entities, candidate_spans, linking_scores, gold_entities
            )
        elif self.loss_type == 'softmax':
            return self._compute_softmax_loss(
                    candidate_entities, candidate_spans, linking_scores, gold_entities
            )

    def _compute_margin_loss(self,
                             candidate_entities,
                             candidate_spans,
                             linking_scores,
                             gold_entities):

        # compute loss
        # in End-to-End Neural Entity Linking
        # loss = max(0, gamma - score) if gold mention
        # loss = max(0, score) if not gold mention
        #
        # torch.nn.MaxMarginLoss(x1, x2, y) = max(0, -y * (x1 - x2) + gamma)
        #   = max(0, -x1 + x2 + gamma)  y = +1
        #   = max(0, gamma - x1) if x2 == 0, y=+1
        #
        #   = max(0, x1 - gamma) if y==-1, x2=0

        candidate_mask = candidate_entities > 0
        # (num_entities, )
        non_masked_scores = linking_scores[candidate_mask]

        # broadcast gold ids to all candidates
        num_candidates = candidate_mask.shape[-1]
        # (batch_size, num_spans, num_candidates)
        broadcast_gold_entities = gold_entities.repeat(
                    1, 1, num_candidates
        )
        # compute +1 / -1 labels for whether each candidate is gold
        positive_labels = (broadcast_gold_entities == candidate_entities).long()
        negative_labels = (broadcast_gold_entities != candidate_entities).long()
        labels = (positive_labels - negative_labels).to(dtype=get_dtype_for_module(self))
        # finally select the non-masked candidates
        # (num_entities, ) with +1 / -1
        non_masked_labels = labels[candidate_mask]

        loss = self.loss(
                non_masked_scores, torch.zeros_like(non_masked_labels),
                non_masked_labels
        )

        # metrics
        self._compute_f1(linking_scores, candidate_spans,
                         candidate_entities,
                         gold_entities)

        return {'loss': loss}

    def _compute_softmax_loss(self, 
                             candidate_entities, 
                             candidate_spans,
                             linking_scores, 
                             gold_entities):

        # compute log softmax
        # linking scores is already masked with -1000 in invalid locations
        # (batch_size, num_spans, max_num_candidates)
        log_prob = self._log_softmax(linking_scores)

        # get the valid scores.
        # needs to be index into the last time of log_prob, with -100
        # for missing values
        num_candidates = log_prob.shape[-1]
        # (batch_size, num_spans, num_candidates)
        broadcast_gold_entities = gold_entities.repeat(
                    1, 1, num_candidates
        )

        # location of the positive label
        positive_labels = (broadcast_gold_entities == candidate_entities).long()
        # index of the positive class
        targets = positive_labels.argmax(dim=-1)

        # fill in the ignore class
        # DANGER: we assume that each instance has exactly one gold
        # label, and that padded instances are ones for which all
        # candidates are invalid
        # (batch_size, num_spans)
        invalid_prediction_mask = (
            candidate_entities != 0
        ).long().sum(dim=-1) == 0
        targets[invalid_prediction_mask] = -100

        loss = self.loss(log_prob.view(-1, num_candidates), targets.view(-1, ))

        # metrics
        self._compute_f1(linking_scores, candidate_spans,
                         candidate_entities,
                         gold_entities)

        return {'loss': loss}

#NOTE: previously inherited from EntityLinkingBase
#Does equation 1,2,3,4,5
class CustomEntityLinkingWithCandidateMentions(CustomEntityLinkingBase):
    def __init__(self,
                 null_entity_id: int,
                 entity_embedding: EntityEmbedder = None,
                 contextual_embedding_dim: int = None,
                 span_encoder_config: Dict[str, int] = None,
                 margin: float = 0.2,
                 decode_threshold: float = 0.0,
                 loss_type: str = 'margin',
                 max_sequence_length: int = 512,
                 dropout: float = 0.1,
                 output_feed_forward_hidden_dim: int = 100,
                 initializer_range: float = 0.02):
        
        #NOTE: see where it depends on entity EntityLinkingBase and remove not needed functions
        super().__init__(null_entity_id=null_entity_id,
                         margin=margin,
                         decode_threshold=decode_threshold,
                         loss_type=loss_type)
        
        if hasattr(entity_embedding, 'get_null_embedding'):
            null_embedding = entity_embedding.get_null_embedding()
        else:
            null_embedding = entity_embedding.weight[null_entity_id, :]

        #NOTE: this model holds no parameter: all the work is done by EntityDisambiguator => can remove this class??

        entity_embedding_dim = entity_embedding.embedding_dim
        
        if type(entity_embedding) == CustomWordNetAllEmbedding:
            for param in entity_embedding.parameters():
                param.requires_grad_(False)
        
        if loss_type == 'margin':
            weighted_entity_threshold = decode_threshold
        else:
            weighted_entity_threshold = None

        self.disambiguator = EntityDisambiguator(
                 contextual_embedding_dim,
                 entity_embedding_dim=entity_embedding_dim,
                 entity_embeddings=entity_embedding,
                 max_sequence_length=max_sequence_length,
                 span_encoder_config=span_encoder_config,
                 dropout=dropout,
                 output_feed_forward_hidden_dim=output_feed_forward_hidden_dim,
                 initializer_range=initializer_range,
                 weighted_entity_threshold=weighted_entity_threshold,
                 null_embedding=null_embedding)


    def get_metrics(self, reset: bool = False):
        metrics = super().get_metrics(reset)
        return metrics

    def unfreeze(self, mode):
        # don't hold an parameters directly, so do nothing
        self.disambiguator.unfreeze(mode)

    def forward(self,
                contextual_embeddings: torch.Tensor,
                tokens_mask: torch.Tensor,
                candidate_spans: torch.Tensor,
                candidate_entities: torch.Tensor,
                candidate_entity_priors: torch.Tensor,
                candidate_segment_ids: torch.Tensor,
                **kwargs):

        disambiguator_output = self.disambiguator(
            contextual_embeddings=contextual_embeddings,
            mask=tokens_mask,
            candidate_spans=candidate_spans,
            candidate_entities=candidate_entities['ids'],
            candidate_entity_priors=candidate_entity_priors,
            candidate_segment_ids=candidate_segment_ids,
            **kwargs
        )

        linking_scores = disambiguator_output['linking_scores']

        return_dict = disambiguator_output

        if 'gold_entities' in kwargs:
            loss_dict = self._compute_loss(
                    candidate_entities['ids'],
                    candidate_spans,
                    linking_scores,
                    kwargs['gold_entities']['ids']
            )
            return_dict.update(loss_dict)

        return return_dict

#@Model.register("soldered_kg")
class CustomSolderedKG(nn.Module):
    def __init__(self,
                 entity_linker: nn.Module,
                 span_attention_config: Dict[str, int], 
                 should_init_kg_to_bert_inverse: bool = True,
                 freeze: bool = False):

        #Do not care about vocab     
        super().__init__()

        #span_attention_config is used to create SpanAttentionLayer
        
        #EntityLinkingWithCandidateMentions
        self.entity_linker = entity_linker
        #200
        self.entity_embedding_dim = self.entity_linker.disambiguator.entity_embedding_dim
        #768
        self.contextual_embedding_dim = self.entity_linker.disambiguator.contextual_embedding_dim

        self.weighted_entity_layer_norm = BertLayerNorm(self.entity_embedding_dim, eps=1e-5)
        init_bert_weights(self.weighted_entity_layer_norm, 0.02)

        self.dropout = torch.nn.Dropout(0.1)

        # the span attention layers
        assert len(span_attention_config) == 4
        config = BertConfig(
            0, # vocab size, not used
            hidden_size=span_attention_config['hidden_size'],
            num_hidden_layers=span_attention_config['num_hidden_layers'],
            num_attention_heads=span_attention_config['num_attention_heads'],
            intermediate_size=span_attention_config['intermediate_size']
        )
        self.span_attention_layer = SpanAttentionLayer(config)
        # already init inside span attention layer

        # for the output!
        self.output_layer_norm = BertLayerNorm(self.contextual_embedding_dim, eps=1e-5)

        #Project back from kg embedding to bert embeddding
        self.kg_to_bert_projection = torch.nn.Linear(
                self.entity_embedding_dim, self.contextual_embedding_dim
        )

        self.should_init_kg_to_bert_inverse = should_init_kg_to_bert_inverse
        #Project back from kg embedding to bert embeddding is initialized with 
        # inverse of projection from bert embeddding to kg embedding
        self._init_kg_to_bert_projection()

        self._freeze_all = freeze

    def _init_kg_to_bert_projection(self):
        if not self.should_init_kg_to_bert_inverse:
            return

        # the output projection we initialize from the bert to kg, after
        # we load the weights
        # projection as the pseudo-inverse
        # w = (entity_dim, contextual_embedding_dim)
        w = self.entity_linker.disambiguator.bert_to_kg_projector.weight.data.detach().numpy()
        w_pseudo_inv = np.dot(np.linalg.inv(np.dot(w.T, w)), w.T)
        b = self.entity_linker.disambiguator.bert_to_kg_projector.bias.data.detach().numpy()
        b_pseudo_inv = np.dot(w_pseudo_inv, b)
        self.kg_to_bert_projection.weight.data.copy_(torch.tensor(w_pseudo_inv))
        self.kg_to_bert_projection.bias.data.copy_(torch.tensor(b_pseudo_inv))

    def get_metrics(self, reset=False):
        return self.entity_linker.get_metrics(reset)

    def unfreeze(self, mode):
        if self._freeze_all:
            for p in self.parameters():
                p.requires_grad_(False)
            self.entity_linker.unfreeze('freeze')
            return

        if mode == 'entity_linking':
            # training the entity linker, fix parameters here
            for p in self.parameters():
                p.requires_grad_(False)
        else:
            for p in self.parameters():
                p.requires_grad_(True)

        # unfreeze will get called after loading weights in the case where
        # we pass a model archive to KnowBert, so re-init here
        self._init_kg_to_bert_projection()

        self.entity_linker.unfreeze(mode)

    def forward(self,
                contextual_embeddings: torch.Tensor,
                tokens_mask: torch.Tensor,
                candidate_spans: torch.Tensor,
                candidate_entities: torch.Tensor,
                candidate_entity_priors: torch.Tensor,
                candidate_segment_ids: torch.Tensor,
                **kwargs):

        linker_output = self.entity_linker(
                contextual_embeddings, tokens_mask,
                candidate_spans, candidate_entities, candidate_entity_priors,
                candidate_segment_ids, **kwargs)

        # update the span representations with the entity embeddings
        span_representations = linker_output['projected_span_representations']
        weighted_entity_embeddings = linker_output['weighted_entity_embeddings']
        #Sum of entity span reprensentations and weighted entity embeddings (equation 6)
        spans_with_entities = self.weighted_entity_layer_norm(
                (span_representations +
                self.dropout(weighted_entity_embeddings)).contiguous()
        )

        # self attention between bert and spans_with_entities (done in projected dimension )
        # span_attention_output = MLP(MultiHeadAttn(H_i^proj,S'e,S'e))
        entity_mask = candidate_spans[:, :, 0] > -1
        span_attention_output = self.span_attention_layer(
                linker_output['projected_bert_representations'],
                spans_with_entities,
                entity_mask
        )
        projected_bert_representations_with_entities = span_attention_output['output']
        entity_attention_probs = span_attention_output["attention_probs"]

        # Project back to full bert dimension 
        bert_representations_with_entities = self.kg_to_bert_projection(
                projected_bert_representations_with_entities
        )
        #Equation 7 of paper
        new_contextual_embeddings = self.output_layer_norm(
                (contextual_embeddings + self.dropout(bert_representations_with_entities)).contiguous()
        )

        return_dict = {'entity_attention_probs': entity_attention_probs,
                       'contextual_embeddings': new_contextual_embeddings,
                       'linking_scores': linker_output['linking_scores']}
        if 'loss' in linker_output:
            return_dict['loss'] = linker_output['loss']

        return return_dict

#@Model.register("knowbert")
#Removed vocab
class CustomKnowBert(CustomBertPretrainedMetricsLoss):
    def __init__(self,
                 soldered_kgs: Dict[str, nn.Module],
                 soldered_layers: Dict[str, int],
                 bert_model_name: str,
                 mode: str = None,
                 state_dict_file: str = None,
                 strict_load_archive: bool = True,
                 debug_cuda: bool = False,
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

        if debug_cuda:
            for m in self.modules():
                m.register_forward_hook(diagnose_forward_hook)
                m.register_backward_hook(diagnose_backward_hook)

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
    def get_metrics(self, reset: bool = False):
        metrics = super().get_metrics(reset)

        for key in self.soldered_kgs.keys():
            module = getattr(self, key + "_soldered_kg")
            module_metrics = module.get_metrics(reset)
            for metric_key, val in module_metrics.items():
                metrics[key + '_' + metric_key] = val

        return metrics

    def forward(self, tokens=None, segment_ids=None, candidates=None,
                lm_label_ids=None, next_sentence_label=None, **kwargs):

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

                contextual_embeddings = kg_output['contextual_embeddings']

                #Add the output of the soldered kg to the output of KnowBert
                output[soldered_kg_key] = {}
                for key in kg_output.keys():
                    if key != 'loss' and key != 'contextual_embeddings':
                        output[soldered_kg_key][key] = kg_output[key]

        # get the pooled CLS output
        pooled_output = self.pooler(contextual_embeddings)

        if lm_label_ids is not None or next_sentence_label is not None:
            # compute MLM and NSP loss
            masked_lm_loss, next_sentence_loss = self._compute_loss(
                    contextual_embeddings,
                    pooled_output,
                    lm_label_ids,
                    next_sentence_label)

            loss = loss + masked_lm_loss + next_sentence_loss

        if 'mask_indicator' in kwargs:
            self._compute_mrr(contextual_embeddings,
                              pooled_output,
                              lm_label_ids['lm_labels'],
                              kwargs['mask_indicator'])

        output['loss'] = loss
        output['pooled_output'] = pooled_output
        output['contextual_embeddings'] = contextual_embeddings

        return output