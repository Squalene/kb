
import torch
import torch.nn as nn
from kb.custom_metrics import Average, CategoricalAccuracy, WeightedAverage, ExponentialMovingAverage, MeanReciprocalRank
from pytorch_pretrained_bert.modeling import BertForPreTraining

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