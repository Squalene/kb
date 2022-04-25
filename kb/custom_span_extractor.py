import torch
import torch.nn as nn
from overrides import overrides
from typing import Dict, Optional, Tuple, Union

from kb import custom_util as util

class SpanExtractor(nn.Module):
    """
    Many NLP models deal with representations of spans inside a sentence.
    SpanExtractors define methods for extracting and representing spans
    from a sentence.

    SpanExtractors take a sequence tensor of shape (batch_size, timesteps, embedding_dim)
    and indices of shape (batch_size, num_spans, 2) and return a tensor of
    shape (batch_size, num_spans, ...), forming some representation of the
    spans.
    """
    @overrides
    def forward(self, # pylint: disable=arguments-differ
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None):
        """
        Given a sequence tensor, extract spans and return representations of
        them. Span representation can be computed in many different ways,
        such as concatenation of the start and end spans, attention over the
        vectors contained inside the span, etc.

        Parameters
        ----------
        sequence_tensor : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, sequence_length, embedding_size)
            representing an embedded sequence of words.
        span_indices : ``torch.LongTensor``, required.
            A tensor of shape ``(batch_size, num_spans, 2)``, where the last
            dimension represents the inclusive start and end indices of the
            span to be extracted from the ``sequence_tensor``.
        sequence_mask : ``torch.LongTensor``, optional (default = ``None``).
            A tensor of shape (batch_size, sequence_length) representing padded
            elements of the sequence.
        span_indices_mask : ``torch.LongTensor``, optional (default = ``None``).
            A tensor of shape (batch_size, num_spans) representing the valid
            spans in the ``indices`` tensor. This mask is optional because
            sometimes it's easier to worry about masking after calling this
            function, rather than passing a mask directly.

        Returns
        -------
        A tensor of shape ``(batch_size, num_spans, embedded_span_size)``,
        where ``embedded_span_size`` depends on the way spans are represented.
        """
        raise NotImplementedError

    def get_input_dim(self) -> int:
        """
        Returns the expected final dimension of the ``sequence_tensor``.
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        Returns the expected final dimension of the returned span representation.
        """
        raise NotImplementedError

class SelfAttentiveSpanExtractor(SpanExtractor):
    """
    Computes span representations by generating an unnormalized attention score for each
    word in the document. Spans representations are computed with respect to these
    scores by normalising the attention scores for words inside the span.

    Given these attention distributions over every span, this module weights the
    corresponding vector representations of the words in the span by this distribution,
    returning a weighted representation of each span.

    Parameters
    ----------
    input_dim : ``int``, required.
        The final dimension of the ``sequence_tensor``.

    Returns
    -------
    attended_text_embeddings : ``torch.FloatTensor``.
        A tensor of shape (batch_size, num_spans, input_dim), which each span representation
        is formed by locally normalising a global attention over the sequence. The only way
        in which the attention distribution differs over different spans is in the set of words
        over which they are normalized.
    """
    def __init__(self,
                 input_dim: int) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._global_attention = torch.nn.Linear(input_dim, 1)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    @overrides
    def forward(self,
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None) -> torch.FloatTensor:

        dtype = sequence_tensor.dtype

        # both of shape (batch_size, num_spans, 1)
        span_starts, span_ends = span_indices.split(1, dim=-1)

        # shape (batch_size, num_spans, 1)
        # These span widths are off by 1, because the span ends are `inclusive`.
        span_widths = span_ends - span_starts

        # We need to know the maximum span width so we can
        # generate indices to extract the spans from the sequence tensor.
        # These indices will then get masked below, such that if the length
        # of a given span is smaller than the max, the rest of the values
        # are masked.
        max_batch_span_width = span_widths.max().item() + 1

        # shape (batch_size, sequence_length, 1)
        global_attention_logits = self._global_attention(sequence_tensor)

        # Shape: (1, 1, max_batch_span_width) eg: [[[0,1]]]
        max_span_range_indices = torch.arange(max_batch_span_width, device = sequence_tensor.device).view(1, 1, -1)

        # Shape: (batch_size, num_spans, max_batch_span_width)
        # This is a broadcasted comparison - for each span we are considering,
        # we are creating a range vector of size max_span_width, but masking values
        # which are greater than the actual length of the span.
        #
        # We're using <= here (and for the mask below) because the span ends are
        # inclusive, so we want to include indices which are equal to span_widths rather
        # than using it as a non-inclusive upper bound.
        span_mask = (max_span_range_indices <= span_widths).to(dtype)
        raw_span_indices = span_ends - max_span_range_indices
        # We also don't want to include span indices which are less than zero,
        # which happens because some spans near the beginning of the sequence
        # have an end index < max_batch_span_width, so we add this to the mask here.
        span_mask = span_mask * (raw_span_indices >= 0).to(dtype)
        span_indices = torch.nn.functional.relu(raw_span_indices.to(dtype)).long()

        # Shape: (batch_size * num_spans * max_batch_span_width)
        flat_span_indices = util.flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = util.batched_index_select(sequence_tensor, span_indices, flat_span_indices)

        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_logits = util.batched_index_select(global_attention_logits,
                                                          span_indices,
                                                          flat_span_indices).squeeze(-1)
        # print(span_attention_logits.shape)
        # print(span_mask.shape)

        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_weights = util.masked_softmax(span_attention_logits, span_mask,
                                                     memory_efficient=True,
                                                     mask_fill_value=-1000)

        # Do a weighted sum of the embedded spans with
        # respect to the normalised attention distributions.
        # Shape: (batch_size, num_spans, embedding_dim)

        #Do weighted sum of embeddings corresponding to same span
        attended_text_embeddings = (span_embeddings*span_attention_weights.unsqueeze(dim=-1)).sum(dim=-2)

        if span_indices_mask is not None:
            # Above we were masking the widths of spans with respect to the max
            # span width in the batch. Here we are masking the spans which were
            # originally passed in as padding.
            return attended_text_embeddings * span_indices_mask.unsqueeze(-1).to(dtype)

        return attended_text_embeddings