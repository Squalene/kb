import torch
import json
from kb.custom_util import init_bert_weights
import h5py

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

            #Reshape to simulate embedding of size 1
            unique_pos_ids = torch.nn.functional.embedding(unique_ids, self.entity_id_to_pos_index.reshape(-1,1)).flatten().contiguous()
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