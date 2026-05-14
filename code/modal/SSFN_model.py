import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from .table import TableEncoder
from .matching_layer import MatchingLayer
import torch.nn.functional as F

class SSFNModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.table_encoder = TableEncoder(config)
        self.inference = InferenceLayer(config)
        self.matching = MatchingLayer(config)
        self.pos_embed_dim = config.pos_embed_dim if hasattr(config, 'pos_embed_dim') else 64
        self.num_pos_tags = config.num_pos_tags if hasattr(config, 'num_pos_tags') else 50
        self.pos_quant_embedding = nn.Embedding(self.num_pos_tags * self.num_pos_tags, self.pos_embed_dim) 
    def forward(self, input_ids, attention_mask, ids, text, adj_pack,
                start_label_masks, end_label_masks,
                length, pos_ids,  
                t_start_labels=None, t_end_labels=None,
                o_start_labels=None, o_end_labels=None,
                table_labels_S=None, table_labels_E=None,
                polarity_labels=None, pairs_true=None,
                ):

        seq = self.bert(input_ids, attention_mask)[0]      
        cls_repr = seq[:, 0, :]    

        if pos_ids is not None:
            batch_size, seq_len = pos_ids.shape
            pos_i = pos_ids.unsqueeze(2)   # [B, L, 1]
            pos_j = pos_ids.unsqueeze(1)   # [B, 1, L]
            pos_pair_ids = pos_i * self.num_pos_tags + pos_j   # [B, L, L]
            pos_quant_matrix = self.pos_quant_embedding(pos_pair_ids)
        else:
            pos_quant_matrix = None

        table = self.table_encoder(seq, attention_mask, pos_quant_matrix=pos_quant_matrix, cls_repr=cls_repr)

        gate_stats = getattr(self.table_encoder, "last_gate_stats", None)
        output = self.inference(table, attention_mask, table_labels_S, table_labels_E)
        output['ids'] = ids
        output['fused_table'] = table  

        if gate_stats is not None:
            output['gate_mean'] = gate_stats["mean"]
            output['gate_std'] = gate_stats["std"]
            output['gate_pos'] = gate_stats["pos_ratio"]

        output = self.matching(output, table, pairs_true)
        return output

class InferenceLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cls_linear_S = nn.Linear(768, 1)
        self.cls_linear_E = nn.Linear(768, 1)

    def span_pruning(self, pred, z, attention_mask):
        mask_length = attention_mask.sum(dim=1) - 2
        length = ((attention_mask.sum(dim=1) - 2) * z).long()
        length[length < 5] = 5
        max_length = mask_length ** 2
        for i in range(length.shape[0]):
            if length[i] > max_length[i]:
                length[i] = max_length[i]
        batch_size = attention_mask.shape[0]
        pred_sort, _ = pred.view(batch_size, -1).sort(descending=True)
        batchs = torch.arange(batch_size).to('cuda')
        topkth = pred_sort[batchs, length - 1].unsqueeze(1)
        return pred >= (topkth.view(batch_size, 1, 1))

    def forward(self, table, attention_mask, table_labels_S, table_labels_E):
        outputs = {}
        logits_S = torch.squeeze(self.cls_linear_S(table), 3)
        logits_E = torch.squeeze(self.cls_linear_E(table), 3)
        loss_func = nn.BCEWithLogitsLoss(weight=(table_labels_S>=0))
        outputs['table_loss_S'] = loss_func(logits_S, table_labels_S.float())
        outputs['table_loss_E'] = loss_func(logits_E, table_labels_E.float())

        S_pred = torch.sigmoid(logits_S) * (table_labels_S>=0)
        E_pred = torch.sigmoid(logits_E) * (table_labels_S>=0)

        if self.config.span_pruning != 0:
            table_predict_S = self.span_pruning(S_pred, self.config.span_pruning, attention_mask)
            table_predict_E = self.span_pruning(E_pred, self.config.span_pruning, attention_mask) 
        else:
            table_predict_S = S_pred>0.5
            table_predict_E = E_pred>0.5
        outputs['table_predict_S'] = table_predict_S
        outputs['table_predict_E'] = table_predict_E
        outputs['table_labels_S'] = table_labels_S
        outputs['table_labels_E'] = table_labels_E
        return outputs     

