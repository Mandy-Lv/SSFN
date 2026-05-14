from .seq2mat import *
from .table_encoder.resnet import ResNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

class TableEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.seq2mat == 'tensor':
            self.seq2mat = TensorSeq2Mat(config)
        elif config.seq2mat == 'tensorcontext':
            self.seq2mat = TensorcontextSeq2Mat(config)
        elif config.seq2mat == 'context':
            self.seq2mat = ContextSeq2Mat(config)
        else:
            self.seq2mat = Seq2Mat(config)
        if config.table_encoder != 'none':
            self.layer = nn.ModuleList([ResNet(config) for _ in range(config.num_table_layers)])
        self.pos_embed_dim = config.pos_embed_dim if hasattr(config, 'pos_embed_dim') else 64
        self.fusion_type = config.fusion_type if hasattr(config, 'fusion_type') else 'add'
        if self.fusion_type == 'add':
            self.pos_projection = nn.Linear(config.hidden_size + self.pos_embed_dim, config.hidden_size)
        elif self.fusion_type == 'mul':
            self.pos_projection = nn.Linear(self.pos_embed_dim, config.hidden_size)
        self.last_gate_stats = None
        self.global_dim = getattr(config, "global_dim", 768)
        self.proj_cls = nn.Linear(config.hidden_size, self.global_dim)
        table_feat_dim = config.hidden_size
        self.proj_wave = nn.Linear(table_feat_dim, self.global_dim)
        self.proj_global = nn.Linear(self.global_dim * 2, self.global_dim)
        self.proj_token = nn.Linear(config.hidden_size, self.global_dim)
        self.delta_mlp = nn.Sequential(
            nn.Linear(self.global_dim * 3, table_feat_dim),
            nn.ReLU(),
            nn.Linear(table_feat_dim, table_feat_dim),
        )

    
        self.gate_layer = nn.Linear(self.global_dim * 3, 1)
        self.use_wavelet = getattr(config, "use_wavelet", True)
        self.wavelet_name = getattr(config, "wavelet_name", "coif2")
        
    def _wavelet_global_pool(self, fused_table):
        """
        fused_table: [B, L, L, C]
        返回 h_wave: [B, C]，每个通道对应其 LL 子带的平均值。
        """
       
        if (pywt is None) or (not self.use_wavelet):
            return fused_table.mean(dim=(1, 2))

        B, L, L2, C = fused_table.shape
        assert L == L2, "表格必须是 LxL 结构"

        x = fused_table.detach().cpu().numpy()  # [B, L, L, C]
        wavelet = pywt.Wavelet(self.wavelet_name)

        h_wave_list = []
        for b in range(B):
            chan_vals = []
            for c in range(C):
                arr = x[b, :, :, c]
                LL, (LH, HL, HH) = pywt.dwt2(arr, wavelet)
                chan_vals.append(LL.mean())
            h_wave_list.append(chan_vals)

        h_wave = torch.tensor(h_wave_list, device=fused_table.device, dtype=fused_table.dtype)
        return h_wave

    def forward(self, seq, mask, pos_quant_matrix=None, cls_repr=None):
        '''
            seq: [B, L, H_bert]
            mask: [B, L]  attention_mask
            pos_quant_matrix: [B, L, L, pos_dim]
            cls_repr: [B, H_bert]
        '''
      
        table = self.seq2mat(seq, seq)   

 
        if pos_quant_matrix is not None:
            if self.fusion_type == 'add':
                fused_table = torch.cat([table, pos_quant_matrix], dim=-1)
                fused_table = self.pos_projection(fused_table)  # [B,L,L,H]
            elif self.fusion_type == 'mul':
                pos_quant_projected = self.pos_projection(pos_quant_matrix)
                fused_table = table * pos_quant_projected
            fused_table = F.relu(fused_table)
        else:
            fused_table = table

        if self.config.table_encoder == 'none':
            return fused_table

        B, L, L2, C = fused_table.shape
        assert L == L2, "表格必须是 LxL 结构"

     
        h_wave = self._wavelet_global_pool(fused_table)    
        h_wave_proj = self.proj_wave(h_wave)               

        if cls_repr is not None:
            h_cls_proj = self.proj_cls(cls_repr)   
        else:
            h_cls_proj = h_wave_proj
   
        g_global = torch.tanh(
            self.proj_global(torch.cat([h_cls_proj, h_wave_proj], dim=-1))
        )                                     

        h_tokens_proj = self.proj_token(seq)      

     
        g_global_b = g_global.view(B, 1, 1, -1)    
        hi = h_tokens_proj.view(B, L, 1, -1)      
        hj = h_tokens_proj.view(B, 1, L, -1)     

       
        z_ij = torch.cat(
            [
                g_global_b.expand(-1, L, L, -1),
                hi.expand(-1, L, L, -1),
                hj.expand(-1, L, L, -1),
            ],
            dim=-1,
        )

        delta_r = self.delta_mlp(z_ij)            
        gate = torch.sigmoid(self.gate_layer(z_ij)) 

       
        if mask is not None:
            pair_mask = (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(-1).float()
            gate = gate * pair_mask

        with torch.no_grad():
            gate_detach = gate.detach()
            if mask is not None:
                valid = pair_mask > 0
                valid_gate = gate_detach[valid]
            else:
                valid_gate = gate_detach.view(-1)

            if valid_gate.numel() > 0:
                mean_val = valid_gate.mean().item()
                std_val = valid_gate.std().item()
                pos_ratio = (valid_gate > 0.5).float().mean().item()
            else:
                mean_val = std_val = pos_ratio = 0.0

            self.last_gate_stats = {
                "mean": mean_val,          
                "std": std_val,
                "pos_ratio": pos_ratio,  
            }
   
        fused_table = fused_table + gate * delta_r 
        for layer_module in self.layer:
            fused_table = layer_module(fused_table)

        return fused_table
