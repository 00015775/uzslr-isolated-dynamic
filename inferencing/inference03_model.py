import torch
import torch.nn as nn
import torch.nn.functional as F

# simply copied the model code experimented on 03_ak_model_dev_v1.ipynb
# but in ECA lazy initiliaztion for self.conv is NOT used, the rest is the same
class ECA(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            masked_x = x * mask_expanded
            sum_x = masked_x.sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            y = sum_x / count
        else:
            y = x.mean(dim=1)
        y = y.unsqueeze(1)
        y = self.conv(y)
        y = torch.sigmoid(y)
        return x * y

class LateDropout(nn.Module):
    def __init__(self, p=0.5, start_step=0):
        super().__init__()
        self.p = p
        self.start_step = start_step
        self.dropout = nn.Dropout(p)
        self.register_buffer('train_counter', torch.tensor(0, dtype=torch.long))
    
    def forward(self, x):
        if self.training:
            if self.train_counter >= self.start_step:
                x = self.dropout(x)
            self.train_counter += 1
        return x

class CausalDWConv1D(nn.Module):
    def __init__(self, channels, kernel_size=17, dilation=1):
        super().__init__()
        self.padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=0, dilation=dilation, groups=channels, bias=False)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x

class Conv1DBlock(nn.Module):
    def __init__(self, channels, kernel_size=17, dilation=1, drop_rate=0.0, expand_ratio=2, activation='swish'):
        super().__init__()
        self.channels = channels
        expanded_channels = channels * expand_ratio
        self.expand_conv = nn.Linear(channels, expanded_channels)
        self.activation = nn.SiLU() if activation == 'swish' else nn.GELU()
        self.dwconv = CausalDWConv1D(expanded_channels, kernel_size, dilation)
        self.bn = nn.BatchNorm1d(expanded_channels)
        self.eca = ECA(kernel_size=5)
        self.project_conv = nn.Linear(expanded_channels, channels)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
    
    def forward(self, x, mask=None):
        skip = x
        x = self.expand_conv(x)
        x = self.activation(x)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        x = self.eca(x, mask)
        x = self.project_conv(x)
        x = self.dropout(x)
        x = x + skip
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim=256, num_heads=4, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2, activation='swish'):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_dropout)
        self.dropout1 = nn.Dropout(drop_rate)
        self.norm2 = nn.BatchNorm1d(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expand, bias=False),
            nn.SiLU() if activation == 'swish' else nn.GELU(),
            nn.Linear(dim * expand, dim, bias=False),
        )
        self.dropout2 = nn.Dropout(drop_rate)
    
    def forward(self, x, mask=None):
        residual = x
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.attn(x, mask)
        x = self.dropout1(x)
        x = x + residual
        residual = x
        x_norm = x.transpose(1, 2)
        x_norm = self.norm2(x_norm)
        x_norm = x_norm.transpose(1, 2)
        x = self.ffn(x_norm)
        x = self.dropout2(x)
        x = x + residual
        return x

class SignLanguageModel(nn.Module):
    def __init__(self, max_len=32, channels=708, num_classes=50, dim=192, dropout_step=0, pad_value=-100.0):
        super().__init__()
        self.max_len = max_len
        self.channels = channels
        self.num_classes = num_classes
        self.dim = dim
        self.pad_value = pad_value
        
        self.stem_conv = nn.Linear(channels, dim, bias=False)
        self.stem_bn = nn.BatchNorm1d(dim)
        
        self.conv_blocks_1 = nn.ModuleList([Conv1DBlock(dim, kernel_size=17, drop_rate=0.2) for _ in range(3)])
        self.transformer_1 = TransformerBlock(dim, expand=2)
        
        self.conv_blocks_2 = nn.ModuleList([Conv1DBlock(dim, kernel_size=17, drop_rate=0.2) for _ in range(3)])
        self.transformer_2 = TransformerBlock(dim, expand=2)
        
        if dim == 384:
            self.conv_blocks_3 = nn.ModuleList([Conv1DBlock(dim, kernel_size=17, drop_rate=0.2) for _ in range(3)])
            self.transformer_3 = TransformerBlock(dim, expand=2)
            self.conv_blocks_4 = nn.ModuleList([Conv1DBlock(dim, kernel_size=17, drop_rate=0.2) for _ in range(3)])
            self.transformer_4 = TransformerBlock(dim, expand=2)
        else:
            self.conv_blocks_3 = None
            self.conv_blocks_4 = None
        
        self.top_conv = nn.Linear(dim, dim * 2)
        self.late_dropout = LateDropout(0.8, start_step=dropout_step)
        self.classifier = nn.Linear(dim * 2, num_classes)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x, mask=None):
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            x = x * mask_expanded
        
        x = self.stem_conv(x)
        x = x.transpose(1, 2)
        x = self.stem_bn(x)
        x = x.transpose(1, 2)
        
        for block in self.conv_blocks_1:
            x = block(x, mask)
        x = self.transformer_1(x, mask)
        
        for block in self.conv_blocks_2:
            x = block(x, mask)
        x = self.transformer_2(x, mask)
        
        if self.conv_blocks_3 is not None:
            for block in self.conv_blocks_3:
                x = block(x, mask)
            x = self.transformer_3(x, mask)
            for block in self.conv_blocks_4:
                x = block(x, mask)
            x = self.transformer_4(x, mask)
        
        x = self.top_conv(x)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            masked_x = x * mask_expanded
            sum_x = masked_x.sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            x = sum_x / count
        else:
            x = x.mean(dim=1)
        
        x = self.late_dropout(x)
        x = self.classifier(x)
        return x