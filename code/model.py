import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GraphNorm


from utils import neighbor0, neighbor0_4


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=784, patch=16):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        z = self.proj(x)  
        B, D, Hp, Wp = z.shape
        seq = z.flatten(2).transpose(1, 2)  
        return seq, (Hp, Wp)


class PatchUnembed(nn.Module):
    def __init__(self, out_ch=3, embed_dim=784, patch=16):
        super().__init__()
        self.patch = patch
        self.deproj = nn.ConvTranspose2d(embed_dim, out_ch, kernel_size=patch, stride=patch)

    def forward(self, seq, hw):
        Hp, Wp = hw
        B, N, D = seq.shape
        z = seq.transpose(1, 2).reshape(B, D, Hp, Wp)  
        x_rec = self.deproj(z)  
        return x_rec

class Conv(nn.Module):
    def __init__(self, h_dim, head, dropout=0.0, ffn_mult=4):
        super().__init__()
        assert h_dim % head == 0
        self.attn = GATv2Conv(h_dim, h_dim // head, heads=head, concat=True,add_self_loops=False, dropout=dropout)
        self.norm1 = nn.BatchNorm1d(h_dim)
        self.drop1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(h_dim, ffn_mult * h_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(ffn_mult * h_dim, h_dim),)
        self.norm2 = nn.BatchNorm1d(h_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.attn(self.norm1(x), edge_index)
        x = x + self.drop1(h)
        h = self.ffn(self.norm2(x))
        x = x + self.drop2(h)
        return x

class Block(nn.Module):
    def __init__(self, in_channel, h_dim, patch_size, head, do_pool=True, dropout=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(in_ch=in_channel, embed_dim=h_dim, patch=patch_size)
        self.patch_unembed = PatchUnembed(out_ch=h_dim, embed_dim=h_dim, patch=patch_size)
        self.conv1 = Conv(h_dim, head, dropout=dropout)
        self.conv2 = Conv(h_dim, head, dropout=dropout)
        self.do_pool = do_pool
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self._printed = False  
    def make_batch_graph(self, x, edge_index, add_loop=False):
        device = x.device
        B, N, d = x.shape
        edge_index = edge_index.to(device)

        if add_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        x = x.reshape(B * N, d)
        E = edge_index.size(1)
        offsets = (torch.arange(B, device=device) * N).repeat_interleave(E)
        edge_index_big = edge_index.repeat(1, B) + offsets.unsqueeze(0)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        data = Data(x=x, edge_index=edge_index_big, batch=batch)
        return data

    def forward(self, x):
        B, C, H, W = x.shape
        seq, (Hp, Wp) = self.patch_embed(x)
        assert Hp == Wp, f"Patch grid not square: Hp={Hp}, Wp={Wp}"

        if not self._printed:
            print("PatchEmbed:", x.shape, "-> seq:", seq.shape, "HpWp:", (Hp, Wp))
            self._printed = True

        N = seq.size(1)
        edge_index = neighbor0(Hp)
        g = self.make_batch_graph(seq, edge_index, add_loop=False)
        z = self.conv1(g.x, g.edge_index)
        z = self.conv2(z, g.edge_index)
        z = z.reshape(B, N, -1)
        x_rec = self.patch_unembed(z, (Hp, Wp))
        if self.do_pool and x_rec.size(-1) >= 2 and x_rec.size(-2) >= 2:
            x_rec = self.maxpool(x_rec)
        return x_rec

class TwoCellBlock(nn.Module):
    def __init__(self, in_channel, h_dim, patch_size, head, dropout=0.0, use_two_convs=False):
        super().__init__()
        self.patch_size = patch_size
        self.use_two_convs = use_two_convs
        self.cell2 = nn.Conv2d(in_channel, h_dim,kernel_size=2 * patch_size, stride=patch_size,bias=True)
        self.pre_norm = nn.GroupNorm(num_groups=4, num_channels=h_dim)
        self.pre_act  = nn.GELU()
        self.pos_proj2 = nn.Linear(2, h_dim)
        self.conv1 = Conv(h_dim, head, dropout=dropout)
        self.conv2 = Conv(h_dim, head, dropout=dropout) if use_two_convs else None

    def make_batch_graph(self, x, edge_index, add_loop=False):
        device = x.device
        B, N, d = x.shape
        edge_index = edge_index.to(device)
        if add_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        x = x.reshape(B * N, d)
        E = edge_index.size(1)
        offsets = (torch.arange(B, device=device) * N).repeat_interleave(E)
        edge_index_big = edge_index.repeat(1, B) + offsets.unsqueeze(0)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        return Data(x=x, edge_index=edge_index_big, batch=batch)

    def forward(self, x_raw):
        z2 = self.cell2(x_raw)                     
        z2 = self.pre_act(self.pre_norm(z2))        
        B, D, H2, W2 = z2.shape
        seq2 = z2.flatten(2).transpose(1, 2)        
        device = seq2.device
        coords = torch.stack(torch.meshgrid(torch.arange(H2, device=device),torch.arange(W2, device=device),indexing="ij"),dim=-1).reshape(-1, 2).float()  
        if H2 > 1:
            coords[:, 0] = coords[:, 0] / (H2 - 1)
        if W2 > 1:
            coords[:, 1] = coords[:, 1] / (W2 - 1)
        pos2 = self.pos_proj2(coords)               
        seq2 = seq2 + pos2.unsqueeze(0)             
        edge_index2 = neighbor0_4(H2)
        g2 = self.make_batch_graph(seq2, edge_index2, add_loop=False)

        out = self.conv1(g2.x, g2.edge_index)
        if self.use_two_convs:
            out = self.conv2(out, g2.edge_index)

        out = out.reshape(B, H2, W2, D).permute(0, 3, 1, 2).contiguous()  
        return out

class CCNN(nn.Module):
    def __init__(self, in_channel, h_dim, head, class_num, layer=5, dropout=0.0):
        super().__init__()

        self.patch = [2, 2, 1, 1, 1]
        self.in_dim = [in_channel, h_dim, h_dim, h_dim, h_dim, h_dim] 
        self.layers = nn.ModuleList([Block(self.in_dim[i], h_dim, self.patch[i], head, do_pool=True, dropout=dropout)for i in range(layer)])
        self.two_cell = TwoCellBlock(in_channel, h_dim, patch_size=self.patch[0], head=head, dropout=dropout) 
        self.fuse_gate2d = nn.Sequential(nn.Conv2d(2 * h_dim, h_dim, kernel_size=1, bias=True),nn.Sigmoid())
        self.fuse_proj2d = nn.Sequential(nn.Conv2d(h_dim, h_dim, kernel_size=1, bias=True),nn.ReLU(),nn.Dropout2d(dropout))  
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(h_dim, class_num)   
         
    def forward(self, x):
        x_raw = x
        x2 = self.two_cell(x_raw)  
        for layer in self.layers:
            x = layer(x)
        x0 = x  
        x2 = F.adaptive_avg_pool2d(x2, output_size=x0.shape[-2:])
        g = self.fuse_gate2d(torch.cat([x0, x2], dim=1))  
        xf = x0 +g*x2
        xf = self.fuse_proj2d(xf)
        xf =  self.pool(xf).flatten(1)
        x = self.fc2(xf)
        return x

