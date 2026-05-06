import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GATv2Conv

from utils import neighbor0_26_3d, neighbor0_6_3d


class PatchEmbed3D(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch: int):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        # x: (B, C, D, H, W)
        z = self.proj(x)  # (B, Demb, Dp, Hp, Wp)
        B, Demb, Dp, Hp, Wp = z.shape
        seq = z.flatten(2).transpose(1, 2)  # (B, N, Demb) with N=Dp*Hp*Wp
        return seq, (Dp, Hp, Wp)


class PatchUnembed3D(nn.Module):
    def __init__(self, out_ch: int, embed_dim: int, patch: int):
        super().__init__()
        self.patch = patch
        self.deproj = nn.ConvTranspose3d(embed_dim, out_ch, kernel_size=patch, stride=patch)

    def forward(self, seq, dhw):
        Dp, Hp, Wp = dhw
        B, N, Demb = seq.shape
        z = seq.transpose(1, 2).reshape(B, Demb, Dp, Hp, Wp)  # (B, Demb, Dp, Hp, Wp)
        x_rec = self.deproj(z)  # (B, out_ch, D, H, W)
        return x_rec


# ------------------------------------------------------------
# Same "Conv" block (GAT + FFN) as your 2D code
# ------------------------------------------------------------
class Conv(nn.Module):
    def __init__(self, h_dim, head, dropout=0.0, ffn_mult=4):
        super().__init__()
        assert h_dim % head == 0

        self.attn = GATv2Conv(h_dim, h_dim // head,heads=head, concat=True,add_self_loops=False, dropout=dropout)

        self.norm1 = nn.BatchNorm1d(h_dim)
        self.drop1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(nn.Linear(h_dim, ffn_mult * h_dim),nn.GELU(), nn.Dropout(dropout),nn.Linear(ffn_mult * h_dim, h_dim))

        self.norm2 = nn.BatchNorm1d(h_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.attn(self.norm1(x), edge_index)
        x = x + self.drop1(h)
        h = self.ffn(self.norm2(x))
        x = x + self.drop2(h)
        return x


# ------------------------------------------------------------
# 0-cell Block for 3D
# ------------------------------------------------------------
class Block3D(nn.Module):
    def __init__(self, in_channel, h_dim, patch_size, head, do_pool=True, dropout=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed3D(in_ch=in_channel, embed_dim=h_dim, patch=patch_size)
        self.patch_unembed = PatchUnembed3D(out_ch=h_dim, embed_dim=h_dim, patch=patch_size)
        self.conv1 = Conv(h_dim, head, dropout=dropout)
        self.conv2 = Conv(h_dim, head, dropout=dropout)
        self.do_pool = do_pool
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self._printed = False

    def make_batch_graph(self, x, edge_index, add_loop=False):
        # x: (B, N, D)
        device = x.device
        B, N, d = x.shape
        edge_index= edge_index.to(device)

        if add_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        x = x.reshape(B * N, d)
        E = edge_index.size(1)

        offsets = (torch.arange(B, device=device) * N).repeat_interleave(E)
        edge_index_big = edge_index.repeat(1, B) + offsets.unsqueeze(0)

        batch = torch.arange(B, device=device).repeat_interleave(N)
        return Data(x=x, edge_index=edge_index_big, batch=batch)

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape

        seq, (Dp, Hp, Wp) = self.patch_embed(x)

        if not self._printed:
            print("PatchEmbed3D:", x.shape, "-> seq:", seq.shape, "DpHpWp:", (Dp, Hp, Wp))
            self._printed = True
        #print("D2,H2,W2 =", D2, H2, W2)
        edge_index = neighbor0_26_3d(Dp, Hp, Wp)  
        g = self.make_batch_graph(seq, edge_index, add_loop=False)
        z = self.conv1(g.x, g.edge_index)
        z = self.conv2(z, g.edge_index)
        N = seq.size(1)
        z = z.reshape(B, N, -1)

        x_rec = self.patch_unembed(z, (Dp, Hp, Wp))

        if self.do_pool and min(x_rec.shape[-3:]) >= 2:
            x_rec = self.maxpool(x_rec)

        return x_rec


# ------------------------------------------------------------
# "2-cell" analogue block for 3D (really a higher-cell cube support)
# kernel = 2p, stride = p on raw volume
# + BatchNorm3d on conv output
# + (z,y,x) positional encoding
# + 6-neighborhood on the (D2,H2,W2) grid
# ------------------------------------------------------------
class TwoCellBlock3D(nn.Module):
    def __init__(self, in_channel, h_dim, patch_size, head, dropout=0.0, use_two_convs=False):
        super().__init__()
        self.use_two_convs = use_two_convs
        self.patch_size = patch_size
        self.cell2= nn.Conv3d( in_channel, h_dim,kernel_size=2 * patch_size,stride=patch_size,bias=True)
        self.pre_norm = nn.GroupNorm(num_groups=4, num_channels=h_dim)
        #self.pre_norm = nn.BatchNorm3d(h_dim)
        self.pre_act = nn.GELU()

        #self.token_norm = nn.LayerNorm(h_dim)
        self.pos_proj2 = nn.Linear(3, h_dim)  # (z,y,x)

        self.conv1 = Conv(h_dim, head, dropout=dropout)
        self.conv2 = Conv(h_dim, head, dropout=dropout) if use_two_convs else None

    def make_batch_graph(self, x, edge_index, add_loop=False):
        device = x.device
        B, N, d = x.shape
        edge_index= edge_index.to(device)

        if add_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        x = x.reshape(B * N, d)
        E = edge_index.size(1)

        offsets = (torch.arange(B, device=device) * N).repeat_interleave(E)
        edge_index_big = edge_index.repeat(1, B) + offsets.unsqueeze(0)

        batch = torch.arange(B, device=device).repeat_interleave(N)
        return Data(x=x, edge_index=edge_index_big, batch=batch)

    def forward(self, x_raw):
        # x_raw: (B, C, D, H, W)
        z = self.cell2(x_raw)          # (B, h_dim, D2, H2, W2)
        z = self.pre_act(self.pre_norm(z))

        B, Demb, D2, H2, W2 = z.shape
        seq2 = z.flatten(2).transpose(1, 2)  # (B, N2, Demb)
        #seq2 = self.token_norm(seq2)

        # positional encoding on same device
        device = seq2.device
        coords = torch.stack(torch.meshgrid(torch.arange(D2, device=device),torch.arange(H2, device=device),torch.arange(W2, device=device),indexing="ij"),dim=-1).reshape(-1, 3).float() 

        # normalize to [0,1]
        if D2 > 1:
            coords[:, 0] /= (D2 - 1)
        if H2 > 1:
            coords[:, 1] /= (H2 - 1)
        if W2 > 1:
            coords[:, 2] /= (W2 - 1)

        seq2 = seq2 + self.pos_proj2(coords).unsqueeze(0)  # (B, N2, Demb)

        edge_index= neighbor0_6_3d(D2, H2, W2)
        g = self.make_batch_graph(seq2, edge_index, add_loop=False)

        out = self.conv1(g.x, g.edge_index)
        if self.use_two_convs:
            out = self.conv2(out, g.edge_index)

        out = out.reshape(B, D2, H2, W2, Demb).permute(0, 4, 1, 2, 3).contiguous()
        return out  # (B, h_dim, D2, H2, W2)


# ------------------------------------------------------------
# CCNN3D (fusion same idea as your CCNN)
# ------------------------------------------------------------
class CCNN3D(nn.Module):
    def __init__(self, in_channel, h_dim, head, class_num, layer=5, dropout=0.0):
        super().__init__()

        self.patch = [2, 2, 1, 1, 1][:layer]
        self.in_dim = [in_channel] + [h_dim] * layer

        self.layers = nn.ModuleList([Block3D(self.in_dim[i], h_dim, self.patch[i], head, do_pool=True, dropout=dropout) for i in range(layer) ])

        self.two_cell = TwoCellBlock3D(in_channel=in_channel,h_dim=h_dim,patch_size=self.patch[0],head=head,dropout=dropout,use_two_convs=False )

        self.fuse_gate3d = nn.Sequential(nn.Conv3d(2 * h_dim, h_dim, kernel_size=1, bias=True),nn.Sigmoid())
        self.fuse_proj3d = nn.Sequential(nn.Conv3d(h_dim, h_dim, kernel_size=1, bias=True),nn.ReLU(),nn.Dropout3d(dropout))

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc2 = nn.Linear(h_dim, class_num)

    def forward(self, x):
        # x: (B, C, D, H, W)
        x_raw = x

        x2 = self.two_cell(x_raw)  # (B, h_dim, D2, H2, W2)

        for layer in self.layers:
            x = layer(x)

        x0 = x  # (B, h_dim, Df, Hf, Wf)

        # align sizes
        x2 = F.adaptive_avg_pool3d(x2, output_size=x0.shape[-3:])
        g = self.fuse_gate3d(torch.cat([x0, x2], dim=1))
        #xf = g * x0 + (1.0 - g) * x2
        xf = (1.0 - g) * x0 + g * x2
        #xf = x0 +g*x2
        xf = self.fuse_proj3d(xf)
        xf = self.pool(xf).flatten(1)  # (B, h_dim)
        return self.fc2(xf)


