import torch
import torch.nn as nn
import torch.nn.functional as F

# 特徴マップを行方向と列方向の確率分布に変えて積を取る
class CoarseMatching(nn.Module):
    def __init__(self, config, d_size):
        super().__init__()
        # general config
        self.border_rm = config['border_rm'] # _CN.MATCH_COARSE.BORDER_RM = 2
        self.temperature = config['dsmax_temperature'] # _CN.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
        self.d_size = d_size # 256

    def forward(self, feat_c0, feat_c1):
        """
        Args:
            feat_c0 (torch.Tensor): [N, L, C]
            feat_c1 (torch.Tensor): [N, S, C]
        Returns:
            conf_matrix (torch.Tensor): [M]
        """
        # normalize
        # feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5, [feat_c0, feat_c1])
        feat_c0, feat_c1 = map(lambda feat: feat / self.d_size ** .5, [feat_c0, feat_c1])

        # sim_matrix_t = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
        sim_matrix_orig = torch.matmul(feat_c0, feat_c1.permute((0, 2, 1)))
        sim_matrix = sim_matrix_orig / self.temperature
        # assert(torch.allclose(sim_matrix_t, sim_matrix, atol=1e-05))

        # conf_matrix = 行方向の確率分布＊列方向の確率分布
        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        return conf_matrix, sim_matrix_orig
