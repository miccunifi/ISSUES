import torch
import torch.nn as nn


class Combiner(nn.Module):
    def __init__(self, convex_tensor: bool, input_dim: int, comb_proj: bool, comb_fusion: str):
        super(Combiner, self).__init__()
        self.map_dim = input_dim
        self.comb_proj = comb_proj
        self.comb_fusion = comb_fusion
        self.convex_tensor = convex_tensor

        if self.convex_tensor:
            branch_out_dim = self.map_dim
        else:
            branch_out_dim = 1

        comb_in_dim = self.map_dim
        comb_concat_out_dim = comb_in_dim

        if self.comb_proj:
            self.comb_image_proj = nn.Sequential(
                nn.Linear(comb_in_dim, 2 * comb_in_dim),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )

            self.comb_text_proj = nn.Sequential(
                nn.Linear(comb_in_dim, 2 * comb_in_dim),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )

            comb_in_dim = 2 * comb_in_dim

        if self.comb_fusion == 'concat':
            branch_in_dim = 2 * comb_in_dim
        elif self.comb_fusion == 'align':
            branch_in_dim = comb_in_dim
        else:
            ValueError()

        self.comb_shared_branch = nn.Sequential(
            nn.Linear(branch_in_dim, 2 * branch_in_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2 * branch_in_dim, branch_out_dim),
            nn.Sigmoid()
        )

        self.comb_concat_branch = nn.Sequential(
            nn.Linear(branch_in_dim, 2 * branch_in_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2 * branch_in_dim, comb_concat_out_dim),
        )

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def forward(self, img_projection, post_features):
        if self.comb_proj:
            proj_img_fea = self.comb_image_proj(img_projection)
            proj_txt_fea = self.comb_text_proj(post_features)
        else:
            proj_img_fea = img_projection
            proj_txt_fea = post_features

        if self.comb_fusion == 'concat':
            comb_features = torch.cat([proj_img_fea, proj_txt_fea], dim=1)
        elif self.comb_fusion == 'align':
            comb_features = torch.mul(proj_img_fea, proj_txt_fea)
        else:
            raise ValueError()

        side_branch = self.comb_shared_branch(comb_features)
        central_branch = self.comb_concat_branch(comb_features)

        features = central_branch + ((1 - side_branch) * img_projection + side_branch * post_features)

        return features
