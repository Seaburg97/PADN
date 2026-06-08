import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from pathlib import Path


PRIOR_DIR = Path(__file__).resolve().parent / "prior_params"
GLOBAL_PRIOR_CSV = PRIOR_DIR / "global_prior_params.csv"
REGION_NAMES = ['ACA', 'MCA', 'PCA', 'Brainstem', 'Cerebellum', 'Cistern', 'IVH']


def load_region_prior_params(prior_csv_path):
    prior_csv_path = Path(prior_csv_path)
    if not prior_csv_path.exists():
        print(f"Warning: prior parameter file not found, using neutral priors: {prior_csv_path}")
        return {
            'beta': np.zeros(len(REGION_NAMES), dtype=np.float32),
            'threshold': np.zeros(len(REGION_NAMES), dtype=np.float32),
            'std': np.ones(len(REGION_NAMES), dtype=np.float32),
        }

    df = pd.read_csv(prior_csv_path, encoding='utf-8-sig')
    if 'region' in df.columns:
        df = df.set_index('region')

    df = df.reindex(REGION_NAMES)
    df[['beta', 'threshold_T', 'std_SD']] = df[['beta', 'threshold_T', 'std_SD']].fillna({
        'beta': 0.0,
        'threshold_T': 0.0,
        'std_SD': 1.0,
    })
    return {
        'beta': df['beta'].to_numpy(dtype=np.float32),
        'threshold': df['threshold_T'].to_numpy(dtype=np.float32),
        'std': df['std_SD'].to_numpy(dtype=np.float32),
    }


CLINICAL_PRIORS_PRE = load_region_prior_params(PRIOR_DIR / "region_prior_params_pre.csv")
CLINICAL_PRIORS_POST = load_region_prior_params(PRIOR_DIR / "region_prior_params_post.csv")
CLINICAL_PRIORS_OR = {
    'pre': CLINICAL_PRIORS_PRE,
    'post': CLINICAL_PRIORS_POST,
}


class _BaseRegionPriorModule(nn.Module):
    def _init_dual_priors(self, num_regions, clinical_priors=None, beta=None, threshold=None):
        if clinical_priors is not None:
            if 'pre' in clinical_priors and 'post' in clinical_priors:
                pre_priors = clinical_priors['pre']
                post_priors = clinical_priors['post']
                self.register_buffer('pre_beta', torch.FloatTensor(pre_priors['beta'][:num_regions]))
                self.register_buffer('pre_threshold', torch.FloatTensor(pre_priors['threshold'][:num_regions]))
                self.register_buffer('pre_std', torch.FloatTensor(pre_priors['std'][:num_regions]))
                self.register_buffer('post_beta', torch.FloatTensor(post_priors['beta'][:num_regions]))
                self.register_buffer('post_threshold', torch.FloatTensor(post_priors['threshold'][:num_regions]))
                self.register_buffer('post_std', torch.FloatTensor(post_priors['std'][:num_regions]))
            else:
                beta = clinical_priors.get('beta')
                threshold = clinical_priors.get('threshold')
                std = clinical_priors.get('std')
                self.register_buffer('pre_beta', torch.FloatTensor(beta[:num_regions]))
                self.register_buffer('pre_threshold', torch.FloatTensor(threshold[:num_regions]))
                self.register_buffer('pre_std', torch.FloatTensor(std[:num_regions]) if std is not None else torch.ones(num_regions))
                self.register_buffer('post_beta', torch.FloatTensor(beta[:num_regions]))
                self.register_buffer('post_threshold', torch.FloatTensor(threshold[:num_regions]))
                self.register_buffer('post_std', torch.FloatTensor(std[:num_regions]) if std is not None else torch.ones(num_regions))
        elif beta is not None:
            self.register_buffer('pre_beta', torch.FloatTensor(beta[:num_regions]))
            self.register_buffer('pre_threshold', torch.FloatTensor(threshold[:num_regions]))
            self.register_buffer('pre_std', torch.ones(num_regions))
            self.register_buffer('post_beta', torch.FloatTensor(beta[:num_regions]))
            self.register_buffer('post_threshold', torch.FloatTensor(threshold[:num_regions]))
            self.register_buffer('post_std', torch.ones(num_regions))
        else:
            self.register_buffer('pre_beta', torch.ones(num_regions))
            self.register_buffer('pre_threshold', torch.zeros(num_regions))
            self.register_buffer('pre_std', torch.ones(num_regions))
            self.register_buffer('post_beta', torch.ones(num_regions))
            self.register_buffer('post_threshold', torch.zeros(num_regions))
            self.register_buffer('post_std', torch.ones(num_regions))

    def compute_raw_prior_no_std(self, volumes, beta, threshold, std=None):
        raw_score = beta.unsqueeze(0) * torch.clamp(
            volumes - threshold.unsqueeze(0),
            min=0.0
        )
        return raw_score


class RegionReconstructionModule(_BaseRegionPriorModule):
    def __init__(
        self,
        in_channels,
        num_regions=7,
        clinical_priors=None,
        beta=None,
        threshold=None,
        prior_phase='both',
    ):
        super().__init__()
        self.num_regions = num_regions
        self.prior_phase = prior_phase
        self._init_dual_priors(num_regions, clinical_priors=clinical_priors, beta=beta, threshold=threshold)

        self.region_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels, 3, 1, 1, bias=False),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, in_channels, 3, 1, 1, bias=False),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
            ) for _ in range(num_regions)
        ])
        self.region_attention_fc = nn.Linear(in_channels, num_regions)
        self.softmax = nn.Softmax(dim=1)
        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, diff_feat, region_masks, volumes=None, return_details=False, enable_prior=True):
        if region_masks.shape[2:] != diff_feat.shape[2:]:
            region_masks = F.interpolate(region_masks, size=diff_feat.shape[2:], mode='nearest')

        region_features = []
        for i in range(self.num_regions):
            mask = region_masks[:, i:i+1]
            masked_feat = diff_feat * mask
            region_features.append(self.region_convs[i](masked_feat))

        global_context = F.adaptive_avg_pool3d(diff_feat, 1).view(diff_feat.size(0), -1)
        image_logits = self.region_attention_fc(global_context)

        pre_prior_scores = torch.zeros_like(image_logits)
        post_prior_scores = torch.zeros_like(image_logits)
        fused_prior_bias = torch.zeros_like(image_logits)

        if volumes is not None and enable_prior:
            if volumes.size(1) >= self.num_regions * 2:
                pre_volumes = volumes[:, :self.num_regions]
                post_volumes = volumes[:, self.num_regions:self.num_regions * 2]
            else:
                pre_volumes = torch.zeros_like(volumes[:, :self.num_regions])
                post_volumes = volumes[:, :self.num_regions]

            if self.prior_phase in ('pre', 'both'):
                pre_prior_scores = self.compute_raw_prior_no_std(pre_volumes, self.pre_beta, self.pre_threshold, self.pre_std)
                fused_prior_bias = fused_prior_bias + pre_prior_scores

            if self.prior_phase in ('post', 'both'):
                post_prior_scores = self.compute_raw_prior_no_std(post_volumes, self.post_beta, self.post_threshold, self.post_std)
                fused_prior_bias = fused_prior_bias + post_prior_scores

        attention_logits = image_logits + fused_prior_bias
        attention_weights = self.softmax(attention_logits)

        weighted_sum = torch.zeros_like(diff_feat)
        for i in range(self.num_regions):
            weighted_sum += region_features[i] * attention_weights[:, i:i+1, None, None, None]

        output_feat = self.fusion(weighted_sum + diff_feat)

        if return_details:
            details = {
                'prior_arch': 'region_reconstruction',
                'image_logits': image_logits,
                'pre_prior_raw': pre_prior_scores,
                'post_prior_raw': post_prior_scores,
                'pre_prior_bias': pre_prior_scores,
                'post_prior_bias': post_prior_scores,
                'fused_prior_bias': fused_prior_bias,
                'attention_logits': attention_logits,
            }
            return output_feat, attention_weights, details
        return output_feat, attention_weights


class PriorBranchEncoder(_BaseRegionPriorModule):
    def __init__(self, num_regions=7, clinical_priors=None, beta=None, threshold=None):
        super().__init__()
        self.num_regions = num_regions
        self._init_dual_priors(num_regions, clinical_priors=clinical_priors, beta=beta, threshold=threshold)
        self.prior_head = nn.Linear(num_regions, 6)

    def forward(self, volumes, return_details=False):
        if volumes is None:
            raise ValueError('PriorBranchEncoder  volumes ')

        if volumes.size(1) >= self.num_regions * 2:
            pre_volumes = volumes[:, :self.num_regions]
            post_volumes = volumes[:, self.num_regions:self.num_regions * 2]
        else:
            pre_volumes = torch.zeros_like(volumes[:, :self.num_regions])
            post_volumes = volumes[:, :self.num_regions]

        pre_prior_scores = self.compute_raw_prior_no_std(pre_volumes, self.pre_beta, self.pre_threshold, self.pre_std)
        post_prior_scores = self.compute_raw_prior_no_std(post_volumes, self.post_beta, self.post_threshold, self.post_std)
        region_scores = pre_prior_scores + post_prior_scores
        prior_logits = self.prior_head(region_scores)

        if return_details:
            return prior_logits, {
                'pre_prior_scores': pre_prior_scores,
                'post_prior_scores': post_prior_scores,
                'region_scores': region_scores,
                'prior_logits': prior_logits,
            }
        return prior_logits


class PhasePriorAttentionHead(nn.Module):
    def __init__(self, in_channels, num_regions=7, dropout=0.5):
        super().__init__()
        self.num_regions = num_regions
        self.region_logit_fc = nn.Linear(in_channels, 1)
        self.softmax = nn.Softmax(dim=1)
        self.head = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 6),
        )

    def forward(self, feature_map, region_masks, prior_scores, return_details=False):
        if region_masks.shape[2:] != feature_map.shape[2:]:
            region_masks = F.interpolate(region_masks, size=feature_map.shape[2:], mode='nearest')

        pooled_features = []
        for idx in range(self.num_regions):
            mask = region_masks[:, idx:idx+1]
            masked_feat = feature_map * mask
            pooled = masked_feat.sum(dim=(2, 3, 4)) / (mask.sum(dim=(2, 3, 4)) + 1e-8)
            pooled_features.append(pooled)
        pooled_features = torch.stack(pooled_features, dim=1)             

        image_region_logits = self.region_logit_fc(pooled_features).squeeze(-1)
        attention_logits = image_region_logits + prior_scores
        attention_weights = self.softmax(attention_logits)

        weighted_feature = torch.sum(
            pooled_features * attention_weights.unsqueeze(-1),
            dim=1,
        )
        prior_logits = self.head(weighted_feature)

        if return_details:
            return prior_logits, {
                'pooled_features': pooled_features,
                'image_region_logits': image_region_logits,
                'prior_scores': prior_scores,
                'attention_logits': attention_logits,
                'attention_weights': attention_weights,
                'weighted_feature': weighted_feature,
                'prior_logits': prior_logits,
            }
        return prior_logits


def load_and_merge_region_template(template_path, target_shape=None):
    template = nib.load(template_path).get_fdata()

    if target_shape is not None:
        zoom_factors = [t / c for t, c in zip(target_shape, template.shape)]
        template_resized = zoom(template, zoom_factors, order=0)
        output_shape = target_shape
    else:
        template_resized = template
        output_shape = template.shape

    region_masks = np.zeros((7, *output_shape), dtype=np.float32)

    region_masks[0] = ((template_resized == 1) | (template_resized == 2)).astype(np.float32)
    region_masks[1] = ((template_resized == 3) | (template_resized == 4)).astype(np.float32)
    region_masks[2] = ((template_resized == 5) | (template_resized == 6)).astype(np.float32)
    region_masks[3] = ((template_resized == 7) | (template_resized == 8)).astype(np.float32)
    region_masks[4] = ((template_resized == 9) | (template_resized == 10)).astype(np.float32)
    region_masks[5] = (template_resized == 11).astype(np.float32)
    region_masks[6] = (template_resized == 12).astype(np.float32)
    return region_masks
