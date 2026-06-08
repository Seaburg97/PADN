import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, cohen_kappa_score
import os
import gc
import random
import json
import time
from tqdm import tqdm
from augmentation_simple import SimpleIntensityAugmentation, NoAugmentation
from region_attention import (
    RegionReconstructionModule,
    PriorBranchEncoder,
    PhasePriorAttentionHead,
    CLINICAL_PRIORS_OR,
    REGION_NAMES,
)

def set_seed(seed=41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
                                               
                                            


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def create_eval_loader(dataset, batch_size, num_workers, generator_seed):
    generator = build_dataloader_generator(generator_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def normalize_detail_rows(values, expected_rows):
    if values is None:
        return None

    if isinstance(values, list):
        if len(values) == expected_rows:
            return values
        if len(values) == 0:
            return values
        repeat_factor = expected_rows // len(values)
        if repeat_factor * len(values) == expected_rows:
            expanded = []
            for value in values:
                expanded.extend([value] * repeat_factor)
            return expanded
        return values

    if not isinstance(values, np.ndarray):
        return values

    if values.shape[0] == expected_rows:
        return values

    if values.ndim >= 3:
        reshaped = values.reshape(-1, values.shape[-1])
        if reshaped.shape[0] == expected_rows:
            return reshaped

    if values.ndim == 2 and values.size == expected_rows:
        reshaped = values.reshape(expected_rows)
        return reshaped

    return values


def infer_patience_counter_from_history(training_history):
    if not training_history:
        return 0

    val_kappas = training_history.get('val_kappa', [])
    if not val_kappas:
        return 0

    best_so_far = float('-inf')
    patience_counter = 0

    for raw_kappa in val_kappas:
        kappa = float(raw_kappa)
        if kappa > best_so_far:
            best_so_far = kappa
            patience_counter = 0
        else:
            patience_counter += 1

    return patience_counter


def load_model_state_compat(model, checkpoint):
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['model_state_dict'],
        strict=False,
    )
    if missing_keys:
        print(f"Warning: checkpointModel config: {missing_keys}")
    ignorable_unexpected = {'region_masks'}
    real_unexpected = [k for k in unexpected_keys if k not in ignorable_unexpected]
    if real_unexpected:
        print(f"Warning: checkpoint: {real_unexpected}")

class WeightedOrdinalLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights                           

    def forward(self, outputs, ordinal_labels, true_labels):
        bce_loss = F.binary_cross_entropy_with_logits(
            outputs, ordinal_labels, reduction='none'
        )              

        if self.class_weights is not None:
            sample_weights = self.class_weights[true_labels.long()]            
            sample_weights = sample_weights.unsqueeze(1)              
            bce_loss = bce_loss * sample_weights

        return bce_loss.mean()

class FocalOrdinalLoss(nn.Module):
    def __init__(self, class_weights=None, alpha=0.25, gamma=2.0):
        super().__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, ordinal_labels, true_labels):
        bce_loss = F.binary_cross_entropy_with_logits(
            outputs, ordinal_labels, reduction='none'
        )              

        probs = torch.sigmoid(outputs)
        pt = torch.where(ordinal_labels == 1, probs, 1 - probs)

        focal_weight = (1 - pt) ** self.gamma

        focal_loss = focal_weight * bce_loss

        if self.class_weights is not None:
            sample_weights = self.class_weights[true_labels.long()]            
            sample_weights = sample_weights.unsqueeze(1)              
            focal_loss = focal_loss * sample_weights

        focal_loss = self.alpha * focal_loss

        return focal_loss.mean()

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1, 1)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.cbam = CBAM(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DepthAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depth_conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1),
                     padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        depth_feat = self.depth_conv(x)
        att = self.attention(depth_feat)
        return x * att + depth_feat

class CrossStreamAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query_conv = nn.Conv3d(channels, channels // 8, 1)
        self.key_conv = nn.Conv3d(channels, channels // 8, 1)
        self.value_conv = nn.Conv3d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, pre, post):
        batch, C, D, H, W = pre.size()

        query = self.query_conv(pre).view(batch, -1, D*H*W).permute(0, 2, 1)
        key = self.key_conv(post).view(batch, -1, D*H*W)
        value = self.value_conv(post).view(batch, -1, D*H*W)

        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, C, D, H, W)

        out = self.gamma * out + pre
        return out

class EnhancedChangeAwareModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.diff_direction = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )
        self.diff_magnitude = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(channels),
            nn.Sigmoid()
        )

        self.depth_att = DepthAttention(channels)

        self.cross_att = CrossStreamAttention(channels)

        self.fusion_weight = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels * 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, pre, post):
        pre_enhanced = self.cross_att(pre, post)
        post_enhanced = self.cross_att(post, pre)

        diff = post_enhanced - pre_enhanced
        diff_dir = self.diff_direction(diff)
        diff_mag = self.diff_magnitude(torch.abs(diff))
        diff_feat = diff_dir * diff_mag

        diff_feat = self.depth_att(diff_feat)

        concat = torch.cat([pre_enhanced, post_enhanced, diff_feat, diff], dim=1)
        weight = self.fusion_weight(concat)

        out = pre_enhanced * (1 - weight) + post_enhanced * weight + diff_feat * 0.5
        return out

class MultiScaleFeaturePyramid(nn.Module):
    def __init__(self, channels_list=[32, 64, 128]):                            
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv3d(c, 128, 1) for c in channels_list           
        ])
        self.output_conv = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, 1),           
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        out = self.lateral_convs[-1](features[-1])

        for i in range(len(features) - 2, -1, -1):
            lateral = self.lateral_convs[i](features[i])
            upsampled = F.interpolate(out, size=lateral.shape[2:],
                                     mode='trilinear', align_corners=False)
            out = lateral + upsampled

        out = self.output_conv(out)
        return out

class EnhancedDualStreamNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        use_region_attention=True,
    ):
        super().__init__()
        self.use_region_attention = use_region_attention

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, 7, 2, 3, bias=False),         
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, 2, 1)
        )

        self.pre_branch = nn.Sequential(
            ResBlock3D(16, 32, 2),         
            ResBlock3D(32, 32, 1)
        )
        self.post_branch = nn.Sequential(
            ResBlock3D(16, 32, 2),
            ResBlock3D(32, 32, 1)
        )

        if use_region_attention:
            self.pre_change_region_attention = RegionReconstructionModule(
                in_channels=32,
                num_regions=7,
                clinical_priors=CLINICAL_PRIORS_OR,
                prior_phase='pre',
            )
            self.post_change_region_attention = RegionReconstructionModule(
                in_channels=32,
                num_regions=7,
                clinical_priors=CLINICAL_PRIORS_OR,
                prior_phase='post',
            )

        self.change_aware = EnhancedChangeAwareModule(32)         

        self.layer1 = nn.Sequential(
            ResBlock3D(32, 64, 2),
            ResBlock3D(64, 64, 1)
        )
        self.layer2 = nn.Sequential(
            ResBlock3D(64, 128, 2),
            ResBlock3D(128, 128, 1)
        )

        self.fpn = MultiScaleFeaturePyramid([32, 64, 128])

    @staticmethod
    def _pool_region_scores(feature_map, region_masks):
        if region_masks.shape[2:] != feature_map.shape[2:]:
            region_masks = F.interpolate(region_masks, size=feature_map.shape[2:], mode='nearest')

        pooled_scores = []
        for idx in range(region_masks.shape[1]):
            mask = region_masks[:, idx:idx+1]
            masked_feat = feature_map * mask
            pooled = masked_feat.sum(dim=(2, 3, 4)) / (mask.sum(dim=(2, 3, 4)) + 1e-8)
            pooled_scores.append(pooled.mean(dim=1, keepdim=True))
        return torch.cat(pooled_scores, dim=1)

    def forward(
        self,
        pre,
        post,
        pre_region_masks=None,
        post_region_masks=None,
        volumes=None,
        return_details=False,
        return_branch_features=False,
        enable_region_prior=True,
        use_region_attention_forward=True,
    ):
        pre_feat = self.stem(pre)
        post_feat = self.stem(post)

        pre_mid = self.pre_branch(pre_feat)
        post_mid = self.post_branch(post_feat)

        pre_response = None
        post_response = None
        pre_change_region_weights = None
        post_change_region_weights = None
        pre_change_region_details = None
        post_change_region_details = None
        if pre_region_masks is not None:
            pre_response = self._pool_region_scores(pre_mid, pre_region_masks)
        if post_region_masks is not None:
            post_response = self._pool_region_scores(post_mid, post_region_masks)

        change_region_weights = None
        change_region_details = None
        if (
            self.use_region_attention
            and use_region_attention_forward
            and pre_region_masks is not None
            and post_region_masks is not None
        ):
            if return_details:
                pre_mid, pre_change_region_weights, pre_change_region_details = self.pre_change_region_attention(
                    pre_mid, pre_region_masks, volumes, return_details=True, enable_prior=enable_region_prior
                )
                post_mid, post_change_region_weights, post_change_region_details = self.post_change_region_attention(
                    post_mid, post_region_masks, volumes, return_details=True, enable_prior=enable_region_prior
                )
            else:
                pre_mid, pre_change_region_weights = self.pre_change_region_attention(
                    pre_mid, pre_region_masks, volumes, enable_prior=enable_region_prior
                )
                post_mid, post_change_region_weights = self.post_change_region_attention(
                    post_mid, post_region_masks, volumes, enable_prior=enable_region_prior
                )
            if pre_change_region_weights is not None and post_change_region_weights is not None:
                change_region_weights = (pre_change_region_weights + post_change_region_weights) / 2.0
            else:
                change_region_weights = pre_change_region_weights if pre_change_region_weights is not None else post_change_region_weights

        fused = self.change_aware(pre_mid, post_mid)
        change_feat = fused

        feat1 = self.layer1(fused)
        feat2 = self.layer2(feat1)

        multi_scale_feat = self.fpn([fused, feat1, feat2])

        if return_details:
            details = {
                'pre_branch': {'region_response': pre_response} if pre_response is not None else {},
                'post_branch': {'region_response': post_response} if post_response is not None else {},
                'pre_change_branch': pre_change_region_details or {},
                'post_change_branch': post_change_region_details or {},
                'change_branch': change_region_details or {},
                'change_feat': change_feat,
            }
            if return_branch_features:
                details['branch_features'] = {
                    'pre_mid_feat': pre_mid,
                    'post_mid_feat': post_mid,
                }
            weights = {
                'change': change_region_weights,
            }
            return multi_scale_feat, weights, details
        return multi_scale_feat, {'change': change_region_weights}


class DualChannelPredictor(nn.Module):
    def __init__(
        self,
        dropout=0.5,
        use_region_attention=True,
        region_masks=None,
        model_mode='current_strong_prior',
    ):
        super().__init__()
        self.model_mode = model_mode
        self.enable_strong_region_prior = use_region_attention and model_mode == 'current_strong_prior'

        self.dual_stream = EnhancedDualStreamNet(
            in_channels=1,
            use_region_attention=self.enable_strong_region_prior,
        )

        if region_masks is not None and use_region_attention:
            self.register_buffer('region_masks', region_masks)
        else:
            self.region_masks = None

        if model_mode == 'residual_prior_branch':
            self.prior_branch = PriorBranchEncoder(
                num_regions=7,
                clinical_priors=CLINICAL_PRIORS_OR,
            )
            self.pre_prior_attention_head = None
            self.post_prior_attention_head = None
        elif model_mode == 'prepost_prior_attention':
            self.prior_branch = PriorBranchEncoder(
                num_regions=7,
                clinical_priors=CLINICAL_PRIORS_OR,
            )
            self.pre_prior_attention_head = PhasePriorAttentionHead(
                in_channels=32,
                num_regions=7,
                dropout=dropout,
            )
            self.post_prior_attention_head = PhasePriorAttentionHead(
                in_channels=32,
                num_regions=7,
                dropout=dropout,
            )
        else:
            self.prior_branch = None
            self.pre_prior_attention_head = None
            self.post_prior_attention_head = None

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 6)
        )

    def forward(
        self,
        pre_ct,
        post_ct,
        volumes=None,
        pre_region_masks=None,
        post_region_masks=None,
        return_details=False,
        enable_region_prior=True,
        disable_all_priors=False,
    ):
        batch_size = pre_ct.size(0)
        region_masks_batch = pre_region_masks
        post_region_masks_batch = post_region_masks
        use_region_attention_forward = (
            region_masks_batch is not None
            and self.enable_strong_region_prior
            and (not disable_all_priors)
        )
        if region_masks_batch is None and self.region_masks is not None:
            region_masks_batch = self.region_masks.unsqueeze(0).expand(
                batch_size, -1, -1, -1, -1
            )
            post_region_masks_batch = region_masks_batch
        if post_region_masks_batch is None:
            post_region_masks_batch = region_masks_batch

        need_branch_details = return_details or self.model_mode == 'prepost_prior_attention'
        if need_branch_details:
            img_feat, region_weights, region_details = self.dual_stream(
                pre_ct, post_ct, region_masks_batch, post_region_masks_batch, volumes,
                return_details=True,
                return_branch_features=self.model_mode == 'prepost_prior_attention',
                enable_region_prior=enable_region_prior and (not disable_all_priors),
                use_region_attention_forward=use_region_attention_forward,
            )
        else:
            img_feat, region_weights = self.dual_stream(
                pre_ct, post_ct, region_masks_batch, post_region_masks_batch, volumes,
                enable_region_prior=enable_region_prior and (not disable_all_priors),
                use_region_attention_forward=use_region_attention_forward,
            )
            region_details = None

        avg_feat = self.global_pool(img_feat).view(img_feat.size(0), -1)
        max_feat = self.max_pool(img_feat).view(img_feat.size(0), -1)

        combined_feat = torch.cat([avg_feat, max_feat], dim=1)

        image_logits = self.classifier(combined_feat)
        base_image_logits = image_logits
        prior_branch_details = None
        prior_logits = torch.zeros(pre_ct.size(0), 6, dtype=combined_feat.dtype, device=combined_feat.device)
        delta_logits = torch.zeros_like(prior_logits)
        pre_prior_logits = torch.zeros_like(prior_logits)
        post_prior_logits = torch.zeros_like(prior_logits)
        pre_attention_details = None
        post_attention_details = None

        if self.model_mode == 'residual_prior_branch' and (not disable_all_priors):
            if return_details:
                prior_logits, prior_branch_details = self.prior_branch(volumes, return_details=True)
            else:
                prior_logits = self.prior_branch(volumes, return_details=False)
            delta_logits = prior_logits
            image_logits = base_image_logits + delta_logits
        elif self.model_mode == 'prepost_prior_attention' and (not disable_all_priors):
            if volumes is None:
                raise ValueError('prepost_prior_attention  volumes ')
            if region_masks_batch is None or post_region_masks_batch is None:
                raise ValueError('prepost_prior_attention  region_masks')

            pre_scores = self.prior_branch.compute_raw_prior_no_std(
                volumes[:, :7], self.prior_branch.pre_beta, self.prior_branch.pre_threshold, self.prior_branch.pre_std
            )
            post_scores = self.prior_branch.compute_raw_prior_no_std(
                volumes[:, 7:14], self.prior_branch.post_beta, self.prior_branch.post_threshold, self.prior_branch.post_std
            )
            region_scores = pre_scores + post_scores
            if need_branch_details:
                branch_features = region_details.get('branch_features', {})
                pre_mid_feat = branch_features['pre_mid_feat']
                post_mid_feat = branch_features['post_mid_feat']
                pre_prior_logits, pre_attention_details = self.pre_prior_attention_head(
                    pre_mid_feat, region_masks_batch, pre_scores, return_details=True
                )
                post_prior_logits, post_attention_details = self.post_prior_attention_head(
                    post_mid_feat, post_region_masks_batch, post_scores, return_details=True
                )
                region_details.pop('branch_features', None)
                prior_branch_details = {
                    'pre_prior_scores': pre_scores,
                    'post_prior_scores': post_scores,
                    'region_scores': region_scores,
                    'prior_logits': pre_prior_logits + post_prior_logits,
                }
            else:
                raise RuntimeError('prepost_prior_attention  branch details ')
            delta_logits = pre_prior_logits + post_prior_logits
            prior_logits = delta_logits
            image_logits = base_image_logits + delta_logits

        out = image_logits

        if return_details:
            if prior_branch_details is None:
                prior_branch_details = {
                    'pre_prior_scores': torch.zeros(pre_ct.size(0), 7, dtype=combined_feat.dtype, device=combined_feat.device),
                    'post_prior_scores': torch.zeros(pre_ct.size(0), 7, dtype=combined_feat.dtype, device=combined_feat.device),
                    'region_scores': torch.zeros(pre_ct.size(0), 7, dtype=combined_feat.dtype, device=combined_feat.device),
                    'prior_logits': torch.zeros(pre_ct.size(0), 6, dtype=combined_feat.dtype, device=combined_feat.device),
                }
            details = {
                'region': region_details or {},
                'image_logits': base_image_logits,
                'prior_branch': prior_branch_details,
                'fusion': {
                    'prior_logits': prior_logits,
                    'delta_logits': delta_logits,
                    'pre_prior_logits': pre_prior_logits,
                    'post_prior_logits': post_prior_logits,
                },
                'pre_prior_attention': pre_attention_details or {},
                'post_prior_attention': post_attention_details or {},
            }
            return out, region_weights, details
        return out, region_weights

class CTDataset(Dataset):
    def __init__(
        self,
        df,
        ct_data_dir,
        transform=None,
        target_shape=(96, 96, 96),
        patient_template_root=None,
        use_region_attention=True,
    ):
        self.df = df.reset_index(drop=True)
        self.ct_data_dir = ct_data_dir
        self.transform = transform
        self.target_shape = target_shape
        self.patient_template_root = patient_template_root
        self.use_region_attention = use_region_attention
        self.flat_folder_mode = bool(self.df.attrs.get('flat_folder_mode', False))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = str(row['patient_id'])
        center = row.get('center', '')
        row_ct_data_dir = row.get('ct_data_dir', self.ct_data_dir)
        row_flat_folder_mode = bool(row.get('flat_folder_mode', self.flat_folder_mode))

        if row_flat_folder_mode:
            pre_ct_path = os.path.join(row_ct_data_dir, f"{patient_id}.nii.gz")
            post_ct_path = os.path.join(row_ct_data_dir, f"{patient_id}-1.nii.gz")
        else:
            pre_ct_path = os.path.join(row_ct_data_dir, center, f"{patient_id}.nii.gz")
            post_ct_path = os.path.join(row_ct_data_dir, center, f"{patient_id}-1.nii.gz")
        try:
            pre_ct = self.load_ct(pre_ct_path)
            post_ct = self.load_ct(post_ct_path)

            if pre_ct.min() < -0.1 or pre_ct.max() > 1.1:
                print(f"Warning: {patient_id} CT: [{pre_ct.min():.3f}, {pre_ct.max():.3f}]")
            if post_ct.min() < -0.1 or post_ct.max() > 1.1:
                print(f"Warning: {patient_id} CT: [{post_ct.min():.3f}, {post_ct.max():.3f}]")

        except Exception as e:
            print(f"Error: Loading {patient_id} (: {center}): {e}")
            pre_ct = torch.zeros(1, *self.target_shape)
            post_ct = torch.zeros(1, *self.target_shape)

        if self.transform:
            pre_ct, post_ct = self.transform(pre_ct, post_ct)

        label = torch.FloatTensor([row['y']])

        volumes = torch.FloatTensor([
            row.get('pre_SAH_ACA_Left_volume', 0.0) + row.get('pre_SAH_ACA_Right_volume', 0.0),
            row.get('pre_SAH_MCA_Left_volume', 0.0) + row.get('pre_SAH_MCA_Right_volume', 0.0),
            row.get('pre_SAH_PCA_Left_volume', 0.0) + row.get('pre_SAH_PCA_Right_volume', 0.0),
            row.get('pre_SAH_Brainstem_Left_volume', 0.0) + row.get('pre_SAH_Brainstem_Right_volume', 0.0),
            row.get('pre_SAH_Cerebellum_Left_volume', 0.0) + row.get('pre_SAH_Cerebellum_Right_volume', 0.0),
            row.get('pre_SAH_Cistern_volume', 0.0),
            row.get('pre_IVH_volume', 0.0),
            row.get('post_SAH_ACA_Left_volume', 0.0) + row.get('post_SAH_ACA_Right_volume', 0.0),
            row.get('post_SAH_MCA_Left_volume', 0.0) + row.get('post_SAH_MCA_Right_volume', 0.0),
            row.get('post_SAH_PCA_Left_volume', 0.0) + row.get('post_SAH_PCA_Right_volume', 0.0),
            row.get('post_SAH_Brainstem_Left_volume', 0.0) + row.get('post_SAH_Brainstem_Right_volume', 0.0),
            row.get('post_SAH_Cerebellum_Left_volume', 0.0) + row.get('post_SAH_Cerebellum_Right_volume', 0.0),
            row.get('post_SAH_Cistern_volume', 0.0),
            row.get('post_IVH_volume', 0.0),
        ])

        pre_region_masks, post_region_masks = self.load_patient_region_masks(patient_id)

        return pre_ct, post_ct, label, volumes, pre_region_masks, post_region_masks

    def load_ct(self, path):
        img = nib.load(path).get_fdata()

        if self.target_shape is not None:
            img = self.resize_3d(img, self.target_shape)

        img = img / 100.0

        img = torch.FloatTensor(img).unsqueeze(0)
        return img

    def resize_3d(self, img, target_shape):
        from scipy.ndimage import zoom

        current_shape = img.shape
        zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]

        resized = zoom(img, zoom_factors, order=1)

        return resized

    def load_patient_region_masks(self, patient_id):
        if not self.use_region_attention or not self.patient_template_root:
            empty = torch.zeros(7, *self.target_shape)
            return empty, empty

        patient_dir = os.path.join(self.patient_template_root, str(patient_id))
        pre_atlas_path = os.path.join(
            patient_dir, 'individualized_annotation_in_preop_mni_affine.nii.gz'
        )
        post_atlas_path = os.path.join(
            patient_dir, 'individualized_annotation_in_postop_mni_affine.nii.gz'
        )

        try:
            pre_masks = self.load_and_merge_patient_atlas(pre_atlas_path)
            post_masks = self.load_and_merge_patient_atlas(post_atlas_path)
            return pre_masks, post_masks
        except Exception as e:
            print(f"Warning: Loading {patient_id}: {e}")
            empty = torch.zeros(7, *self.target_shape)
            return empty, empty

    def load_and_merge_patient_atlas(self, atlas_path):
        atlas = nib.load(atlas_path).get_fdata()
        if self.target_shape is not None:
            atlas = self.resize_3d_nearest(atlas, self.target_shape)

        masks = np.zeros((7, *atlas.shape), dtype=np.float32)
        masks[0] = np.isin(atlas, [1, 2]).astype(np.float32)
        masks[1] = np.isin(atlas, [3, 4]).astype(np.float32)
        masks[2] = np.isin(atlas, [5, 6]).astype(np.float32)
        masks[3] = np.isin(atlas, [7, 8]).astype(np.float32)
        masks[4] = np.isin(atlas, [9, 10]).astype(np.float32)
        masks[5] = (atlas == 11).astype(np.float32)
        masks[6] = np.ones_like(atlas, dtype=np.float32)
        return torch.FloatTensor(masks)

    def resize_3d_nearest(self, img, target_shape):
        from scipy.ndimage import zoom

        current_shape = img.shape
        zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]
        return zoom(img, zoom_factors, order=0)

def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler=None,
    accumulation_steps=1,
    enable_region_prior=True,
    disable_all_priors=False,
):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch_idx, (pre_ct, post_ct, labels, volumes, pre_region_masks, post_region_masks) in enumerate(tqdm(loader, desc='Training')):
        pre_ct = pre_ct.to(device, non_blocking=True)
        post_ct = post_ct.to(device, non_blocking=True)
        labels_device = labels.to(device)
        volumes = volumes.to(device, non_blocking=True)
        pre_region_masks = pre_region_masks.to(device, non_blocking=True)
        post_region_masks = post_region_masks.to(device, non_blocking=True)

        ordinal_labels = torch.zeros(labels.size(0), 6, device=device)
        for i, label in enumerate(labels):
            ordinal_labels[i, :int(label.item())] = 1

        if scaler is not None:
            with autocast('cuda'):
                outputs, _ = model(
                    pre_ct, post_ct, volumes,
                    pre_region_masks=pre_region_masks,
                    post_region_masks=post_region_masks,
                    enable_region_prior=enable_region_prior,
                    disable_all_priors=disable_all_priors,
                )
                loss = criterion(outputs, ordinal_labels, labels_device.view(-1))
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs, _ = model(
                pre_ct, post_ct, volumes,
                pre_region_masks=pre_region_masks,
                post_region_masks=post_region_masks,
                enable_region_prior=enable_region_prior,
                disable_all_priors=disable_all_priors,
            )
            loss = criterion(outputs, ordinal_labels, labels_device.view(-1))
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        _, preds = decode_ordinal_predictions(outputs)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.view(-1).cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    if (batch_idx + 1) % accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    all_preds = np.asarray(all_preds, dtype=int)
    all_labels = np.asarray(all_labels, dtype=int)
    accuracy = (all_preds == all_labels).mean()
    mae = np.mean(np.abs(all_preds - all_labels))
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic', labels=[0,1,2,3,4,5,6])

    torch.cuda.empty_cache()
    gc.collect()

    return total_loss / len(loader), accuracy, mae, kappa

def _append_details_buffer(buffer, details):
    if details is None:
        return
    for section_name, section_value in details.items():
        if section_value is None:
            continue
        if isinstance(section_value, dict):
            section_buffer = buffer.setdefault(section_name, {})
            for key, value in section_value.items():
                if isinstance(value, dict):
                    nested_buffer = section_buffer.setdefault(key, {})
                    _append_details_buffer(nested_buffer, value)
                elif value is not None:
                    if isinstance(value, torch.Tensor):
                        stored_value = value.detach().cpu().numpy()
                    elif isinstance(value, np.ndarray):
                        stored_value = value
                    else:
                        stored_value = value
                    section_buffer.setdefault(key, []).append(stored_value)
        else:
            if isinstance(section_value, torch.Tensor):
                stored_value = section_value.detach().cpu().numpy()
            elif isinstance(section_value, np.ndarray):
                stored_value = section_value
            else:
                stored_value = section_value
            buffer.setdefault(section_name, []).append(stored_value)


def _merge_numpy_batches(values):
    first = values[0]
    if isinstance(first, str):
        return values
    if first.ndim == 1:
        return np.concatenate(values, axis=0)
    return np.vstack(values)


def _stack_detail_buffers(buffer):
    stacked = {}
    for key, value in buffer.items():
        if isinstance(value, dict):
            nested = _stack_detail_buffers(value)
            if nested:
                stacked[key] = nested
        elif value:
            stacked[key] = _merge_numpy_batches(value)
    return stacked


def _compute_change_region_scores(change_feat, region_masks):
    batch_size = change_feat.shape[0]
    region_masks = region_masks.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
    if region_masks.shape[2:] != change_feat.shape[2:]:
        region_masks = F.interpolate(region_masks, size=change_feat.shape[2:], mode='nearest')

    pooled_scores = []
    for idx in range(region_masks.shape[1]):
        mask = region_masks[:, idx:idx+1]
        masked_feat = change_feat * mask
        pooled = masked_feat.sum(dim=(2, 3, 4)) / (mask.sum(dim=(2, 3, 4)) + 1e-8)
        pooled_scores.append(pooled.mean(dim=1, keepdim=True))
    return torch.cat(pooled_scores, dim=1)


def _safe_geometric_score(metric_a, metric_b):
    product = float(metric_a) * float(metric_b)
    return max(product, 0.0) ** 0.5


def decode_ordinal_predictions(outputs, threshold=0.5):
    probs = torch.sigmoid(outputs)
    discrete_preds = (probs > threshold).sum(dim=1).long()
    return probs, discrete_preds


def validate(
    model,
    loader,
    criterion,
    device,
    collect_region_weights=False,
    enable_region_prior=True,
    disable_all_priors=False,
):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    all_probs = []
    all_region_weights = [] if collect_region_weights else None
    all_details = {} if collect_region_weights else None

    with torch.no_grad():
        for batch_idx, (pre_ct, post_ct, labels, volumes, pre_region_masks, post_region_masks) in enumerate(tqdm(loader, desc='Validation')):
            pre_ct = pre_ct.to(device)
            post_ct = post_ct.to(device)
            labels_device = labels.to(device)
            volumes = volumes.to(device)
            pre_region_masks = pre_region_masks.to(device)
            post_region_masks = post_region_masks.to(device)

            ordinal_labels = torch.zeros(labels.size(0), 6, device=device)
            for i, label in enumerate(labels):
                ordinal_labels[i, :int(label.item())] = 1

            with autocast('cuda'):
                if collect_region_weights:
                    outputs, region_weights, details = model(
                        pre_ct, post_ct, volumes,
                        pre_region_masks=pre_region_masks,
                        post_region_masks=post_region_masks,
                        return_details=True,
                        enable_region_prior=enable_region_prior,
                        disable_all_priors=disable_all_priors,
                    )
                    if (not disable_all_priors) and details.get('region', {}).get('change_feat') is not None:
                        details['region']['change_region_scores'] = _compute_change_region_scores(
                            details['region']['change_feat'],
                            (pre_region_masks + post_region_masks) / 2.0,
                        )
                    _append_details_buffer(all_details, details)
                else:
                    outputs, region_weights = model(
                        pre_ct, post_ct, volumes,
                        pre_region_masks=pre_region_masks,
                        post_region_masks=post_region_masks,
                        enable_region_prior=enable_region_prior,
                        disable_all_priors=disable_all_priors,
                    )
                loss = criterion(outputs, ordinal_labels, labels_device.view(-1))

            total_loss += loss.item()

            probs, preds = decode_ordinal_predictions(outputs)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            if collect_region_weights and region_weights is not None:
                all_region_weights.append({
                    key: value.cpu().numpy() if value is not None else None
                    for key, value in region_weights.items()
                })

            if (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)                 

    accuracy = (all_preds == all_labels).mean()
    all_preds = np.asarray(all_preds, dtype=int)
    all_labels = np.asarray(all_labels, dtype=int)
    mae = np.mean(np.abs(all_preds - all_labels))
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic', labels=[0,1,2,3,4,5,6])

    binary_labels = (all_labels >= 3).astype(int)

    binary_preds = (all_preds >= 3).astype(int)

    binary_probs = all_probs[:, 2]

    from sklearn.metrics import roc_auc_score, confusion_matrix

    binary_auc = roc_auc_score(binary_labels, binary_probs)
    binary_acc = (binary_preds == binary_labels).mean()

    tn, fp, fn, tp = confusion_matrix(binary_labels, binary_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    torch.cuda.empty_cache()
    gc.collect()

    if collect_region_weights and all_region_weights:
        stacked_region_weights = {}
        for branch_name in ['change']:
            branch_values = [item[branch_name] for item in all_region_weights if item.get(branch_name) is not None]
            stacked_region_weights[branch_name] = np.vstack(branch_values) if branch_values else None
        all_region_weights = stacked_region_weights
        all_details = _stack_detail_buffers(all_details)
        return (total_loss / len(loader), accuracy, mae, kappa,
                binary_auc, binary_acc, sensitivity, specificity, f1_score,
                all_preds, all_labels, all_probs, all_region_weights, all_details)
    else:
        return (total_loss / len(loader), accuracy, mae, kappa,
                binary_auc, binary_acc, sensitivity, specificity, f1_score,
                all_preds, all_labels, all_probs, None, None)

def prepare_data(csv_dir, ct_data_dir):
    center_map = {
        'yjs': 'yjs',
        'tl': 'tl',
        'fy': 'fy',
        'ay': 'ay',
        'th': 'output_th',
        'aq': 'output_aq'
    }

    train_files = {
        'featuresyjs.csv': 'yjs',
        'featurestl.csv': 'tl',
        'featuresfy.csv': 'fy',
        'featuresay.csv': 'ay',
        'featuresaq.csv': 'aq'
    }
    train_dfs = []

    for fname, center_key in train_files.items():
        df = pd.read_csv(os.path.join(csv_dir, fname))
        df['center'] = center_map[center_key]
        train_dfs.append(df)

    df_train = pd.concat(train_dfs, ignore_index=True)

    df_train['y'] = df_train['mRS'].astype(int)

    print(f"Train setsamples: {len(df_train)}")

    missing_count = 0
    for idx, row in df_train.iterrows():
        patient_id = str(row['patient_id'])
        center = row['center']
        pre_path = os.path.join(ct_data_dir, center, f"{patient_id}.nii.gz")
        post_path = os.path.join(ct_data_dir, center, f"{patient_id}-1.nii.gz")
        if not os.path.exists(pre_path) or not os.path.exists(post_path):
            missing_count += 1

    print(f"Train setCTsamples: {missing_count}/{len(df_train)}")

    return df_train


def load_test_set(csv_path, center_name=None, flat_folder_mode=False):
    df_test = pd.read_csv(csv_path)
    if center_name is not None:
        df_test['center'] = center_name
    if 'y' not in df_test.columns:
        df_test['y'] = df_test['mRS'].astype(int)
    df_test.attrs['flat_folder_mode'] = flat_folder_mode
    return df_test


def load_combined_test_set(test_cfgs):
    dfs = []
    for test_cfg in test_cfgs:
        df = load_test_set(
            csv_path=test_cfg['csv_path'],
            center_name=test_cfg.get('center'),
            flat_folder_mode=test_cfg.get('flat_folder_mode', False),
        )
        df['test_source'] = test_cfg['name']
        df['ct_data_dir'] = test_cfg['ct_data_dir']
        df['flat_folder_mode'] = bool(test_cfg.get('flat_folder_mode', False))
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined.attrs['flat_folder_mode'] = False
    return combined


def build_external_test_sets(config):
    test_registry = {
        'th': {
            'name': 'external_th',
            'csv_path': './data/csv/featuresth.csv',
            'ct_data_dir': './data/external/th',
            'center': 'th',
            'flat_folder_mode': True,
            'output_prefix': 'external_th',
        },
        'ay2': {
            'name': 'external_ay2',
            'csv_path': './data/csv/featuresay2.csv',
            'ct_data_dir': './data/external/ay2',
            'center': 'ay2',
            'flat_folder_mode': True,
            'output_prefix': 'external_ay2',
        },
        'efy': {
            'name': 'external_efy',
            'csv_path': './data/csv/featuresefy.csv',
            'ct_data_dir': './data/external/efy',
            'center': 'efy',
            'flat_folder_mode': True,
            'output_prefix': 'external_efy',
        },
    }

    test_mode = config.get('test_mode', 'combined')
    if test_mode == 'combined':
        return [
            {
                'name': 'Test-Combined',
                'combined': True,
                'members': [test_registry['th'], test_registry['ay2'], test_registry['efy']],
                'ct_data_dir': config['ct_data_dir'],
                'output_prefix': 'Test-Combined',
            }
        ]

    if test_mode == 'all_separate':
        return [test_registry['th'], test_registry['ay2'], test_registry['efy']]

    if test_mode not in test_registry:
        valid_modes = ['combined', 'all_separate'] + sorted(test_registry)
        raise ValueError(f"test_mode={test_mode} ,: {valid_modes}")

    return [test_registry[test_mode]]


def run_training_pipeline(config, df_train, df_val, df_test, device, run_name='main_model'):
    os.makedirs(config['save_dir'], exist_ok=True)
    test_name = config.get('current_test_name', 'test')
    test_output_prefix = config.get('current_test_output_prefix', 'test')

    print(f"\n{'#' * 70}")
    print(f"{run_name} Start")
    print(f"Output directory: {config['save_dir']}")
    print(f"Train set: {len(df_train)} samples (mRS: {df_train['y'].value_counts().sort_index().to_dict()})")
    print(f"Validation set: {len(df_val)} samples (mRS: {df_val['y'].value_counts().sort_index().to_dict()})")
    print(f"Test set[{test_name}]: {len(df_test)} samples (mRS: {df_test['y'].value_counts().sort_index().to_dict()})")
    print(f"{'#' * 70}")

    from sklearn.utils.class_weight import compute_class_weight
    train_labels = df_train['y'].values
    class_counts = np.bincount(train_labels, minlength=7)
    class_weights = 1.0 / np.sqrt(class_counts)

    if config['poor_outcome_weight_multiplier'] != 1.0:
        print(f"\nweight(mRS 3-6)x {config['poor_outcome_weight_multiplier']}")
        class_weights[3:] *= config['poor_outcome_weight_multiplier']
        class_weights = class_weights / class_weights.sum() * len(class_weights)

    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    print(f"\nClass weights:")
    for i, w in enumerate(class_weights):
        print(f"  mRS {i}: weight={w:.4f} (samples={class_counts[i]})")

    if config['use_augmentation']:
        print("\n(,)")
        train_transform = SimpleIntensityAugmentation(
            noise_std=config['noise_std'],
            brightness_range=config['brightness_range'],
            contrast_range=config['contrast_range']
        )
    else:
        train_transform = NoAugmentation()

    train_ct_data_dir = config.get('train_ct_data_dir', config['ct_data_dir'])

    train_dataset = CTDataset(df_train, train_ct_data_dir,
                             transform=train_transform,
                             target_shape=config['target_shape'],
                             patient_template_root=config.get('patient_template_root'),
                             use_region_attention=config['use_region_attention'])
    val_dataset = CTDataset(df_val, train_ct_data_dir,
                           transform=NoAugmentation(),
                           target_shape=config['target_shape'],
                           patient_template_root=config.get('patient_template_root'),
                           use_region_attention=config['use_region_attention'])
    test_dataset = CTDataset(df_test, config['ct_data_dir'],
                            transform=NoAugmentation(),
                            target_shape=config['target_shape'],
                            patient_template_root=config.get('patient_template_root'),
                            use_region_attention=config['use_region_attention'])

    dataloader_seed = config.get('seed', 42)
    train_generator = build_dataloader_generator(dataloader_seed)
    val_generator = build_dataloader_generator(dataloader_seed + 1)
    test_generator = build_dataloader_generator(dataloader_seed + 2)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             shuffle=True, num_workers=config['num_workers'],
                             pin_memory=True, drop_last=True,
                             worker_init_fn=seed_worker,
                             generator=train_generator,
                             persistent_workers=config['num_workers'] > 0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=config['num_workers'],
                           pin_memory=True,
                           worker_init_fn=seed_worker,
                           generator=val_generator,
                           persistent_workers=config['num_workers'] > 0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=config['num_workers'],
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=test_generator,
                            persistent_workers=config['num_workers'] > 0)
    train_export_loader = create_eval_loader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        generator_seed=dataloader_seed + 3,
    )

    model = DualChannelPredictor(
        dropout=config['dropout'],
        use_region_attention=config['use_region_attention'],
        region_masks=None,
        model_mode=config.get('model_mode', 'current_strong_prior'),
    )
    model = model.to(device)

    print(f"\nModel config: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    if config['use_focal_loss']:
        criterion = FocalOrdinalLoss(
            class_weights=class_weights_tensor,
            alpha=config['focal_loss_alpha'],
            gamma=config['focal_loss_gamma']
        )
        print(f"\nLoss function: Focal Loss")
        print(f"  gamma: {config['focal_loss_gamma']} (samples)")
        print(f"  alpha: {config['focal_loss_alpha']} ()")
    else:
        criterion = WeightedOrdinalLoss(class_weights=class_weights_tensor)
        print(f"\nLoss function: Weighted Ordinal Loss")

    print(f"  weight: {config['poor_outcome_weight_multiplier']}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])

    warmup_epochs = config.get('warmup_epochs', 5)
    num_epochs = config['num_epochs']
    disable_all_priors = config.get('disable_all_priors', False)

    min_lr = 1e-6
    start_factor = 0.01
    eta_min_factor = min_lr / config['learning_rate']

    def lr_lambda(epoch_idx):
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return start_factor + (1.0 - start_factor) * (epoch_idx / warmup_epochs)

        cosine_total = max(1, num_epochs - warmup_epochs)
        cosine_epoch = min(max(epoch_idx - warmup_epochs, 0), cosine_total)
        cosine_factor = 0.5 * (1.0 + np.cos(np.pi * cosine_epoch / cosine_total))
        return eta_min_factor + (1.0 - eta_min_factor) * cosine_factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    scaler = GradScaler('cuda') if config['use_amp'] and torch.cuda.is_available() else None

    start_epoch = 0
    best_test_kappa = 0
    checkpoint_path = os.path.join(config['save_dir'], 'checkpoint.pth')
    patience_counter = 0

    if config['resume'] and os.path.exists(checkpoint_path):
        print(f"\nResume from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        load_model_state_compat(model, checkpoint)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception:
            print("  Warning: scheduler state_dict,SkippingLoading")
        start_epoch = checkpoint['epoch']
        best_val_kappa = checkpoint.get('best_val_kappa', checkpoint.get('best_val_acc', 0))
        best_composite_score = checkpoint.get('best_composite_score', 0)
        best_kappa_recall = checkpoint.get('best_kappa_recall', 0)
        best_kappa_f1 = checkpoint.get('best_kappa_f1', 0)
        patience_counter = checkpoint.get('patience_counter', 0)
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(
            f"Resume from epoch {start_epoch} continue training,Validation setKappa: {best_val_kappa:.4f}, "
            f"composite score: {best_composite_score:.4f}, patience counter: {patience_counter}/{config['early_stopping_patience']}\n"
        )
    else:
        best_val_kappa = 0
        best_composite_score = 0
        best_kappa_recall = 0
        best_kappa_f1 = 0

    if config.get('test_only', False):
        print("\n[test_only ] SkippingTraining,LoadingTest...")
        start_epoch = config['num_epochs']

    history_csv_path = os.path.join(config['save_dir'], 'training_history.csv')
    if config['resume'] and os.path.exists(history_csv_path):
        history_df = pd.read_csv(history_csv_path)
        training_history = history_df.to_dict('list')
        print(f"LoadingTraining({len(history_df)} )")
        if config['resume'] and os.path.exists(checkpoint_path) and 'patience_counter' not in checkpoint:
            patience_counter = infer_patience_counter_from_history(training_history)
            print(
                f"checkpointSavedpatience counter,: "
                f"{patience_counter}/{config['early_stopping_patience']}"
            )
    else:
        training_history = {
            'epochs': [],
            'learning_rates': [],
            'train_loss': [],
            'train_acc': [],
            'train_mae': [],
            'train_kappa': [],
            'val_loss': [],
            'val_acc': [],
            'val_mae': [],
            'val_kappa': [],
            'val_binary_auc': [],
            'val_binary_acc': [],
            'val_sensitivity': [],
            'val_specificity': [],
            'val_f1': [],
            'val_composite_score': []
        }

    start_time = time.time()

    for epoch in range(start_epoch, config['num_epochs']):
        current_lr = optimizer.param_groups[0]['lr']
        enable_region_prior = True
        effective_region_prior = enable_region_prior and (not disable_all_priors)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['num_epochs']} | LR: {current_lr:.6f}")
        print(f"Disable all priors: {disable_all_priors}")
        print(f"Region prior enabled: {effective_region_prior}")
        print(f"Model mode: {config.get('model_mode', 'current_strong_prior')}")
        print("Region prior architecture: dual_branch_reconstruction")
        print("Region prior phase: both")
        print('='*60)

        train_loss, train_acc, train_mae, train_kappa = train_epoch(model, train_loader, criterion,
                                           optimizer, device, scaler, config['accumulation_steps'],
                                           enable_region_prior=enable_region_prior,
                                           disable_all_priors=disable_all_priors)
        (val_loss, val_acc, val_mae, val_kappa,
         val_binary_auc, val_binary_acc, val_sensitivity, val_specificity, val_f1,
         val_preds, val_labels, val_probs, _, _) = validate(
            model, val_loader, criterion, device,
            enable_region_prior=enable_region_prior,
            disable_all_priors=disable_all_priors,
        )

        val_composite_score = (val_kappa * val_sensitivity) ** 0.5

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr != old_lr:
            print(f"\nLearning rate changed: {old_lr:.6f} -> {new_lr:.6f}\n")

        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['train_mae'].append(train_mae)
        training_history['train_kappa'].append(train_kappa)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['val_mae'].append(val_mae)
        training_history['val_kappa'].append(val_kappa)
        training_history['val_binary_auc'].append(val_binary_auc)
        training_history['val_binary_acc'].append(val_binary_acc)
        training_history['val_sensitivity'].append(val_sensitivity)
        training_history['val_specificity'].append(val_specificity)
        training_history['val_f1'].append(val_f1)
        training_history['val_composite_score'].append(val_composite_score)
        training_history['epochs'].append(epoch + 1)
        training_history['learning_rates'].append(current_lr)

        gc.collect()
        torch.cuda.empty_cache()

        print(f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Train MAE={train_mae:.4f}, Train Kappa={train_kappa:.4f}")
        print(f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val MAE={val_mae:.4f}, Val Kappa={val_kappa:.4f}")
        print(f"Val Binary: AUC={val_binary_auc:.4f}, Acc={val_binary_acc:.4f}, Sens={val_sensitivity:.4f}, Spec={val_specificity:.4f}, F1={val_f1:.4f}")
        print(f"Val composite score (sqrt(KappaxRecall)): {val_composite_score:.4f}")

        def _save_ckpt(path, extra_keys=None):
            ckpt = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_kappa': best_val_kappa,
                'best_composite_score': best_composite_score,
                'best_kappa_recall': best_kappa_recall,
                'best_kappa_f1': best_kappa_f1,
                'patience_counter': patience_counter,
            }
            if extra_keys:
                ckpt.update(extra_keys)
            if scaler:
                ckpt['scaler_state_dict'] = scaler.state_dict()
            torch.save(ckpt, path)

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            patience_counter = 0
            print(f"Saved Saved best_kappa.pth (Kappa={val_kappa:.4f})")
            _save_ckpt(os.path.join(config['save_dir'], 'best_kappa.pth'))

            val_binary_labels_best = (val_labels >= 3).astype(int)
            val_binary_preds_best = (val_preds >= 3).astype(int)
            val_df = pd.DataFrame({
                'patient_id': df_val['patient_id'].values,
                'true_label': val_labels,
                'pred_label': val_preds,
                'binary_true': val_binary_labels_best,
                'binary_pred': val_binary_preds_best,
                'prob_poor_outcome': val_probs[:, 2],
            })
            for i in range(6):
                val_df[f'prob_mrs_gt_{i}'] = val_probs[:, i]
            val_df.to_csv(os.path.join(config['save_dir'], 'val_predictions_best_kappa.csv'), index=False)
        else:
            patience_counter += 1
            print(f"Warning Validation setKappa ({patience_counter}/{config['early_stopping_patience']})")

            if patience_counter >= config['early_stopping_patience']:
                print(f"\nEarly stopping triggered！{config['early_stopping_patience']}Kappa")
                print(f"Best Kappa: {best_val_kappa:.4f}")
                break

        val_kappa_recall = _safe_geometric_score(val_kappa, val_sensitivity)
        if val_kappa_recall > best_kappa_recall:
            best_kappa_recall = val_kappa_recall
            best_composite_score = val_kappa_recall
            print(f"Saved Saved best_kappa_recall.pth (sqrt(KappaxRecall)={val_kappa_recall:.4f})")
            _save_ckpt(os.path.join(config['save_dir'], 'best_kappa_recall.pth'))
            _val_bin_labels = (val_labels >= 3).astype(int)
            _val_bin_preds = (val_preds >= 3).astype(int)
            _val_df = pd.DataFrame({
                'patient_id': df_val['patient_id'].values,
                'true_label': val_labels,
                'pred_label': val_preds,
                'binary_true': _val_bin_labels,
                'binary_pred': _val_bin_preds,
                'prob_poor_outcome': val_probs[:, 2],
            })
            for _i in range(6):
                _val_df[f'prob_mrs_gt_{_i}'] = val_probs[:, _i]
            _val_df.to_csv(os.path.join(config['save_dir'], 'val_predictions_best_kappa_recall.csv'), index=False)

        val_kappa_f1 = _safe_geometric_score(val_kappa, val_f1)
        if val_kappa_f1 > best_kappa_f1:
            best_kappa_f1 = val_kappa_f1
            print(f"Saved Saved best_kappa_f1.pth (sqrt(KappaxF1)={val_kappa_f1:.4f})")
            _save_ckpt(os.path.join(config['save_dir'], 'best_kappa_f1.pth'))
            _val_bin_labels = (val_labels >= 3).astype(int)
            _val_bin_preds = (val_preds >= 3).astype(int)
            _val_df = pd.DataFrame({
                'patient_id': df_val['patient_id'].values,
                'true_label': val_labels,
                'pred_label': val_preds,
                'binary_true': _val_bin_labels,
                'binary_pred': _val_bin_preds,
                'prob_poor_outcome': val_probs[:, 2],
            })
            for _i in range(6):
                _val_df[f'prob_mrs_gt_{_i}'] = val_probs[:, _i]
            _val_df.to_csv(os.path.join(config['save_dir'], 'val_predictions_best_kappa_f1.csv'), index=False)

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_kappa': best_val_kappa,
            'best_composite_score': best_composite_score,
            'best_kappa_recall': best_kappa_recall,
            'best_kappa_f1': best_kappa_f1,
            'patience_counter': patience_counter,
        }
        if scaler:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        torch.save(checkpoint, checkpoint_path)

        history_df = pd.DataFrame(training_history)
        history_df.to_csv(os.path.join(config['save_dir'], 'training_history.csv'), index=False)

    if not config.get('test_only', False):
        print(f"\n{'='*60}")
        print(f"TrainingDone！Validation setKappa: {best_val_kappa:.4f}")
        print(f"Total training time: {(time.time() - start_time) / 60:.2f} ")
        print('='*60)

        history_df = pd.DataFrame(training_history)
        history_df.to_csv(os.path.join(config['save_dir'], 'training_history.csv'), index=False)
        print(f"\nTrainingSaved: {config['save_dir']}/training_history.csv")

        with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
            config_save = config.copy()
            config_save['target_shape'] = list(config_save['target_shape'])
            json.dump(config_save, f, indent=4)
        print(f"Saved: {config['save_dir']}/config.json")

    from sklearn.metrics import confusion_matrix, classification_report

    def _build_contribution_df(base_df, labels, preds, probs, region_weights, details):
        region_names = REGION_NAMES
        expected_rows = len(base_df)
        extra_columns = {}

        def _set_column(name, values):
            extra_columns[name] = values

        def _export_region_named_columns(df, prefix, values):
            if values is None:
                return
            values = normalize_detail_rows(values, expected_rows)
            if not isinstance(values, np.ndarray) or values.ndim != 2 or values.shape[0] != expected_rows:
                return
            for idx, region_name in enumerate(region_names):
                _set_column(f'{prefix}_{region_name}', values[:, idx])

        out_df = pd.DataFrame({
            'patient_id': base_df['patient_id'].values,
            'true_label': labels,
            'pred_label': preds,
            'binary_true': (labels >= 3).astype(int),
            'binary_pred': (preds >= 3).astype(int),
            'prob_poor_outcome': probs[:, 2],
        })
        for i in range(6):
            _set_column(f'prob_mrs_gt_{i}', probs[:, i])

        if region_weights is not None:
            if region_weights.get('change') is not None:
                for idx, region_name in enumerate(region_names):
                    _set_column(f'change_region_weight_{region_name}', region_weights['change'][:, idx])

        if details is not None:
            pre_region_details = details.get('region', {}).get('pre_branch', {})
            post_region_details = details.get('region', {}).get('post_branch', {})
            pre_change_region_details = details.get('region', {}).get('pre_change_branch', {})
            post_change_region_details = details.get('region', {}).get('post_change_branch', {})
            change_region_details = details.get('region', {}).get('change_branch', {})
            image_logits = details.get('image_logits')
            prior_branch_details = details.get('prior_branch', {})
            fusion_details = details.get('fusion', {})

            if image_logits is not None:
                image_logits = normalize_detail_rows(image_logits, expected_rows)
                for idx in range(image_logits.shape[1]):
                    _set_column(f'image_logit_{idx}', image_logits[:, idx])

            prior_logits = prior_branch_details.get('prior_logits')
            if prior_logits is not None:
                prior_logits = normalize_detail_rows(prior_logits, expected_rows)
                for idx in range(prior_logits.shape[1]):
                    _set_column(f'prior_logit_{idx}', prior_logits[:, idx])

            pre_prior_logits = fusion_details.get('pre_prior_logits')
            if pre_prior_logits is not None:
                pre_prior_logits = normalize_detail_rows(pre_prior_logits, expected_rows)
                for idx in range(pre_prior_logits.shape[1]):
                    _set_column(f'pre_prior_logit_{idx}', pre_prior_logits[:, idx])

            post_prior_logits = fusion_details.get('post_prior_logits')
            if post_prior_logits is not None:
                post_prior_logits = normalize_detail_rows(post_prior_logits, expected_rows)
                for idx in range(post_prior_logits.shape[1]):
                    _set_column(f'post_prior_logit_{idx}', post_prior_logits[:, idx])

            for prefix, key in [
                ('pre_prior_branch', 'pre_prior_scores'),
                ('post_prior_branch', 'post_prior_scores'),
                ('region_prior_branch', 'region_scores'),
            ]:
                _export_region_named_columns(out_df, prefix, prior_branch_details.get(key))

            delta_logits = fusion_details.get('delta_logits')
            if delta_logits is not None:
                delta_logits = normalize_detail_rows(delta_logits, expected_rows)
                for idx in range(delta_logits.shape[1]):
                    _set_column(f'delta_prior_logit_{idx}', delta_logits[:, idx])

            for branch_prefix, branch_details in [
                ('pre_prior_attention', details.get('pre_prior_attention', {})),
                ('post_prior_attention', details.get('post_prior_attention', {})),
            ]:
                _export_region_named_columns(out_df, f'{branch_prefix}_image_logit', branch_details.get('image_region_logits'))
                _export_region_named_columns(out_df, f'{branch_prefix}_prior_score', branch_details.get('prior_scores'))
                _export_region_named_columns(out_df, f'{branch_prefix}_attention_logit', branch_details.get('attention_logits'))
                _export_region_named_columns(out_df, f'{branch_prefix}_attention_weight', branch_details.get('attention_weights'))

            for branch_prefix, branch_details in [('pre_branch', pre_region_details), ('post_branch', post_region_details)]:
                region_response = branch_details.get('region_response')
                if region_response is not None:
                    for idx, region_name in enumerate(region_names):
                        _set_column(f'{branch_prefix}_response_{region_name}', region_response[:, idx])

            for branch_prefix, branch_details in [('pre_change_branch', pre_change_region_details), ('post_change_branch', post_change_region_details)]:
                prior_arch = branch_details.get('prior_arch')
                if prior_arch is not None:
                    arch_val = prior_arch[0] if isinstance(prior_arch, list) else prior_arch
                    _set_column(f'{branch_prefix}_prior_arch', [arch_val] * expected_rows)
                for prefix, key in [
                    ('image', 'image_logits'),
                    ('pre_prior_raw', 'pre_prior_raw'),
                    ('post_prior_raw', 'post_prior_raw'),
                    ('pre_prior', 'pre_prior_bias'),
                    ('post_prior', 'post_prior_bias'),
                    ('fused_prior', 'fused_prior_bias'),
                    ('attention', 'attention_logits'),
                ]:
                    value = branch_details.get(key)
                    if value is not None:
                        value = normalize_detail_rows(value, expected_rows)
                        if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[0] != expected_rows:
                            continue
                        for idx in range(value.shape[1]):
                            _set_column(f'{branch_prefix}_{prefix}_{idx}', value[:, idx])
                _export_region_named_columns(out_df, f'{branch_prefix}_image_logit', branch_details.get('image_logits'))
                _export_region_named_columns(out_df, f'{branch_prefix}_pre_prior_raw', branch_details.get('pre_prior_raw'))
                _export_region_named_columns(out_df, f'{branch_prefix}_post_prior_raw', branch_details.get('post_prior_raw'))
                _export_region_named_columns(out_df, f'{branch_prefix}_pre_prior', branch_details.get('pre_prior_bias'))
                _export_region_named_columns(out_df, f'{branch_prefix}_post_prior', branch_details.get('post_prior_bias'))
                _export_region_named_columns(out_df, f'{branch_prefix}_fused_prior', branch_details.get('fused_prior_bias'))
                _export_region_named_columns(out_df, f'{branch_prefix}_attention_logit', branch_details.get('attention_logits'))

            for branch_prefix, branch_details in [('change_branch', change_region_details)]:
                prior_arch = branch_details.get('prior_arch')
                if prior_arch is not None:
                    arch_val = prior_arch[0] if isinstance(prior_arch, list) else prior_arch
                    _set_column(f'{branch_prefix}_prior_arch', [arch_val] * expected_rows)
                for prefix, key in [
                    ('image', 'image_logits'),
                    ('pre_prior_raw', 'pre_prior_raw'),
                    ('post_prior_raw', 'post_prior_raw'),
                    ('pre_prior', 'pre_prior_bias'),
                    ('post_prior', 'post_prior_bias'),
                    ('fused_prior', 'fused_prior_bias'),
                    ('attention', 'attention_logits'),
                ]:
                    value = branch_details.get(key)
                    if value is not None:
                        value = normalize_detail_rows(value, expected_rows)
                        if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[0] != expected_rows:
                            continue
                        for idx in range(value.shape[1]):
                            _set_column(f'{branch_prefix}_{prefix}_{idx}', value[:, idx])
                _export_region_named_columns(out_df, f'{branch_prefix}_image_logit', branch_details.get('image_logits'))
                _export_region_named_columns(out_df, f'{branch_prefix}_pre_prior_raw', branch_details.get('pre_prior_raw'))
                _export_region_named_columns(out_df, f'{branch_prefix}_post_prior_raw', branch_details.get('post_prior_raw'))
                _export_region_named_columns(out_df, f'{branch_prefix}_pre_prior', branch_details.get('pre_prior_bias'))
                _export_region_named_columns(out_df, f'{branch_prefix}_post_prior', branch_details.get('post_prior_bias'))
                _export_region_named_columns(out_df, f'{branch_prefix}_fused_prior', branch_details.get('fused_prior_bias'))
                _export_region_named_columns(out_df, f'{branch_prefix}_attention_logit', branch_details.get('attention_logits'))

            fused_prior_pre = change_region_details.get('pre_prior_bias')
            fused_prior_post = change_region_details.get('post_prior_bias')
            if fused_prior_pre is not None:
                fused_prior_pre = normalize_detail_rows(fused_prior_pre, expected_rows)
                for idx, region_name in enumerate(region_names):
                    _set_column(f'pre_prior_{region_name}', fused_prior_pre[:, idx])
            if fused_prior_post is not None:
                fused_prior_post = normalize_detail_rows(fused_prior_post, expected_rows)
                for idx, region_name in enumerate(region_names):
                    _set_column(f'post_prior_{region_name}', fused_prior_post[:, idx])
            fused_prior_bias = change_region_details.get('fused_prior_bias')
            if fused_prior_bias is not None:
                fused_prior_bias = normalize_detail_rows(fused_prior_bias, expected_rows)
                for idx, region_name in enumerate(region_names):
                    _set_column(f'fused_prior_{region_name}', fused_prior_bias[:, idx])
            change_image_logits = change_region_details.get('image_logits')
            if change_image_logits is not None:
                change_image_logits = normalize_detail_rows(change_image_logits, expected_rows)
                for idx, region_name in enumerate(region_names):
                    _set_column(f'change_image_logit_{region_name}', change_image_logits[:, idx])
            change_attention_logits = change_region_details.get('attention_logits')
            if change_attention_logits is not None:
                change_attention_logits = normalize_detail_rows(change_attention_logits, expected_rows)
                for idx, region_name in enumerate(region_names):
                    _set_column(f'change_attention_logit_{region_name}', change_attention_logits[:, idx])

            change_scores = details.get('region', {}).get('change_region_scores')
            if change_scores is not None:
                change_scores = normalize_detail_rows(change_scores, expected_rows)
                for idx, region_name in enumerate(region_names):
                    _set_column(f'change_score_{region_name}', change_scores[:, idx])

        if extra_columns:
            out_df = pd.concat([out_df, pd.DataFrame(extra_columns, index=out_df.index)], axis=1)
        return out_df

    def _required_contribution_columns():
        required = [f'image_logit_{i}' for i in range(6)]
        model_mode = config.get('model_mode', 'current_strong_prior')
        if model_mode == 'prepost_prior_attention':
            required.extend([f'pre_prior_logit_{i}' for i in range(6)])
            required.extend([f'post_prior_logit_{i}' for i in range(6)])
            required.extend([f'delta_prior_logit_{i}' for i in range(6)])
        elif model_mode == 'residual_prior_branch':
            required.extend([f'delta_prior_logit_{i}' for i in range(6)])
        return required

    def _csv_has_required_columns(csv_path, required_columns):
        if not os.path.exists(csv_path):
            return False
        try:
            header_df = pd.read_csv(csv_path, nrows=1)
        except Exception:
            return False
        return all(col in header_df.columns for col in required_columns)

    def _run_val(model_file):
        ckpt_path = os.path.join(config['save_dir'], model_file)
        suffix = model_file.replace('.pth', '')
        out_csv = os.path.join(config['save_dir'], f'val_predictions_{suffix}.csv')
        out_detail_csv = os.path.join(config['save_dir'], f'val_contributions_{suffix}.csv')
        if not os.path.exists(ckpt_path):
            print(f"\n[SkippingValidation set] {model_file} not found")
            return
        required_columns = _required_contribution_columns()
        if (
            os.path.exists(out_csv)
            and _csv_has_required_columns(out_detail_csv, required_columns)
            and not config.get('test_only', False)
        ):
            return
        print(f"\nValidation set - {model_file}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        load_model_state_compat(model, ckpt)
        (_, _, _, _,
         _, _, _, _, _,
         v_preds, v_labels, v_probs, v_rw, v_details) = validate(
            model, val_loader, criterion, device,
            collect_region_weights=True,
            enable_region_prior=not disable_all_priors,
            disable_all_priors=disable_all_priors,
        )
        v_df = _build_contribution_df(df_val, v_labels, v_preds, v_probs, v_rw, v_details)
        v_df.to_csv(out_csv, index=False)
        v_df.to_csv(out_detail_csv, index=False)
        print(f"Validation setSaved: val_predictions_{suffix}.csv")
        print(f"Validation setSaved: val_contributions_{suffix}.csv")

    def _run_train(model_file):
        ckpt_path = os.path.join(config['save_dir'], model_file)
        suffix = model_file.replace('.pth', '')
        out_csv = os.path.join(config['save_dir'], f'train_predictions_{suffix}.csv')
        out_detail_csv = os.path.join(config['save_dir'], f'train_contributions_{suffix}.csv')
        if not os.path.exists(ckpt_path):
            print(f"\n[SkippingTrain set] {model_file} not found")
            return
        required_columns = _required_contribution_columns()
        if (
            os.path.exists(out_csv)
            and _csv_has_required_columns(out_detail_csv, required_columns)
            and not config.get('test_only', False)
        ):
            return
        print(f"\nTrain set - {model_file}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        load_model_state_compat(model, ckpt)
        (_, _, _, _,
         _, _, _, _, _,
         tr_preds, tr_labels, tr_probs, tr_rw, tr_details) = validate(
            model, train_export_loader, criterion, device,
            collect_region_weights=True,
            enable_region_prior=not disable_all_priors,
            disable_all_priors=disable_all_priors,
        )
        tr_df = _build_contribution_df(df_train, tr_labels, tr_preds, tr_probs, tr_rw, tr_details)
        tr_df.to_csv(out_csv, index=False)
        tr_df.to_csv(out_detail_csv, index=False)
        print(f"Train setSaved: train_predictions_{suffix}.csv")
        print(f"Train setSaved: train_contributions_{suffix}.csv")

    if not config.get('test_only', False):
        _run_train('best_kappa.pth')
        _run_train('best_kappa_recall.pth')
        _run_train('best_kappa_f1.pth')
        _run_val('best_kappa.pth')
        _run_val('best_kappa_recall.pth')
        _run_val('best_kappa_f1.pth')
    else:
        print("\n[test_only ] SkippingTrain set/Validation set,Test set.")

    def _run_test(model_file, label):
        ckpt_path = os.path.join(config['save_dir'], model_file)
        if not os.path.exists(ckpt_path):
            print(f"\n[Skipping] {model_file} not found")
            return
        print(f"\n{'='*60}")
        print(f"Test set - {label} ({model_file})")
        print(f"{'='*60}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        load_model_state_compat(model, ckpt)

        (t_loss, t_acc, t_mae, t_kappa,
         t_auc, t_bacc, t_sens, t_spec, t_f1,
         t_preds, t_labels, t_probs, t_rw, t_details) = validate(
            model, test_loader, criterion, device,
            collect_region_weights=True,
            enable_region_prior=not disable_all_priors,
            disable_all_priors=disable_all_priors,
        )

        print(f"Loss={t_loss:.4f}  Acc={t_acc:.4f}  MAE={t_mae:.4f}  Kappa={t_kappa:.4f}")
        print(f"AUC={t_auc:.4f}  Sens={t_sens:.4f}  Spec={t_spec:.4f}  F1={t_f1:.4f}")

        bin_labels = (t_labels >= 3).astype(int)
        bin_preds  = (t_preds >= 3).astype(int)
        print("\nBinary confusion matrix:")
        print(confusion_matrix(bin_labels, bin_preds))
        print(classification_report(bin_labels, bin_preds,
                                    target_names=['Good(0-2)', 'Poor(3-6)'], zero_division=0))

        suffix = model_file.replace('.pth', '')
        res_df = _build_contribution_df(df_test, t_labels, t_preds, t_probs, t_rw, t_details)
        pred_name = f'{test_output_prefix}_predictions_{suffix}.csv'
        detail_name = f'{test_output_prefix}_contributions_{suffix}.csv'
        res_df.to_csv(os.path.join(config['save_dir'], pred_name), index=False)
        res_df.to_csv(os.path.join(config['save_dir'], detail_name), index=False)
        print(f"Saved: {pred_name}")
        print(f"Saved: {detail_name}")

        return t_kappa, t_sens, t_f1, t_auc, t_preds, t_labels, t_probs, t_rw

    result_kappa        = _run_test('best_kappa.pth',        'Best Kappa')
    result_kappa_recall = _run_test('best_kappa_recall.pth', 'Best √(Kappa×Recall)')
    result_kappa_f1     = _run_test('best_kappa_f1.pth',     'Best √(Kappa×F1)')

    if result_kappa is not None:
        test_kappa, test_sensitivity, test_f1, test_binary_auc, test_preds, test_labels, test_probs, test_region_weights = result_kappa
        binary_labels = (test_labels >= 3).astype(int)
        binary_preds  = (test_preds >= 3).astype(int)
    else:
        test_region_weights = None

    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams['font.family'] = 'DejaVu Sans'

    model_specs = [
        ('best_kappa',        'Best Kappa',          'Blues'),
        ('best_kappa_recall', 'Best √(Kappa×Recall)', 'Greens'),
        ('best_kappa_f1',     'Best √(Kappa×F1)',    'Oranges'),
    ]

    for suffix, title, cmap_color in model_specs:
        csv_path = os.path.join(config['save_dir'], f'{test_output_prefix}_predictions_{suffix}.csv')
        val_csv_path = os.path.join(config['save_dir'], f'val_predictions_{suffix}.csv')
        if not os.path.exists(csv_path):
            print(f"[Skipping] {csv_path} not found")
            continue
        if not os.path.exists(val_csv_path):
            print(f"[Skipping] {val_csv_path} not found")
            continue

        res_df = pd.read_csv(csv_path)
        test_labels_cm = res_df['true_label'].values
        test_preds_cm  = res_df['pred_label'].values

        val_pred_df = pd.read_csv(val_csv_path)
        val_labels_cm = val_pred_df['true_label'].values
        val_preds_cm  = val_pred_df['pred_label'].values

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        cm_val = confusion_matrix(val_labels_cm, val_preds_cm, labels=list(range(7)))
        val_acc_cm = (val_labels_cm == val_preds_cm).mean()
        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Purples',
                    xticklabels=[f'mRS {i}' for i in range(7)],
                    yticklabels=[f'mRS {i}' for i in range(7)],
                    ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title(f'Validation Set (n={len(val_labels_cm)})\nAcc={val_acc_cm:.2%}',
                          fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=12)

        cm_test = confusion_matrix(test_labels_cm, test_preds_cm, labels=list(range(7)))
        test_acc_cm = (test_labels_cm == test_preds_cm).mean()
        sns.heatmap(cm_test, annot=True, fmt='d', cmap=cmap_color,
                    xticklabels=[f'mRS {i}' for i in range(7)],
                    yticklabels=[f'mRS {i}' for i in range(7)],
                    ax=axes[1], cbar_kws={'label': 'Count'})
        axes[1].set_title(f'{test_name} — {title} (n={len(test_labels_cm)})\nAcc={test_acc_cm:.2%}',
                          fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        axes[1].set_ylabel('True Label', fontsize=12)

        plt.tight_layout()
        out_path = os.path.join(config['save_dir'], f'{test_output_prefix}_confusion_matrices_{suffix}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: confusion_matrices_{suffix}.png")


def main():
    set_seed(42)

    config = get_config()
    device = torch.device(config['device'])

    print(f"Device: {device}")
    print(f"Mixed precision: {config['use_amp']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Accumulation steps: {config['accumulation_steps']}")
    print(f"Effective batch size: {config['batch_size'] * config['accumulation_steps']} (batch size)")
    print(f"Model config: {config}\n")

    df_train_full = prepare_data(config['csv_dir'], config['ct_data_dir'])

    val_quota = {
        0: 150,
        1: 41,
        2: 43,
        3: 43,
        4: 32,
        5: 28,
        6: 13,
    }
    train_quota = {
        0: 842,
        1: 231,
        2: 238,
        3: 236,
        4: 183,
        5: 156,
        6: 80,
    }

    train_parts = []
    val_parts = []
    print("\nmRSTraining/Validation set:")
    for label in sorted(val_quota.keys()):
        label_data = df_train_full[df_train_full['y'] == label].sample(
            frac=1.0,
            random_state=config.get('seed', 42),
        ).reset_index(drop=True)
        expected_total = train_quota[label] + val_quota[label]
        if len(label_data) < expected_total:
            raise ValueError(
                f"mRS={label} samples, {len(label_data)}, {expected_total}"
            )

        label_val = label_data.iloc[:val_quota[label]].copy()
        label_train = label_data.iloc[val_quota[label]:val_quota[label] + train_quota[label]].copy()

        if len(label_val) != val_quota[label] or len(label_train) != train_quota[label]:
            raise ValueError(f"mRS={label} ,.")

        train_parts.append(label_train)
        val_parts.append(label_val)

        print(
            f"  mRS={label}: Training{len(label_train)}, Validation{len(label_val)} "
            f"(Training {train_quota[label]}, Validation {val_quota[label]})"
        )

    df_train = pd.concat(train_parts, ignore_index=True)
    df_val = pd.concat(val_parts, ignore_index=True)

    print("\nFinal split summary:")
    print(f"  Train set: {len(df_train)}")
    print(f"  Validation set: {len(df_val)}")
    print(f"  Train setmRS: {df_train['y'].value_counts().sort_index().to_dict()}")
    print(f"  Validation setmRS: {df_val['y'].value_counts().sort_index().to_dict()}")

    test_sets = build_external_test_sets(config)

    base_save_dir = config['save_dir']
    base_ct_data_dir = config['ct_data_dir']

    for test_cfg in test_sets:
        run_config = dict(config)
        run_config['save_dir'] = test_cfg.get('save_dir', base_save_dir)
        run_config['ct_data_dir'] = test_cfg.get('ct_data_dir', base_ct_data_dir)
        run_config['current_test_name'] = test_cfg['name']
        run_config['current_test_output_prefix'] = test_cfg.get('output_prefix', test_cfg['name'])

        if test_cfg.get('combined', False):
            df_test = load_combined_test_set(test_cfg['members'])
        else:
            df_test = load_test_set(
                csv_path=test_cfg['csv_path'],
                center_name=test_cfg.get('center'),
                flat_folder_mode=test_cfg.get('flat_folder_mode', False),
            )
        run_training_pipeline(run_config, df_train, df_val, df_test, device, run_name=f"main_model-{test_cfg['name']}")


                                                                                 
                                                                                 
def get_config():
    config = {
        'batch_size': 4,
        'seed': 42,
        'accumulation_steps': 4,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'dropout': 0.6,
        'num_epochs': 100,

        'val_split': 0.15,
        'early_stopping_patience': 20,

        'use_amp': True,
        'num_workers': 12,
        'resume':True,
        'test_only': True,

        'use_augmentation': True,
        'noise_std': 0.04,
        'brightness_range': 0.4,
        'contrast_range': 0.4,

        'use_focal_loss': False,
        'focal_loss_gamma': 2.0,
        'focal_loss_alpha': 0.25,
        'poor_outcome_weight_multiplier': 1.0,

        'use_region_attention': True,
        'disable_all_priors': True,
        'model_mode': 'prepost_prior_attention',
        'space_mode': 'current',
        'region_prior_mode': 'current',

        'target_shape': (182, 218, 182),

        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './outputs/dl_models',
        'csv_dir': './data/csv',
        'ct_data_dir': './data/registered_ct',
        'patient_template_root': '../RegistrationAndSkullStripping/result_v3',
        'patient_space_prior_csv_dir': './prior_params',

        'test_mode': 'combined',
    }
    return config


if __name__ == '__main__':
    main()
