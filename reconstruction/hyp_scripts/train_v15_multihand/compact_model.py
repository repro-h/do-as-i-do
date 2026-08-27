"""Trainable local-token trajectory model over frozen compact Pi3X candidates."""

import math

import torch
import torch.nn as nn


class CompactMultiHandPi3XTrajectoryModel(nn.Module):
    def __init__(
        self,
        point_dim,
        metric_dim,
        token_dim=128,
        hidden_dim=192,
        heads=4,
        temporal_layers=2,
        dropout=0.1,
        max_window_size=64,
        max_xy_m=1.5,
        max_depth_m=2.5,
        initial_depth_m=0.85,
        translation_parameterization="ray_depth_uv",
        max_image_offset_fraction=0.15,
    ):
        super().__init__()
        if token_dim % heads:
            raise ValueError("token_dim must be divisible by heads")
        self.max_xy_m = float(max_xy_m)
        self.max_depth_m = float(max_depth_m)
        self.translation_parameterization = str(translation_parameterization)
        self.max_image_offset_fraction = float(max_image_offset_fraction)
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(point_dim), nn.Linear(point_dim, token_dim)
        )
        self.patch_encoder = nn.Sequential(
            nn.Linear(5, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim)
        )
        self.metric_encoder = nn.Sequential(
            nn.LayerNorm(metric_dim), nn.Linear(metric_dim, token_dim), nn.GELU()
        )
        self.joint_encoder = nn.Sequential(
            nn.Linear(6, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim)
        )
        self.joint_embedding = nn.Embedding(21, token_dim)
        self.missing_embedding = nn.Parameter(torch.zeros(token_dim))
        self.global_query = nn.Parameter(torch.randn(token_dim) * 0.02)
        self.local_attention = nn.MultiheadAttention(
            token_dim, heads, dropout=dropout, batch_first=True
        )
        self.global_attention = nn.MultiheadAttention(
            token_dim, heads, dropout=dropout, batch_first=True
        )
        self.local_norm = nn.LayerNorm(token_dim)
        self.global_norm = nn.LayerNorm(token_dim)
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(token_dim * 4 + 2),
            nn.Linear(token_dim * 4 + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_position = nn.Parameter(
            torch.randn(max_window_size, hidden_dim) * 0.01
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=temporal_layers)
        self.window_base = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.frame_residual = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self._initialize_heads(initial_depth_m)

    def _initialize_heads(self, initial_depth_m):
        nn.init.normal_(self.window_base[-1].weight, std=1e-3)
        nn.init.zeros_(self.window_base[-1].bias)
        ratio = min(max(initial_depth_m / self.max_depth_m, 1e-4), 1.0 - 1e-4)
        self.window_base[-1].bias.data[2] = math.log(ratio / (1.0 - ratio))
        nn.init.normal_(self.frame_residual[-1].weight, std=1e-3)
        nn.init.zeros_(self.frame_residual[-1].bias)

    def forward(self, batch):
        feature = batch["joint_patch_features"].float()
        patch_uv = batch["joint_patch_uv"].float()
        patch_confidence = batch["joint_patch_confidence"].float()
        patch_valid = batch["joint_patch_valid"]
        joint_uv = batch["joint_uv"]
        joint_valid = batch["joint_query_valid"]
        visibility = batch["joint_visibility"].clamp(0.0, 1.0)
        slot_valid = batch["hand_slot_valid"]
        batch_size, time, hands, joints, candidates, _ = feature.shape
        if joints != 21:
            raise ValueError(f"Expected 21 joints, got {joints}")
        if time > self.temporal_position.shape[0]:
            raise ValueError("Window exceeds max_window_size")

        root_uv = joint_uv[:, :, :, :1]
        local_uv = joint_uv - root_uv
        metadata = torch.cat((
            joint_uv, local_uv,
            joint_valid.to(joint_uv.dtype)[..., None],
            visibility[..., None],
        ), dim=-1)
        query = self.joint_encoder(metadata)
        joint_ids = torch.arange(joints, device=query.device).view(1, 1, 1, joints)
        identity = self.joint_embedding(joint_ids)
        query = query + identity

        query_expanded = joint_uv[..., None, :].expand(-1, -1, -1, -1, candidates, -1)
        patch_metadata = torch.cat((
            patch_uv,
            patch_uv - query_expanded,
            patch_confidence[..., None],
        ), dim=-1)
        key = self.point_encoder(feature) + self.patch_encoder(patch_metadata)
        flat = batch_size * time * hands * joints
        key = key.reshape(flat, candidates, -1)
        local_query = query.reshape(flat, 1, -1)
        padding = ~patch_valid.reshape(flat, candidates)
        all_padding = padding.all(dim=1)
        padding[all_padding, 0] = False
        attended, _ = self.local_attention(
            local_query, key, key, key_padding_mask=padding, need_weights=False
        )
        attended = self.local_norm(attended + local_query).reshape(
            batch_size, time, hands, joints, -1
        )
        reliability = joint_valid.to(joint_uv.dtype) * visibility
        missing = self.missing_embedding.view(1, 1, 1, 1, -1) + identity
        joint_tokens = (
            reliability[..., None] * attended
            + (1.0 - reliability[..., None]) * missing
        )

        global_feature = batch["global_features"].float()
        global_uv = batch["global_uv"].float()
        global_confidence = batch["global_confidence"].float()
        global_count = global_feature.shape[2]
        global_meta = torch.cat((
            global_uv[:, None].expand(-1, time, -1, -1),
            global_confidence[..., None],
            torch.zeros(
                batch_size, time, global_count, 2,
                dtype=global_feature.dtype, device=global_feature.device,
            ),
        ), dim=-1)
        global_key = self.point_encoder(global_feature) + self.patch_encoder(global_meta)
        global_key = global_key[:, :, None].expand(-1, -1, hands, -1, -1)
        global_key = global_key.reshape(batch_size * time * hands, global_count, -1)
        global_query = self.global_query.view(1, 1, -1).expand(
            batch_size * time * hands, 1, -1
        )
        global_token, _ = self.global_attention(
            global_query, global_key, global_key, need_weights=False
        )
        global_token = self.global_norm(global_token + global_query).reshape(
            batch_size, time, hands, -1
        )

        joint_weight = slot_valid[..., None].to(feature.dtype) * (
            0.1 + 0.9 * reliability
        )
        pooled = (joint_tokens * joint_weight[..., None]).sum(dim=3)
        pooled = pooled / joint_weight.sum(dim=3, keepdim=True).clamp_min(1.0)
        wrist = joint_tokens[:, :, :, 0]
        metric = self.metric_encoder(batch["metric_window_features"].float())
        metric = metric[:, None, None].expand(-1, time, hands, -1)
        observed_fraction = joint_valid.to(feature.dtype).mean(dim=3, keepdim=True)
        visible_fraction = visibility.mean(dim=3, keepdim=True)
        frame = self.frame_encoder(torch.cat((
            wrist, pooled, global_token, metric,
            observed_fraction, visible_fraction,
        ), dim=-1))

        frame = frame.permute(0, 2, 1, 3).reshape(batch_size * hands, time, -1)
        frame = frame + self.temporal_position[:time][None]
        temporal_valid = slot_valid.permute(0, 2, 1).reshape(batch_size * hands, time)
        temporal_padding = ~temporal_valid
        temporal_padding[temporal_padding.all(dim=1), 0] = False
        frame = self.temporal(frame, src_key_padding_mask=temporal_padding)
        weight = temporal_valid.to(frame.dtype)[..., None]
        window = (frame * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        base_raw = self.window_base(window)
        residual_raw = self.frame_residual(frame)
        residual_raw = residual_raw - (
            residual_raw * weight
        ).sum(dim=1, keepdim=True) / weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        raw = base_raw[:, None] + residual_raw
        depth = torch.sigmoid(raw[..., 2:3]) * self.max_depth_m
        if self.translation_parameterization == "direct_xyz":
            xy = torch.tanh(raw[..., :2]) * self.max_xy_m
            translation = torch.cat((xy, depth), dim=-1).reshape(
                batch_size, hands, time, 3
            ).permute(0, 2, 1, 3)
            return translation, None

        root_uv01 = (batch["ray_anchor_uv"] + 1.0) * 0.5
        image_wh = batch["image_wh"][:, :, None].expand(-1, -1, hands, -1)
        root_pixels = root_uv01 * (image_wh - 1.0).clamp_min(1.0)
        image_offset = torch.tanh(raw[..., :2]).reshape(
            batch_size, hands, time, 2
        ).permute(0, 2, 1, 3)
        predicted_pixels = (
            root_pixels + image_offset * image_wh * self.max_image_offset_fraction
        )
        depth = depth.reshape(batch_size, hands, time, 1).permute(0, 2, 1, 3)
        intrinsics = batch["intrinsics"][:, :, None]
        x = (
            (predicted_pixels[..., 0] - intrinsics[..., 0, 2])
            / intrinsics[..., 0, 0].clamp_min(1e-6) * depth[..., 0]
        )
        y = (
            (predicted_pixels[..., 1] - intrinsics[..., 1, 2])
            / intrinsics[..., 1, 1].clamp_min(1e-6) * depth[..., 0]
        )
        return torch.stack((x, y, depth[..., 0]), dim=-1), predicted_pixels
