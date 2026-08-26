"""Shared-scene, side-free multi-hand Pi3X trajectory model."""

import math

import torch
import torch.nn as nn


class MultiHandPi3XTrajectoryModel(nn.Module):
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
        spatial_bias=6.0,
        max_xy_m=1.5,
        max_depth_m=2.5,
        initial_depth_m=0.85,
        translation_parameterization="ray_depth_uv",
        max_image_offset_fraction=0.15,
    ):
        super().__init__()
        if token_dim % heads:
            raise ValueError("token_dim must be divisible by heads")
        self.heads = int(heads)
        self.spatial_bias = float(spatial_bias)
        self.max_xy_m = float(max_xy_m)
        self.max_depth_m = float(max_depth_m)
        self.translation_parameterization = str(translation_parameterization)
        self.max_image_offset_fraction = float(max_image_offset_fraction)
        if self.translation_parameterization not in ("direct_xyz", "ray_depth_uv"):
            raise ValueError(
                f"Unknown translation parameterization: {self.translation_parameterization}"
            )
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(point_dim), nn.Linear(point_dim, token_dim)
        )
        self.grid_encoder = nn.Sequential(
            nn.Linear(3, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim)
        )
        self.metric_encoder = nn.Sequential(
            nn.LayerNorm(metric_dim), nn.Linear(metric_dim, token_dim), nn.GELU()
        )
        self.joint_encoder = nn.Sequential(
            nn.Linear(6, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim)
        )
        self.joint_embedding = nn.Embedding(21, token_dim)
        self.missing_embedding = nn.Parameter(torch.zeros(token_dim))
        self.hand_global_query = nn.Parameter(torch.randn(token_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            token_dim, heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(token_dim)
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
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
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
        point = batch["point_features"]
        grid_uv = batch["grid_uv"]
        confidence = batch["grid_confidence"]
        joint_uv = batch["joint_uv"]
        joint_valid = batch["joint_query_valid"]
        visibility = batch["joint_visibility"].clamp(0.0, 1.0)
        slot_valid = batch["hand_slot_valid"]
        batch_size, time, height, width, _ = point.shape
        hands, joints = joint_uv.shape[2:4]
        if joints != 21:
            raise ValueError(f"Expected 21 joints, got {joints}")
        if time > self.temporal_position.shape[0]:
            raise ValueError("Window exceeds max_window_size")

        scene_key = self.point_encoder(
            point.reshape(batch_size * time, height * width, -1)
        )
        flat_grid_uv = grid_uv.reshape(batch_size * time, height * width, 2)
        scene_key = scene_key + self.grid_encoder(torch.cat((
            flat_grid_uv,
            confidence.reshape(batch_size * time, height * width, 1),
        ), dim=-1))
        # Scene features are encoded once, then shared by every detected hand.
        scene_key = scene_key[:, None].expand(-1, hands, -1, -1).reshape(
            batch_size * time * hands, height * width, -1
        )

        root_uv = joint_uv[:, :, :, :1]
        local_uv = joint_uv - root_uv
        metadata = torch.cat((
            joint_uv,
            local_uv,
            joint_valid.to(joint_uv.dtype)[..., None],
            visibility[..., None],
        ), dim=-1)
        observed_query = self.joint_encoder(metadata)
        joint_ids = torch.arange(
            joints, device=observed_query.device
        ).view(1, 1, 1, joints)
        joint_identity = self.joint_embedding(joint_ids)
        observed_query = observed_query + joint_identity
        missing_query = (
            self.missing_embedding.view(1, 1, 1, 1, -1)
            + joint_identity
        )
        reliability = joint_valid.to(joint_uv.dtype) * visibility
        query = (
            reliability[..., None] * observed_query
            + (1.0 - reliability[..., None]) * missing_query
        )
        global_query = self.hand_global_query.view(1, 1, 1, 1, -1).expand(
            batch_size, time, hands, 1, -1
        )
        query = torch.cat((query, global_query), dim=3).reshape(
            batch_size * time * hands, joints + 1, -1
        )

        repeated_grid_uv = flat_grid_uv[:, None].expand(
            -1, hands, -1, -1
        ).reshape(batch_size * time * hands, height * width, 2)
        distance = (
            joint_uv.reshape(batch_size * time * hands, joints, 1, 2)
            - repeated_grid_uv[:, None]
        ).square().sum(dim=-1)
        joint_bias = (
            -self.spatial_bias
            * distance
            * reliability.reshape(batch_size * time * hands, joints, 1)
        )
        global_bias = torch.zeros(
            batch_size * time * hands, 1, height * width,
            dtype=joint_bias.dtype, device=joint_bias.device,
        )
        attention_bias = torch.cat((joint_bias, global_bias), dim=1)
        attention_bias = attention_bias.repeat_interleave(self.heads, dim=0)
        key_padding = ~batch["grid_valid"].reshape(
            batch_size * time, height * width
        )
        key_padding = key_padding[:, None].expand(-1, hands, -1).reshape(
            batch_size * time * hands, height * width
        )
        all_invalid = key_padding.all(dim=1)
        key_padding[all_invalid, 0] = False
        attended, _ = self.cross_attention(
            query, scene_key, scene_key,
            key_padding_mask=key_padding,
            attn_mask=attention_bias,
            need_weights=False,
        )
        attended = self.cross_norm(attended + query).reshape(
            batch_size, time, hands, joints + 1, -1
        )
        joint_tokens = attended[:, :, :, :joints]
        global_token = attended[:, :, :, -1]
        # Occluded joints retain a small contribution; they are not hard-deleted.
        joint_weight = (
            slot_valid[..., None].to(point.dtype)
            * (0.1 + 0.9 * reliability)
        )
        pooled = (joint_tokens * joint_weight[..., None]).sum(dim=3)
        pooled = pooled / joint_weight.sum(dim=3, keepdim=True).clamp_min(1.0)
        wrist = joint_tokens[:, :, :, 0]
        metric = self.metric_encoder(batch["metric_window_features"])
        metric = metric[:, None, None].expand(-1, time, hands, -1)
        observed_fraction = joint_valid.to(point.dtype).mean(dim=3, keepdim=True)
        visible_fraction = visibility.mean(dim=3, keepdim=True)
        frame = self.frame_encoder(torch.cat((
            wrist, pooled, global_token, metric,
            observed_fraction, visible_fraction,
        ), dim=-1))

        frame = frame.permute(0, 2, 1, 3).reshape(batch_size * hands, time, -1)
        frame = frame + self.temporal_position[:time][None]
        temporal_valid = slot_valid.permute(0, 2, 1).reshape(batch_size * hands, time)
        padding = ~temporal_valid
        all_padding = padding.all(dim=1)
        padding[all_padding, 0] = False
        frame = self.temporal(frame, src_key_padding_mask=padding)
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
            translation = torch.cat((xy, depth), dim=-1)
            translation = translation.reshape(
                batch_size, hands, time, 3
            ).permute(0, 2, 1, 3)
            return translation, None

        # Missing detector frames use a temporally propagated ray anchor rather
        # than leaking that frame's GT wrist location into output composition.
        root_uv01 = (batch["ray_anchor_uv"] + 1.0) * 0.5
        image_wh = batch["image_wh"][:, :, None].expand(-1, -1, hands, -1)
        root_pixels = root_uv01 * (image_wh - 1.0).clamp_min(1.0)
        image_offset = torch.tanh(raw[..., :2]).reshape(
            batch_size, hands, time, 2
        ).permute(0, 2, 1, 3)
        image_offset = image_offset * image_wh * self.max_image_offset_fraction
        predicted_pixels = root_pixels + image_offset
        depth = depth.reshape(batch_size, hands, time, 1).permute(0, 2, 1, 3)
        intrinsics = batch["intrinsics"][:, :, None]
        x = (
            (predicted_pixels[..., 0] - intrinsics[..., 0, 2])
            / intrinsics[..., 0, 0].clamp_min(1e-6)
            * depth[..., 0]
        )
        y = (
            (predicted_pixels[..., 1] - intrinsics[..., 1, 2])
            / intrinsics[..., 1, 1].clamp_min(1e-6)
            * depth[..., 0]
        )
        translation = torch.stack((x, y, depth[..., 0]), dim=-1)
        return translation, predicted_pixels
