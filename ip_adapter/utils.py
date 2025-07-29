from typing import List

import torch
import torch.nn.functional as F
from diffusers.models.adapter import AdapterBlock
from torch import nn

from ip_adapter.resampler import Resampler, ResamplerV2


def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")


if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


class IPAdapterTrainer(torch.nn.Module):
    """IPAdapterTrainer"""

    def __init__(self, unet, image_proj_model, adapter_modules, t2i_adapter=None, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj = image_proj_model
        self.ip_adapter = adapter_modules
        self.t2i_adapter = t2i_adapter

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, unet, noisy_latents, timesteps, encoder_hidden_states_caption, encoder_hidden_states_triplets, image_embeds, image_embeds2=None,
                unet_added_cond_kwargs=None):
        ip_tokens = self.image_proj(image_embeds)
        if encoder_hidden_states_triplets is not None:
            encoder_hidden_states = torch.cat([encoder_hidden_states_caption, encoder_hidden_states_triplets, ip_tokens], dim=1)
        else:
            encoder_hidden_states = torch.cat([encoder_hidden_states_caption, ip_tokens], dim=1)
        down_block_additional_residuals = None
        if image_embeds2 is not None and self.t2i_adapter is not None:
            down_block_additional_residuals = self.t2i_adapter(image_embeds2)
        # Predict the noise residual and compute loss
        kwargs = {
            "down_block_additional_residuals": down_block_additional_residuals
        }
        if unet_added_cond_kwargs is not None:
            kwargs["unet_added_cond_kwargs"] = unet_added_cond_kwargs

        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states, **kwargs).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        if self.t2i_adapter is not None:
            orig_t2i_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.t2i_adapter.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)
        if self.t2i_adapter is not None:
            if "t2i_adapter" in state_dict:
                self.t2i_adapter.load_state_dict(state_dict["t2i_adapter"], strict=True)
            else:
                print("Warning: t2i_adapter not found in checkpoint, skipping loading.")

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        if self.t2i_adapter is not None:
            new_t2i_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.t2i_adapter.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"
        if self.t2i_adapter is not None:
            assert orig_t2i_adapter_sum != new_t2i_adapter_sum, "Weights of t2i_adapter did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


class FacerAdapter(nn.Module):
    def __init__(
            self,
            in_channels: int = 64,
            channels=None,
            num_res_blocks: int = 2,
            downscale_factor: int = 1,
            downs=None,
    ):
        super().__init__()

        if channels is None:
            channels = [320, 640, 1280, 1280]

        if downs is None:
            downs = [True, True, True, False]

        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=1, padding=0)

        self.body = nn.ModuleList(
            [
                AdapterBlock(channels[0], channels[0], num_res_blocks),
                *[
                    AdapterBlock(channels[i - 1], channels[i], num_res_blocks,
                                 down=False if downs is None else downs[i - 1])
                    for i in range(1, len(channels))
                ],
            ]
        )

        self.total_downscale_factor = downscale_factor * 2 ** (len(channels) - 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv_in(x)

        features = []

        for block in self.body:
            x = block(x)
            features.append(x)

        return features


def init_ip_adapter(unet, image_encoder, num_tokens=16, t2i_adapter=None, XL=False, ckpt_path=None, usev2=False):
    if XL:
        num_tokens = 4
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_encoder.config.hidden_size,
            output_dim=unet.config.cross_attention_dim,
            ff_mult=4
        )
    else:
        num_tokens = 16 if num_tokens is None else 32
        resampler_class = Resampler if not usev2 else ResamplerV2
        image_proj_model = resampler_class(
            dim=unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12 if not usev2 else 16,
            num_queries=num_tokens,
            embedding_dim=image_encoder.config.hidden_size,
            output_dim=unet.config.cross_attention_dim,
            ff_mult=4
        )
    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    hidden_size = 1
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                               num_tokens=num_tokens)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    return IPAdapterTrainer(unet, image_proj_model, adapter_modules, t2i_adapter=t2i_adapter, ckpt_path=ckpt_path)
