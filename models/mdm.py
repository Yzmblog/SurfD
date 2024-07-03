import torch
import torch.nn as nn
import clip

import torch
import torch.nn as nn
from models.openaimodel import UNetModel

class MDM(nn.Module):

    def __init__(self, modeltype, num_actions, dropout=0.1, activation="gelu", legacy=False, dataset='deepfasion3d', clip_dim=512,
                 arch='OpenUNet', clip_version=None, **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.num_actions = num_actions

        self.dataset = dataset

        self.dropout = dropout

        self.activation = activation
        self.clip_dim = clip_dim

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch

        if self.arch == 'OpenUNet':
            num_classes=None
            if 'category' in self.cond_mode:
                num_classes = self.num_actions
            self.Unet = UNetModel(
                in_channels=1,
                model_channels=224,
                out_channels=1,
                num_res_blocks=2,
                attention_resolutions=[ 4, 2, 1 ],
                dropout=0,
                channel_mult=(1, 2, 4, 4),
                conv_resample=True,
                dims=1,
                num_classes=num_classes,
                use_checkpoint=True,
                use_fp16=False,
                num_heads=8,
                num_head_channels=-1,
                num_heads_upsample=-1,
                use_scale_shift_norm=False,
                resblock_updown=False,
                use_new_attention_order=False,
                use_spatial_transformer=False,    # custom transformer support
                transformer_depth=1,              # custom transformer support
                context_dim=clip_dim,                 # custom transformer support
                n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
                legacy=False,)

        self.clip_version = clip_version
        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                #self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')

                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)

            if 'sketch' in self.cond_mode:
                self.clip_version = clip_version

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, _ = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model


    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        texts = clip.tokenize(raw_text, truncate=True).to(device) 
        return self.clip_model.encode_text(texts).float()

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])


        if 'sketch' in self.cond_mode or 'img' in self.cond_mode:
            context = y['context']
            output = self.Unet(x, timesteps=timesteps, context=context)
        elif self.cond_mode == 'no_cond':
            output = self.Unet(x, timesteps=timesteps, y=None)
        elif 'text' in self.cond_mode:
            output = self.Unet(x, timesteps=timesteps, context=enc_text)
        else:
            output = self.Unet(x, timesteps=timesteps, y=y['action_text'])

        return output

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)


