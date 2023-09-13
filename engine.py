import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import clip

from combiner import Combiner
from textualInversion import TextualInversion


CLIP_IMG_ENC_OUTPUT_DIM_BEFORE_PROJ = 1024


class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, drop_probs):
        super(LinearProjection, self).__init__()
        # Define trainable projection layers
        map_layers = [nn.Linear(input_dim, output_dim),
                      nn.Dropout(p=drop_probs[0])]

        for _ in range(1, num_layers):
            map_layers.extend(
                [nn.ReLU(), nn.Linear(output_dim, output_dim), nn.Dropout(p=drop_probs[0])])

        self.proj = nn.Sequential(*map_layers)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def forward(self, x):
        return self.proj(x)


class HateClassifier(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        # self.save_hyperparameters(logger=False)
        self.dataset = args.dataset
        self.num_mapping_layers = args.num_mapping_layers
        self.map_dim = args.map_dim
        self.fusion = args.fusion
        self.num_pre_output_layers = args.num_pre_output_layers
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size

        self.name = args.name
        self.fast_process = args.fast_process

        self.proj_map = args.proj_map

        self.pretrained_proj = args.pretrained_proj_weights
        self.freeze_proj = args.freeze_proj_layers

        self.convex_tensor = args.convex_tensor
        self.comb_proj = args.comb_proj
        self.comb_fusion = args.comb_fusion

        self.enh_text = args.enh_text
        self.phi_freeze = args.phi_freeze
        self.text_inv_proj = args.text_inv_proj
        self.phi_inv_proj = args.phi_inv_proj
        self.post_inv_proj = args.post_inv_proj

        self.acc = torchmetrics.Accuracy(task='binary')
        self.auroc = torchmetrics.AUROC(task='binary')

        self.pretrained_weights_path = f'./resources/pretrained_weights/{self.dataset}'

        # load pre-trained CLIP model
        self.clip_model, _ = clip.load("ViT-L/14", device="cuda", jit=False)

        # remove CLIP image encoder projection (textual projection must be computed again without projection product)
        self.clip_model.visual.proj = None

        # set CLIP model to float32 type
        self.clip_model.float()

        # freeze CLIP weights
        for _, p in self.clip_model.named_parameters():
            p.requires_grad_(False)

        self.image_map = LinearProjection(CLIP_IMG_ENC_OUTPUT_DIM_BEFORE_PROJ, self.map_dim,
                                          self.num_mapping_layers, args.drop_probs)
        self.text_map = LinearProjection(self.clip_model.token_embedding.embedding_dim, self.map_dim,
                                         self.num_mapping_layers, args.drop_probs)

        if self.name in ['hate-clipper', 'adaptation']:
            if args.fusion == 'align':
                pre_output_input_dim = self.map_dim
            elif args.fusion == 'concat':
                pre_output_input_dim = self.map_dim * 2
        elif self.name == 'text-only':
            if self.proj_map:
                pre_output_input_dim = self.map_dim
            else:
                pre_output_input_dim = self.clip_model.token_embedding.embedding_dim
        elif self.name == 'image-only':
            if self.proj_map:
                pre_output_input_dim = self.map_dim
            else:
                pre_output_input_dim = CLIP_IMG_ENC_OUTPUT_DIM_BEFORE_PROJ

        elif self.name == 'sum':
            # proj_map is used by default
            pre_output_input_dim = self.map_dim

        elif self.name == 'combiner':
            # proj_map is used by default
            self.comb = Combiner(self.convex_tensor, self.map_dim, self.comb_proj, self.comb_fusion)

            if self.pretrained_proj:
                # Load pre-trained weights
                assert self.num_mapping_layers == 1
                if self.dataset == 'hmc':
                    assert self.map_dim in [1024, 768]
                    weights = f'hmc_{self.map_dim}_projection_embeddings.pt'
                elif self.dataset == 'harmeme':
                    assert self.map_dim == 768
                    weights = f'harmeme_{self.map_dim}_projection_embeddings.pt'
                else:
                    raise ValueError()

                state_dict = torch.load(f'{self.pretrained_weights_path}/{weights}')['state_dict']

                with torch.no_grad():
                    self.image_map.proj[0].weight.copy_(state_dict['image_proj_weight'])
                    self.image_map.proj[0].bias.copy_(state_dict['image_proj_bias'])
                    self.text_map.proj[0].weight.copy_(state_dict['text_proj_weight'])
                    self.text_map.proj[0].bias.copy_(state_dict['text_proj_bias'])

                if self.freeze_proj:
                    # freeze projection layers
                    for name, p in self.image_map.named_parameters():
                        p.requires_grad_(False)
                    for name, p in self.text_map.named_parameters():
                        p.requires_grad_(False)

            pre_output_input_dim = self.map_dim

        elif self.name == 'text-inv':
            self.text_inv = TextualInversion(self.clip_model, CLIP_IMG_ENC_OUTPUT_DIM_BEFORE_PROJ, self.phi_inv_proj,
                                             self.text_inv_proj, self.post_inv_proj, args.drop_probs, self.phi_freeze,
                                             self.enh_text, self.map_dim, self.num_mapping_layers)

            pre_output_input_dim = self.text_inv.output_dim

        elif self.name == 'text-inv-fusion':
            self.text_inv = TextualInversion(self.clip_model, CLIP_IMG_ENC_OUTPUT_DIM_BEFORE_PROJ, self.phi_inv_proj,
                                             self.text_inv_proj, self.post_inv_proj, args.drop_probs, self.phi_freeze,
                                             self.enh_text, self.map_dim, self.num_mapping_layers)

            self.image_map = LinearProjection(CLIP_IMG_ENC_OUTPUT_DIM_BEFORE_PROJ, self.map_dim,
                                              self.num_mapping_layers, args.drop_probs)

            pre_output_input_dim = self.text_inv.output_dim

        elif self.name == 'text-inv-comb':
            self.comb = Combiner(self.convex_tensor, self.map_dim, self.comb_proj, self.comb_fusion)

            self.text_inv = TextualInversion(self.clip_model, CLIP_IMG_ENC_OUTPUT_DIM_BEFORE_PROJ, self.phi_inv_proj,
                                             self.text_inv_proj, self.post_inv_proj, args.drop_probs, self.phi_freeze,
                                             self.enh_text, self.map_dim, self.num_mapping_layers)

            self.image_map = LinearProjection(CLIP_IMG_ENC_OUTPUT_DIM_BEFORE_PROJ, self.map_dim,
                                              self.num_mapping_layers, args.drop_probs)

            if self.fusion == 'align':
                pre_output_input_dim = self.map_dim
            elif self.fusion == 'concat':
                pre_output_input_dim = 2 * self.map_dim
            else:
                raise ValueError()

            if self.pretrained_proj:
                assert self.num_mapping_layers == 1

                if self.dataset == 'hmc':
                    assert self.map_dim in [1024, 768]
                    weights = f'hmc_{self.map_dim}_projection_embeddings.pt'
                    weights_768 = f'hmc_768_projection_embeddings.pt'

                elif self.dataset == 'harmeme':
                    assert self.map_dim == 768
                    weights_768 = f'harmeme_{self.map_dim}_projection_embeddings.pt'
                    weights = weights_768

                else:
                    raise ValueError()

                state_dict = torch.load(f'{self.pretrained_weights_path}/{weights}')['state_dict']
                state_dict_768 = torch.load(f'{self.pretrained_weights_path}/{weights_768}')['state_dict']

                with torch.no_grad():
                    self.image_map.proj[0].weight.copy_(state_dict['image_proj_weight'])
                    self.image_map.proj[0].bias.copy_(state_dict['image_proj_bias'])
                    self.text_inv.pre_inversion_map[0].weight.copy_(state_dict_768['image_proj_weight'])
                    self.text_inv.pre_inversion_map[0].bias.copy_(state_dict_768['image_proj_bias'])

                if self.freeze_proj:
                    # freeze projection layers
                    for name, p in self.image_map.proj.named_parameters():
                        p.requires_grad_(False)
                    for name, p in self.text_inv.pre_inversion_map.named_parameters():
                        p.requires_grad_(False)
        else:
            raise ValueError()

        pre_output_layers = [nn.Dropout(p=args.drop_probs[1])]
        output_input_dim = pre_output_input_dim

        if self.num_pre_output_layers >= 1:
            pre_output_layers.extend(
                [nn.Linear(pre_output_input_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])
            output_input_dim = self.map_dim

        for _ in range(1, self.num_pre_output_layers):
            pre_output_layers.extend(
                [nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])

        self.pre_output = nn.Sequential(*pre_output_layers)
        self.output = nn.Linear(output_input_dim, 1)

        self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, batch):
        pass

    def compute_CLIP_features_without_proj(self, clip_model, img_input, text_input):
        # CLIP image encoder projection is disabled in the init method
        image_features = clip_model.visual(img_input.type(clip_model.dtype))

        # Compute CLIP text encoder output without the textual projection
        x = clip_model.token_embedding(text_input).type(clip_model.dtype)
        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = clip_model.ln_final(x).type(clip_model.dtype)
        text_features = x[torch.arange(x.shape[0]), text_input.argmax(dim=-1)]

        return image_features, text_features

    def common_step(self, batch):
        if self.fast_process:
            image_features = batch['images']
            text_features = batch['texts']
        else:
            image_features, text_features = self.compute_CLIP_features_without_proj(self.clip_model,
                                                                                    batch['pixel_values'],
                                                                                    batch['texts'])
        if self.enh_text:
            prompt = batch['enhanced_texts']
        else:
            prompt = batch['simple_prompt']

        output = {}

        if self.name in ['hate-clipper', 'adaptation']:
            image_features = self.image_map(image_features)
            # image_features = F.normalize(image_features, p=2, dim=1)  # [batch_size, d]

            text_features = self.text_map(text_features)
            # text_features = F.normalize(text_features, p=2, dim=1)  # [batch_size, d]

            if self.fusion == 'align':
                features = torch.mul(image_features, text_features)
            elif self.fusion == 'concat':
                features = torch.cat([image_features, text_features], dim=1)
            else:
                raise ValueError()

        elif self.name == 'text-only':
            if self.proj_map:
                features = self.text_map(text_features)
            else:
                features = text_features
            # features = F.normalize(features, p=2, dim=1)

        elif self.name == 'image-only':
            if self.proj_map:
                features = self.image_map(image_features)
            else:
                features = image_features
            # features = F.normalize(features, p=2, dim=1)

        elif self.name == 'sum':
            img_features = self.image_map(image_features)
            txt_features = self.text_map(text_features)
            features = img_features + txt_features
            # features = F.normalize(features, p=2, dim=1)

        elif self.name == 'combiner':
            proj_img_features = self.image_map(image_features)
            proj_txt_features = self.text_map(text_features)

            features = self.comb(proj_img_features, proj_txt_features)

        elif self.name == 'text-inv':
            features = self.text_inv(prompt, image_features)

        elif self.name == 'text-inv-fusion':
            features = self.text_inv(prompt, image_features)

            img_projection = self.image_map(image_features)

            if self.fusion == 'concat':
                features = torch.cat([features, img_projection], dim=1)
            elif self.fusion == 'align':
                features = torch.mul(features, img_projection)
            else:
                raise ValueError()

        elif self.name == 'text-inv-comb':
            txt_features = self.text_inv(prompt, image_features)

            img_projection = self.image_map(image_features)

            features = self.comb(img_projection, txt_features)

        else:
            raise ValueError()

        features_pre_output = self.pre_output(features)
        logits = self.output(features_pre_output).squeeze(dim=1)  # [batch_size, 1]
        preds_proxy = torch.sigmoid(logits)

        preds = (preds_proxy >= 0.5).long()

        output['loss'] = self.cross_entropy_loss(logits, batch['labels'].float())
        output['accuracy'] = self.acc(preds, batch['labels'])
        output['auroc'] = self.auroc(preds_proxy, batch['labels'])

        return output

    def training_step(self, batch, batch_idx):
        output = self.common_step(batch)

        total_loss = output['loss']

        self.log('train/total_loss', total_loss)
        self.log('train/loss', output['loss'])
        self.log('train/accuracy', output['accuracy'])
        self.log('train/auroc', output['auroc'])

        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self.common_step(batch)

        total_loss = output['loss']

        self.log(f'val/total_loss', total_loss)
        self.log(f'val/loss', output['loss'])
        self.log(f'val/accuracy', output['accuracy'])
        self.log(f'val/auroc', output['auroc'])

        return total_loss

    def test_step(self, batch, batch_idx, dataloader_idx):
        if self.dataset == 'hmc':
            prefix_map = {
                0: 'dev_seen',
                1: 'test_seen',
                2: 'dev_unseen',
                3: 'test_unseen'
            }
        elif self.dataset == 'harmeme':
            prefix_map = {
                0: 'val',
                1: 'test',
            }
        else:
            raise ValueError()

        prefix = prefix_map[dataloader_idx]

        output = self.common_step(batch)

        self.log(f'{prefix}/accuracy', output['accuracy'])
        self.log(f'{prefix}/auroc', output['auroc'])

        return output

    def training_epoch_end(self, outputs):
        self.acc.reset()
        self.auroc.reset()

    def validation_epoch_end(self, outputs):
        self.acc.reset()
        self.auroc.reset()

    def test_epoch_end(self, outputs):
        self.acc.reset()
        self.auroc.reset()

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if p.requires_grad]}
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer


def create_model(args):
    model = HateClassifier(args=args)
    return model
