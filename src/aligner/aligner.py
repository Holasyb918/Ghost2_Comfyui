import torch
import torch.nn as nn

# embedder
from .embedder import Embedder

# generator
from .generator import Generator


class AlignerModule(nn.Module):
    def __init__(self, cfg, inference=False):
        super(AlignerModule, self).__init__()
        self.inference = inference

        self.embedder = Embedder(**cfg.model.embed)
        self.gen = Generator(
            d_por=cfg.model.embed.d_por,
            d_id=cfg.model.embed.d_id,
            d_pose=cfg.model.embed.d_pose,
            d_exp=cfg.model.embed.d_exp,
            **cfg.model.gen,
        )

        # self.disc = Discriminator(**cfg.model.discr)
        # if not self.inference:
        #     self.aligner_loss = AlignerLoss(
        #         id_encoder=self.embedder.id_encoder,
        #         disc=self.disc,
        #         gaze_start = cfg.train_options.gaze_start,
        #         **cfg.train_options.weights
        #     )
        # optim_options = cfg.train_options.optim
        # self.g_lr = optim_options.g_lr
        # self.d_lr = optim_options.d_lr
        # self.g_clip = optim_options.g_clip
        # self.d_clip = optim_options.d_clip
        # self.betas = (optim_options.beta1, optim_options.beta2)
        # self.segment_model = None
        # self.automatic_optimization = False

        # self.save_hyperparameters()

        if cfg.model.segment:
            hf_name = "facebook/mask2former-swin-tiny-coco-instance"
            Mask2Former_ckpt_path = os.path.join(WEIGHTS_PATH, hf_name)
            if not os.path.exists(Mask2Former_ckpt_path):
                Mask2Former_ckpt_path = hf_name
            self.segment_model = StyleMatte(Mask2Former_ckpt_path=Mask2Former_ckpt_path)
            segment_model_ckpt_path = os.path.join(WEIGHTS_PATH, "stylematte_synth.pth")
            self.segment_model.load_state_dict(
                torch.load(segment_model_ckpt_path, map_location="cpu")
            )

            self.segment_model.eval()

        # if not self.inference:
        #     self.lpips = LPIPS()
        #     self.psnr = PSNR()
        #     self.ssim = SSIM()
        #     self.mssim = MS_SSIM()
        # self.val_outputs = [[], []]

    def forward(self, X_dict, use_geometric_augmentations=False):
        return self.gen(
            self.embedder(
                X_dict, use_geometric_augmentations=use_geometric_augmentations
            )
        )

    def configure_optimizers(self):

        opt_G = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.gen.parameters()),
            lr=self.g_lr,
            betas=self.betas,
            eps=1e-5,
        )
        opt_D = torch.optim.Adam(
            self.disc.parameters(), lr=self.d_lr, betas=self.betas, eps=1e-5
        )
        return opt_G, opt_D

    def calc_mask(self, batch):
        batch = dict(list(batch.items()))
        batch_shape = list(batch["face_wide"].shape)
        batch_shape[2] = 1
        device = batch["face_wide"].device
        dtype = batch["face_wide"].dtype
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype)[
            None, :, None, None
        ]
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype)[
            None, :, None, None
        ]

        with torch.no_grad():
            normalized = (batch["face_wide"] + 1) / 2
            normalized = (normalized - mean) / std

            mask = self.segment_model(
                torch.flatten(normalized, start_dim=0, end_dim=1)
            ).reshape(*batch_shape)
        batch["face_wide_mask"] = mask

        return batch

    def training_step(self, train_batch, batch_idx):

        X_dict = make_X_dict(
            X_arc=train_batch["face_arc"],
            X_wide=train_batch["face_wide"],
            X_mask=train_batch[
                "face_wide_mask"
            ],  # if self.segment_model is not None else None,
            X_emotion=train_batch["face_emoca"],
            X_keypoints=train_batch["keypoints"],
            segmentation=train_batch["segmentation"],
        )

        opt_G, opt_D = self.optimizers()

        data_dict = self.forward(X_dict, use_geometric_augmentations=True)

        losses = self.aligner_loss(data_dict, X_dict, epoch=self.current_epoch)

        def closure_G():
            opt_G.zero_grad()
            self.manual_backward(losses["L_G"], retain_graph=True)
            self.clip_gradients(opt_G, gradient_clip_val=self.g_clip)
            return losses["L_G"]

        opt_G.step(closure=closure_G)

        def closure_D():
            opt_D.zero_grad()
            self.manual_backward(losses["L_D"])
            self.clip_gradients(opt_D, gradient_clip_val=self.d_clip)
            return losses["L_D"]

        opt_D.step(closure=closure_D)

        logs = dict((k, v.item()) for k, v in losses.items())
        self.log_dict(logs)

        return data_dict

    def validation_step(self, val_batch, batch_idx, dataloader_idx):

        X_dict = make_X_dict(
            val_batch["face_arc"], val_batch["face_wide"], val_batch["face_wide_mask"]
        )

        with torch.no_grad():
            outputs = self.forward(X_dict)

        if dataloader_idx == 0:
            masked_output = blend_alpha(
                outputs["fake_rgbs"], X_dict["target"]["face_wide_mask"]
            )

            lpips_val = self.lpips(masked_output, X_dict["target"]["face_wide"])
            psnr_val = self.psnr(masked_output, X_dict["target"]["face_wide"])
            ssim_val = self.ssim(masked_output, X_dict["target"]["face_wide"])
            mssim_val = self.mssim(masked_output, X_dict["target"]["face_wide"])

            id_dict = self.aligner_loss.id_loss(outputs, X_dict, return_embeds=True)
            id_metric = F.cosine_similarity(
                id_dict["fake_embeds"], id_dict["real_embeds"]
            ).mean()

            metrics = {
                "LPIPS": lpips_val,
                "PSNR": psnr_val,
                "SSIM": ssim_val,
                "MS_SSIM": mssim_val,
                "ID self": id_metric,
            }

        if dataloader_idx == 1:
            id_dict = self.aligner_loss.id_loss(outputs, X_dict, return_embeds=True)
            id_score = F.cosine_similarity(
                id_dict["fake_embeds"], id_dict["real_embeds"]
            ).mean()
            metrics = {"ID cross": id_score}

        out_dict = {
            "fake_rgbs": outputs["fake_rgbs"],
            "fake_segm": outputs["fake_segm"],
            "metrics": metrics,
        }

        if dataloader_idx == 0:
            self.val_outputs[0].append(out_dict)
        else:
            self.val_outputs[1].append(out_dict)

        return out_dict

    def on_validation_epoch_end(self):
        outputs = self.val_outputs
        self_metics = outputs[0]  # self reenacment dataloader, list of dicts for epoch
        cross_metics = outputs[1]  # cross reenacment

        keys_self = list(self_metics[0]["metrics"].keys())
        keys_cross = list(cross_metics[0]["metrics"].keys())

        losses_self = {key: [] for key in keys_self}
        losses_cross = {key: [] for key in keys_cross}

        for batch_n in range(len(outputs[0])):

            for key in keys_self:
                losses_self[key].append(self_metics[batch_n]["metrics"][key].item())

        for batch_n in range(len(outputs[1])):
            for key in keys_cross:
                losses_cross[key].append(cross_metics[batch_n]["metrics"][key].item())

        for key, val in losses_self.items():
            self.log(key, np.mean(val), sync_dist=True)

        for key, val in losses_cross.items():
            self.log(key, np.mean(val), sync_dist=True)

        self.val_outputs[0].clear()
        self.val_outputs[1].clear()
