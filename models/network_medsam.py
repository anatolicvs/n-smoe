import torch
import torch.nn as nn


class MedSAM2(nn.Module):
    def __init__(self, sam2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.sam2 = sam2
        # enabled for training
        # for param in self.sam2.sam_prompt_encoder.parameters():
        #     param.requires_grad = False

    def forward(self, image, box):
        """
        image: (B, 3, 1024, 1024)
        box: (B, 2, 2)
        """
        _features = self._image_encoder(image)
        img_embed, high_res_features = (
            _features["image_embed"],
            _features["high_res_feats"],
        )
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_coords = box_torch.reshape(-1, 2, 2)  # (B, 4) to (B, 2, 2)
                box_labels = torch.tensor(
                    [[2, 3]], dtype=torch.int, device=image.device
                )
                box_labels = box_labels.repeat(box_torch.size(0), 1)
            concat_points = (box_coords, box_labels)

            sparse_embeddings, dense_embeddings = self.sam2.sam_prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=None,
            )
        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = (
            self.sam2.sam_mask_decoder(
                image_embeddings=img_embed,  # (B, 256, 64, 64)
                image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )
        )

        return low_res_masks_logits

    def _image_encoder(self, input_image):
        backbone_out = self.sam2.forward_image(input_image)
        _, vision_feats, _, _ = self.sam2._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.sam2.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2.no_mem_embed
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        return _features


# from sam2.sam2_image_predictor import SAM2ImagePredictor
# self.predictor = SAM2ImagePredictor(self.sam2)
# def foward(self, image, box):
#     self.predictor.set_image(image)
