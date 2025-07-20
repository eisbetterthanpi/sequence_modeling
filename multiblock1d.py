# @title ijepa multiblock 1d
# https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py
import torch

class MaskCollator(object):
    def __init__(self, length=200,
        enc_mask_scale=(0.2, 0.8), pred_mask_scale=(0.2, 0.8),
        nenc=1, npred=2, min_keep=4, allow_overlap=False):
        super().__init__()
        self.length = length
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks

    def _sample_block_size(self, scale):
        _rand = torch.rand(1).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.length * mask_scale) # num patches to keep
        # -- Sample block aspect-ratio
        # -- Compute block height and width (given scale and aspect-ratio)
        l = max_keep#int(round(math.sqrt(max_keep * aspect_ratio)))
        while l >= self.length: l -= 1 # crop mask to be smaller than img
        return l

    def _sample_block_mask(self, length, acceptable_regions=None):
        l = length
        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            left = torch.randint(0, self.length - l, (1,))
            mask = torch.zeros(self.length, dtype=torch.int32)
            mask[left:left+l] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones(self.length, dtype=torch.int32)
        mask_complement[left:left+l] = 0
        # --
        return mask, mask_complement

    def __call__(self, B):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        p_size = self._sample_block_size(scale=self.pred_mask_scale)
        e_size = self._sample_block_size(scale=self.enc_mask_scale)

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.length
        min_keep_enc = self.length
        for _ in range(B):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions= None
            except Exception as e:
                print(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)
        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        return collated_masks_enc, collated_masks_pred

batch=64
length=200
mask_collator = MaskCollator(length=length, enc_mask_scale=(.85, 1.), pred_mask_scale=(.15, .2),
        nenc=1, npred=4, min_keep=4,
        # allow_overlap=True)
        allow_overlap=False)

collated_masks_enc, collated_masks_pred = mask_collator(batch)
context_indices, trg_indices = torch.stack(collated_masks_enc).squeeze(0), torch.stack(collated_masks_pred).transpose(0,1).flatten(1).unique(dim=1) # [num_msk, b,num_tok]->[b,num_tok] # [64, 65], [64, 32]
# print(context_indices.shape, trg_indices.shape)


# plt.pcolormesh(mask)
b=64
mask = torch.zeros(batch ,length)
mask[torch.arange(batch).unsqueeze(-1), trg_indices] = 1
mask[torch.arange(batch).unsqueeze(-1), context_indices] = .5
# mask = mask[None,...]
# print(mask.shape)
# mask = mask[:,None,None,:]#.repeat(1,3,1,1)
# print(mask.shape)

# import numpy as np
# import matplotlib.pyplot as plt
# def imshow(img):
#     npimg = img.numpy()
#     plt.rcParams["figure.figsize"] = (8,4)
#     # plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.pcolormesh(np.transpose(npimg, (1, 2, 0)))
#     # plt.imshow(npimg)
#     plt.show()
# # imshow(mask)
# import torchvision
# print(torchvision.utils.make_grid(mask, nrow=1).shape)
# # imshow(torchvision.utils.make_grid(mask, nrow=8))
# imshow(torchvision.utils.make_grid(mask, nrow=1))
