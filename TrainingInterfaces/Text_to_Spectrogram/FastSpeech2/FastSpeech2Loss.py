"""
Taken from ESPNet
"""

import torch
import torch.nn.functional as F

from Layers.DurationPredictor import DurationPredictorLoss
from Utility.diverse_losses import ssim
from Utility.utils import make_non_pad_mask


def weights_nonzero_speech(target):
    # target : B x T x mel
    # Assign weight 1.0 to all labels except for padding (id=0).
    dim = target.size(-1)
    return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)


def mse_loss(decoder_output, target):
    # decoder_output : B x T x n_mel
    # target : B x T x n_mel
    assert decoder_output.shape == target.shape
    mse_loss = F.mse_loss(decoder_output, target, reduction='none')
    weights = weights_nonzero_speech(target)
    mse_loss = (mse_loss * weights).sum() / weights.sum()
    return mse_loss


def ssim_loss(decoder_output, target, bias=6.0):
    # decoder_output : B x T x n_mel
    # target : B x T x n_mel
    assert decoder_output.shape == target.shape
    weights = weights_nonzero_speech(target)
    decoder_output = decoder_output[:, None] + bias
    target = target[:, None] + bias
    ssim_loss = 1 - ssim(decoder_output, target, size_average=False)
    ssim_loss = (ssim_loss * weights).sum() / weights.sum()
    return ssim_loss


class FastSpeech2Loss(torch.nn.Module):

    def __init__(self, use_masking=True, use_weighted_masking=False, include_portaspeech_losses=False):
        """
            use_masking (bool):
                Whether to apply masking for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to weighted masking in loss calculation.
        """
        super().__init__()

        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)
        self.include_portaspeech_losses = include_portaspeech_losses

    def forward(self, after_outs, before_outs, d_outs, p_outs, e_outs, ys,
                ds, ps, es, ilens, olens, ):
        """
        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            d_outs (LongTensor): Batch of outputs of duration predictor (B, Tmax).
            p_outs (Tensor): Batch of outputs of pitch predictor (B, Tmax, 1).
            e_outs (Tensor): Batch of outputs of energy predictor (B, Tmax, 1).
            ys (Tensor): Batch of target features (B, Lmax, odim).
            ds (LongTensor): Batch of durations (B, Tmax).
            ps (Tensor): Batch of target token-averaged pitch (B, Tmax, 1).
            es (Tensor): Batch of target token-averaged energy (B, Tmax, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration predictor loss value.
            Tensor: Pitch predictor loss value.
            Tensor: Energy predictor loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            before_outs = before_outs.masked_select(out_masks)
            if after_outs is not None:
                after_outs = after_outs.masked_select(out_masks)
            ys = ys.masked_select(out_masks)
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            d_outs = d_outs.masked_select(duration_masks)
            ds = ds.masked_select(duration_masks)
            pitch_masks = make_non_pad_mask(ilens).unsqueeze(-1).to(ys.device)
            p_outs = p_outs.masked_select(pitch_masks)
            e_outs = e_outs.masked_select(pitch_masks)
            ps = ps.masked_select(pitch_masks)
            es = es.masked_select(pitch_masks)

        # calculate loss
        l1_loss = self.l1_criterion(before_outs, ys)
        if after_outs is not None:
            l1_loss += self.l1_criterion(after_outs, ys)
        duration_loss = self.duration_criterion(d_outs, ds)
        pitch_loss = self.mse_criterion(p_outs, ps)
        energy_loss = self.mse_criterion(e_outs, es)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
            out_weights /= ys.size(0) * ys.size(2)
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            duration_weights = (duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float())
            duration_weights /= ds.size(0)

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(out_masks).sum()
            duration_loss = (duration_loss.mul(duration_weights).masked_select(duration_masks).sum())
            pitch_masks = duration_masks.unsqueeze(-1)
            pitch_weights = duration_weights.unsqueeze(-1)
            pitch_loss = pitch_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            energy_loss = (energy_loss.mul(pitch_weights).masked_select(pitch_masks).sum())

        if self.include_portaspeech_losses:
            ssim_loss_value = ssim_loss(after_outs, ys)
            mse_loss_value = mse_loss(after_outs, ys)
            return l1_loss, ssim_loss_value, mse_loss_value, duration_loss, pitch_loss, energy_loss
        else:
            return l1_loss, duration_loss, pitch_loss, energy_loss
