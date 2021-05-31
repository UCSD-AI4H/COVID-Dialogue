import torch
from torch import nn
from torch.nn import functional as F

from fairseq.models.transformer import EncoderOut

import random


class BARTMultiGPUWrapper(nn.Module):
    def __init__(self, model_name):
        nn.Module.__init__(self)

        self._interface = torch.hub.load('pytorch/fairseq', model_name)

        # tmp = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de',
        #                        tokenizer='moses', bpe='subword_nmt')
        # self._interface.task.build_dataset_for_inference = \
        #     tmp.task.build_dataset_for_inference

        self._n_split_gpus = None

    def split_to_gpus(self, n_gpus):
        if self._n_split_gpus == n_gpus:
            return

        assert n_gpus <= 2
        assert n_gpus <= torch.cuda.device_count()

        if n_gpus == 2:
            self.encoder.cuda(0)
            self.decoder.cuda(1)
        else:
            self.cuda()

        torch.cuda.empty_cache()
        self._n_split_gpus = n_gpus

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens,
            features_only=False, classification_head_name=None, **kwargs
    ):
        if classification_head_name is not None:
            features_only = True

        encoder_out = encoder_forward(
            self=self.encoder,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            **kwargs)._asdict()

        for key in encoder_out:
            if isinstance(encoder_out[key], torch.Tensor):
                encoder_out[key] = encoder_out[key].to(
                    next(self.decoder.parameters()).device)
        encoder_out = EncoderOut(**encoder_out)

        prev_output_tokens = prev_output_tokens.to(
            next(self.decoder.parameters()).device)
        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            **kwargs,
        )

        if classification_head_name is not None:
            sentence_representation = x[src_tokens.eq(
                self.encoder.dictionary.eos()), :].view(
                x.size(0), -1, x.size(-1))[:, -1,:]
            x = self.classification_heads[classification_head_name](
                sentence_representation)

        x = x.to('cuda')

        return x, extra

    @property
    def model(self):
        return self._interface.model

    @property
    def sample(self):
        return self._interface.sample

    @property
    def encode(self):
        return self._interface.encode

    @property
    def decode(self):
        return self._interface.decode

    @property
    def encoder(self):
        return self._interface.model.encoder

    @property
    def decoder(self):
        return self._interface.model.decoder

    @property
    def dictionary(self):
        return self._interface.model.decoder.dictionary


def encoder_forward(self, src_tokens, src_lengths, cls_input=None, return_all_hiddens=False, **unused):

    def _forward_embedding():
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(
            src_tokens.to(next(self.embed_tokens.parameters()).device))
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens.to(next(
                self.embed_positions.parameters()).device)).to(embed.device)
        if self.layernorm_embedding:
            x = self.layernorm_embedding(
                x.to(next(self.layernorm_embedding.parameters()).device))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    if self.layer_wise_attention:
        return_all_hiddens = True

    x, encoder_embedding = _forward_embedding()
    x = x.to('cuda')
    encoder_embedding = encoder_embedding.to('cuda')

    # B x T x C -> T x B x C
    x = x.transpose(0, 1)

    # compute padding mask
    encoder_padding_mask = src_tokens.eq(self.padding_idx)
    if not encoder_padding_mask.any():
        encoder_padding_mask = None

    encoder_states = [] if return_all_hiddens else None

    # encoder layers
    for layer in self.layers:
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        dropout_probability = random.uniform(0, 1)
        if not self.training or (dropout_probability > self.encoder_layerdrop):
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)

    if self.layer_norm:
        x = self.layer_norm(x)
        if return_all_hiddens:
            encoder_states[-1] = x

    return EncoderOut(
        encoder_out=x,  # T x B x C
        encoder_padding_mask=encoder_padding_mask,  # B x T
        encoder_embedding=encoder_embedding,  # B x T x C
        encoder_states=encoder_states,  # List[T x B x C]
    )


# https://github.com/pytorch/fairseq/blob/5028ed1b6bedd526dee27ea731284f43e87303f0/fairseq/criterions/label_smoothed_cross_entropy.py#L12
def label_smoothed_nll_loss(lprobs, target, epsilon, smoothing_labels=None, ignore_index=None, reduce=True):
    assert (ignore_index is None) or (smoothing_labels is None)

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)

    if smoothing_labels is None:
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    else:
        smooth_loss = -lprobs[..., smoothing_labels].sum(dim=-1, keepdim=True)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss