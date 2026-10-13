"""Original implementation of BSDA-inspired perturbation on observed image tokens.

Method source: https://arxiv.org/abs/2403.06138 and author's BSDAWarp.
Adaptations: projected 128D image tokens, LayerNorm for singleton observed-image
batches, one synthetic view, per-sample KL; this is not the paper reproduction.
"""
import torch
from torch import nn

POLICY = {'method':'BSDA-inspired observed projected-image-token augmentation',
          'feature_dimension':128,'drop_probability':.8,'synthetic_views':1,
          'maximum_alpha':.5,'ramp_epochs':25,'KL_weight':8e-4,'reconstruction_weight':1.,
          'normalization':'LayerNorm replacing BatchNorm for singleton observed-image batches',
          'input_scope':'observed image tokens after modality dropout only',
          'control':'identical extra original-image CE and fusion-forward budget; no semantic noise or VAE losses',
          'paper':'https://arxiv.org/abs/2403.06138','exact_paper_reproduction':False}


class ImageTokenBSDA(nn.Module):
    def __init__(self,dimension):
        super().__init__()
        def stack():
            return nn.Sequential(nn.Linear(dimension,dimension),nn.LayerNorm(dimension),nn.GELU(),
                                 nn.Linear(dimension,dimension),nn.LayerNorm(dimension),nn.GELU())
        self.encode = stack()
        self.variance = nn.Linear(dimension,dimension)
        self.decode = stack()

    def forward(self,x):
        logvar = self.variance(self.encode(x)).clamp(-10,10)
        magnitude = torch.randn_like(x)*torch.exp(logvar*.5)
        keep = (torch.rand_like(x) >= POLICY['drop_probability']).to(x.dtype)
        perturbed = x + magnitude*keep/(1-POLICY['drop_probability'])*(x!=0).to(x.dtype)
        reconstruction = .5*(self.decode(magnitude)-x).square().mean()
        kl = .5*(logvar.exp()-1-logvar).sum(1).mean()
        auxiliary = POLICY['KL_weight']*kl+reconstruction
        return perturbed,auxiliary


def modality_embeddings(model,batch):
    fusion=model.fusion
    embeddings={'image':fusion.image_projection(batch['image_features']),'tct':fusion.tct_encoder(batch['tct'])}
    present={'image':batch['image_present'],'tct':batch['tct_present']}
    for name in ('history','dna','methylation'):
        embeddings[name]=getattr(fusion,f'{name}_encoder')(batch[name],batch[f'{name}_mask'])
        present[name]=batch[f'{name}_mask'].any(1)
    return embeddings,present


def training_objective(model,batch,weights,recipe,epoch):
    embeddings,present=modality_embeddings(model,batch)
    output=model.forward_from_embeddings(embeddings,present)
    ce=nn.functional.cross_entropy(output,batch['label'],weight=weights)
    observed=present['image'].bool()
    if not observed.any():
        return ce
    selected={name:e[observed] for name,e in embeddings.items()}
    selected_present={name:p[observed] for name,p in present.items()}
    auxiliary=ce.new_zeros(())
    if recipe=='bsda_image':
        selected['image'],auxiliary=model.image_bsda(selected['image'])
    elif recipe!='identity_image_repeat':
        raise ValueError('Unrecognized semantic augmentation recipe')
    augmented=model.forward_from_embeddings(selected,selected_present)
    extra_ce=nn.functional.cross_entropy(augmented,batch['label'][observed],weight=weights)
    alpha=POLICY['maximum_alpha']*min(epoch/POLICY['ramp_epochs'],1.)
    return ce+alpha*(extra_ce+auxiliary)
