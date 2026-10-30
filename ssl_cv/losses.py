import torch
import torch.nn.functional as F


def nt_xent(z1, z2, temperature=0.2):
    """SimCLR / InfoNCE loss.

    For a batch of N images we have 2N views. For each view, its partner view is the
    positive and the other 2N-2 views are negatives. The loss is a (2N-1)-way softmax
    classification of "which one is my partner", with cosine similarity / temperature
    as logits.
    """
    z = F.normalize(torch.cat([z1, z2]).float(), dim=1)
    n = z1.shape[0]
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))  # a view is not its own positive
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)
