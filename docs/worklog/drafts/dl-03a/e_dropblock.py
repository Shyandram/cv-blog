import numpy as np

def dropblock_mask(feat, block, keep_prob, rng):
    """One DropBlock mask for a feat x feat map (Ghiasi et al., Algorithm 1)."""
    gamma = (1 - keep_prob) / block ** 2 * feat ** 2 / (feat - block + 1) ** 2
    half = block // 2
    seeds = np.zeros((feat, feat), bool)                       # only where a whole block fits
    seeds[half:feat - half, half:feat - half] = rng.random((feat - block + 1,) * 2) < gamma
    mask = np.ones((feat, feat))
    for i, j in zip(*np.nonzero(seeds)):
        mask[i - half:i + half + 1, j - half:j + half + 1] = 0.0
    return mask, gamma

rng = np.random.default_rng(0)
feat, block, keep = 38, 7, 0.9
masks = [dropblock_mask(feat, block, keep, rng)[0] for _ in range(2000)]
gamma = dropblock_mask(feat, block, keep, rng)[1]
dropped = 1 - np.mean([m.mean() for m in masks])
print(f"gamma = {gamma:.5f} | target drop rate {1 - keep:.2f} | measured {dropped:.3f}")
m = masks[0]
print("first mask: dropped units", int((m == 0).sum()), "| rescale factor", round(m.size / m.sum(), 3))
