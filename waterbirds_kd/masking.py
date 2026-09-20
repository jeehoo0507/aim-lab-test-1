import torch


def select_tokens(attention, method, keep, foreground=None, generator=None):
    """Return unique patch indices [B,K] and actual swaps [B]. No CLS indices.

    Both rescue controls use the same per-image feasible swap count. For
    foreground rescue, outgoing BG / incoming FG are sampled uniformly.
    """
    attention = attention.detach()
    b, n = attention.shape
    if not 1 <= keep <= n:
        raise ValueError(f"keep must be in [1,{n}]")
    swaps = torch.zeros(b, device=attention.device, dtype=torch.long)
    if method in ("ce", "full", "teacher"):
        return None, swaps
    if method == "random":
        scores = torch.rand((b, n), device=attention.device, generator=generator)
        return scores.topk(keep, dim=1).indices, swaps
    selected = attention.topk(keep, dim=1).indices
    if method == "student":
        return selected, swaps
    if not method.startswith(("foreground_rescue_", "random_rescue_")):
        raise ValueError(f"Unknown mask method: {method}")
    if foreground is None:
        raise ValueError("Rescue and matched random rescue require segmentation")
    requested = int(method.rsplit("_", 1)[1])
    if requested < 0:
        raise ValueError("rescue count must be nonnegative")
    for i in range(b):
        present = torch.zeros(n, device=attention.device, dtype=torch.bool)
        present[selected[i]] = True
        outgoing_bg = torch.where(~foreground[i, selected[i]])[0]
        incoming_fg = torch.where(foreground[i] & ~present)[0]
        count = min(requested, outgoing_bg.numel(), incoming_fg.numel())
        swaps[i] = count
        if not count:
            continue
        if method.startswith("foreground"):
            outgoing, incoming = outgoing_bg, incoming_fg
        else:
            outgoing = torch.arange(keep, device=attention.device)
            incoming = torch.where(~present)[0]
        out_order = torch.randperm(outgoing.numel(), device=attention.device, generator=generator)[:count]
        in_order = torch.randperm(incoming.numel(), device=attention.device, generator=generator)[:count]
        selected[i, outgoing[out_order]] = incoming[in_order]
    return selected, swaps


def binary_mask(indices, batch_size=None, device=None):
    if indices is None:
        return torch.ones(batch_size, 196, dtype=torch.bool, device=device)
    mask = torch.zeros(indices.shape[0], 196, dtype=torch.bool, device=indices.device)
    return mask.scatter_(1, indices, True)
