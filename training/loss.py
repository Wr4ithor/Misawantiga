import torch
import torch.nn.functional as F


def calculate_loss(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """
    logits  : (batch, seq_len, vocab_size)
    targets : (batch, seq_len)          <- already shifted
    """
    # flatten so we can use the efficient cross-entropy
    logits = logits.view(-1, logits.size(-1)) # (batch*seq, vocab)
    targets = targets.view(-1) # (batch*seq,)

    loss = F.cross_entropy(
        logits,
        targets,
        ignore_index=ignore_index, # useful for padding
        reduction="mean"
    )
    return loss