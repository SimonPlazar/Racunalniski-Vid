import torch
import re
import os
import pickle


def extract_epoch(filename):
    """Extract epoch or iteration number from checkpoint filename"""
    # Try iteration first
    match = re.search(r"iteration_(\d+)", filename)
    if match:
        return int(match.group(1))
    # Fallback to epoch
    match = re.search(r"epoch_(\d+)", filename)
    return int(match.group(1)) if match else -1


def save_checkpoint_generic(checkpoint_dir, epoch_or_iter, state_dict, max_checkpoints=4):
    """
    Save checkpoint with automatic naming based on what's in state_dict.
    Supports both epoch-based and iteration-based training.
    """
    # Determine if we're using epoch or iteration
    if 'iteration' in state_dict:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iteration_{epoch_or_iter}.pth")
        state_dict.setdefault('iteration', epoch_or_iter)
        prefix = "checkpoint_iteration_"
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch_or_iter}.pth")
        state_dict.setdefault('epoch', epoch_or_iter)
        prefix = "checkpoint_epoch_"

    torch.save(state_dict, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")

    # Clean old checkpoints
    checkpoints = [f for f in os.listdir(checkpoint_dir)
                   if f.startswith(prefix) and f.endswith(".pth")]
    checkpoints = sorted(checkpoints, key=extract_epoch)

    while len(checkpoints) > max_checkpoints:
        old_ckpt = os.path.join(checkpoint_dir, checkpoints[0])
        os.remove(old_ckpt)
        print(f"🗑️  Removed old checkpoint: {old_ckpt}")
        checkpoints.pop(0)


def load_checkpoint_generic(checkpoint_dir, device='cpu'):
    """
    Load the latest checkpoint from directory.
    Supports both epoch-based and iteration-based checkpoints.
    """
    if not os.path.exists(checkpoint_dir):
        print("🚀 Checkpoint directory doesn't exist, starting from scratch")
        return {}

    # Look for both epoch and iteration based checkpoints
    checkpoints = [f for f in os.listdir(checkpoint_dir)
                   if (f.startswith("checkpoint_epoch_") or f.startswith("checkpoint_iteration_"))
                   and f.endswith(".pth")]

    if not checkpoints:
        print("🚀 No checkpoint found, starting from scratch")
        return {}

    checkpoints = sorted(checkpoints, key=extract_epoch)
    latest_ckpt = os.path.join(checkpoint_dir, checkpoints[-1])

    # Try loading with default settings first. On PyTorch >=2.6 torch.load may default to
    # weights_only=True which can raise an UnpicklingError for full checkpoints that
    # include arbitrary Python objects. If that happens, retry with weights_only=False
    # but warn the user (only do this if you trust the checkpoint source).
    try:
        checkpoint = torch.load(latest_ckpt, map_location=device)
    except Exception as e:
        # If it's an UnpicklingError related to 'Weights only load failed', retry
        err_str = str(e)
        if 'Weights only load failed' in err_str or isinstance(e, pickle.UnpicklingError):
            print("⚠️  torch.load raised an UnpicklingError (weights-only). Retrying with weights_only=False — only do this for trusted checkpoints.")
            try:
                checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
            except Exception as e2:
                print(f"❌ Failed to load checkpoint even after retry: {e2}")
                raise
        else:
            # re-raise unexpected exceptions
            raise

    # Display appropriate info based on what's in checkpoint
    if 'iteration' in checkpoint:
        print(f"✅ Loaded checkpoint: {latest_ckpt} (iteration {checkpoint['iteration']})")
    else:
        print(f"✅ Loaded checkpoint: {latest_ckpt} (epoch {checkpoint.get('epoch', '?')})")
    return checkpoint


# Example usage:
#
# Load checkpoint if exists
# checkpoint = load_checkpoint_generic(checkpoint_dir, device)
# if checkpoint:
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_iteration = checkpoint.get('iteration', 1) + 1
#     val_losses = checkpoint.get('val_losses', [])
#     val_epes = checkpoint.get('val_epes', [])
#
# Save checkpoint
# save_checkpoint_generic(
#     checkpoint_dir,
#     iteration,
#     {
#         'iteration': iteration,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'val_losses': val_losses,
#         'val_epes': val_epes,
#     },
#     max_checkpoints=5
# )

