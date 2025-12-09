import torch
import re
import os
import pickle


def extract_epoch(filename):
    match = re.search(r"epoch_(\d+)", filename)
    return int(match.group(1)) if match else -1


def save_checkpoint_generic(checkpoint_dir, epoch, state_dict, max_checkpoints=4):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")

    # Add epoch to state_dict if not present
    if 'epoch' not in state_dict:
        state_dict['epoch'] = epoch

    torch.save(state_dict, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")

    # Clean old checkpoints
    checkpoints = [f for f in os.listdir(checkpoint_dir)
                   if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    checkpoints = sorted(checkpoints, key=extract_epoch)

    while len(checkpoints) > max_checkpoints:
        old_ckpt = os.path.join(checkpoint_dir, checkpoints[0])
        os.remove(old_ckpt)
        print(f"🗑️  Removed old checkpoint: {old_ckpt}")
        checkpoints.pop(0)


def load_checkpoint_generic(checkpoint_dir, device='cpu'):
    checkpoints = [f for f in os.listdir(checkpoint_dir)
                   if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    checkpoints = sorted(checkpoints, key=extract_epoch)

    if checkpoints:
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

        print(f"✅ Loaded checkpoint: {latest_ckpt} (epoch {checkpoint.get('epoch', '?')})")
        return checkpoint
    else:
        print("🚀 No checkpoint found, starting from scratch")
        return {}


# example calls

# Load checkpoint if exists
# checkpoint = load_checkpoint_generic(out_dir, device)
# if checkpoint:
#     G_A2B.load_state_dict(checkpoint['G_A2B'])
#     G_B2A.load_state_dict(checkpoint['G_B2A'])
#     D_A.load_state_dict(checkpoint['D_A'])
#     D_B.load_state_dict(checkpoint['D_B'])
#     optimizer_G.load_state_dict(checkpoint['optimizer_G'])
#     optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
#     optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])
#     start_epoch = checkpoint['epoch']
#     G_losses = checkpoint.get('G_losses', [])
#     D_A_losses = checkpoint.get('D_A_losses', [])
#     D_B_losses = checkpoint.get('D_B_losses', [])
#     test_A = checkpoint.get('test_A')
#     test_B = checkpoint.get('test_B')

# Save checkpoint
# if epoch % 1 == 0 or epoch == num_epochs:
#     save_checkpoint_generic(out_dir, epoch, {
#         'G_A2B': G_A2B.state_dict(),
#         'G_B2A': G_B2A.state_dict(),
#         'D_A': D_A.state_dict(),
#         'D_B': D_B.state_dict(),
#         'optimizer_G': optimizer_G.state_dict(),
#         'optimizer_D_A': optimizer_D_A.state_dict(),
#         'optimizer_D_B': optimizer_D_B.state_dict(),
#         'G_losses': G_losses,
#         'D_A_losses': D_A_losses,
#         'D_B_losses': D_B_losses,
#         'test_A': test_A.cpu(),
#         'test_B': test_B.cpu(),
#         'config': training_config,
#     })
