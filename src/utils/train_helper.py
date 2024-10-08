import sys
import pandas as pd
import matplotlib.pyplot as plt
import torch

def check_and_make_directory(output_dir):
    """Check if the output directory exists and create it if not."""
    if output_dir.exists():
        answer = input(f"The directory '{output_dir}' already exists. Do you want to overwrite it? (y/n): ")
        if answer.lower() != "y":
            print("The process was interrupted.")
            sys.exit(0)
    output_dir.mkdir(exist_ok=True, parents=True)

def create_csv(train_losses, validation_losses, train_psnres, validation_psnres, output_dir):
    """Create CSV files for losses and PSNRs using pandas"""
    epochs = list(range(1, len(train_losses) + 1))
    losses_df = pd.DataFrame({
        'Epochs': epochs,
        'Train loss': train_losses,
        'Valid loss': validation_losses
    })
    psnrs_df = pd.DataFrame({
        'Epochs': epochs,
        'Train PSNR': train_psnres,
        'Valid PSNR': validation_psnres
    })
    losses_df.to_csv(output_dir / "losses.csv", index=False)
    psnrs_df.to_csv(output_dir / "psnrs.csv", index=False)

def plot_curve(x_values, y_values, labels, xlabel, ylabel, output_path, ylim=None):
    """General function to plot and save a curve"""
    plt.plot(x_values, y_values[0], label=labels[0])
    plt.plot(x_values, y_values[1], label=labels[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_learning_curve(train_losses, validation_losses, output_dir):
    """Plot learning curve"""
    epochs = range(1, len(train_losses) + 1)
    plot_curve(epochs, [train_losses, validation_losses], ['Train Loss', 'Valid Loss'], 
               'Epochs', 'Loss', output_dir / "learning_curve.png", ylim=[0, 0.013])

def plot_psnr_curve(train_psnres, validation_psnres, output_dir):
    """Plot PSNR curve"""
    epochs = range(1, len(train_psnres) + 1)
    plot_curve(epochs, [train_psnres, validation_psnres], ['Train PSNR', 'Valid PSNR'], 
               'Epochs', 'PSNR', output_dir / "psnr_curve.png")

def save_checkpoint(epoch, model, optimizer, scheduler, scaler, train_losses, validation_losses, train_psnres, validation_psnres, output_dir):
    """Save a checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'train_losses': train_losses,
        'valid_losses': validation_losses,
        'train_psnres': train_psnres,
        'valid_psnres': validation_psnres
    }
    torch.save(checkpoint, output_dir / 'checkpoint.pth')

def load_checkpoint(model, optimizer, scheduler, scaler, train_losses, validation_losses, train_psnres, validation_psnres, checkpoint_path):
    """Load a checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    train_losses = checkpoint['train_losses']
    validation_losses = checkpoint['valid_losses']
    train_psnres = checkpoint['train_psnres']
    validation_psnres = checkpoint['valid_psnres']
    return checkpoint['epoch'], model, optimizer, scheduler, scaler, train_losses, validation_losses, train_psnres, validation_psnres

def visualize_batch(images, labels, save_dir, epoch, num_samples=4):
    """Visualize a batch of images and labels."""
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    for i in range(num_samples):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(images[i].transpose(1, 2, 0))
        axes[0].axis('off')
        axes[1].imshow(labels[i].transpose(1, 2, 0))
        axes[1].axis('off')
        plt.savefig(save_dir / f'epoch{epoch}_sample{i}.png')
        plt.close(fig)