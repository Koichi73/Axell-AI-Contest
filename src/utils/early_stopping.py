from pathlib import Path
import torch

class EarlyStopping:
    def __init__(self, output_dir, patience):
        """
        Args:
            output_dir (str): The directory where the model will be saved.
            patience (int): The number of epochs to wait before early stopping.
        """
        self.output_dir = Path(output_dir)
        self.patience = patience
        self.early_stop = False
        self._patience_counter = 0
        self._best_psnr = float('-inf')
        self._min_delta = 0.0

    def __call__(self, valid_psnr, model):
        if valid_psnr > self._best_psnr - self._min_delta:
            self._best_psnr = valid_psnr
            self._patience_counter = 0
            torch.save(model.state_dict(), self.output_dir / "model.pth")
        else:
            self._patience_counter += 1
            if self._patience_counter >= self.patience:
                print("Early stopping triggered.")
                self.early_stop = True