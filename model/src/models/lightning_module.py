import torch
import torch.nn.functional as F
import pytorch_lightning as L
from typing import Dict, Any, Optional
import numpy as np
from .ast_model import Evaluator, masked_mae, masked_mae_weighted, cosine_distillation_loss


class CrescendAILightningModule(L.LightningModule):
    """PyTorch Lightning module for CrescendAI piano performance analysis."""
    
    def __init__(
        self,
        num_dims: int = 19,
        num_datasets: int = 8,
        ds_embed_dim: int = 16,
        emb_dim: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        distillation_weight: float = 0.1,
        use_weighted_loss: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = Evaluator(
            num_dims=num_dims,
            num_datasets=num_datasets,
            ds_embed_dim=ds_embed_dim,
            emb_dim=emb_dim
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.distillation_weight = distillation_weight
        self.use_weighted_loss = use_weighted_loss
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, mel_spec: torch.Tensor, dataset_id: torch.Tensor):
        return self.model(mel_spec, dataset_id)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")
    
    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        mel_spec = batch["mel_spec"]
        dataset_id = batch["dataset_id"]
        targets = batch["targets"]
        mask = batch["mask"]
        
        # Forward pass
        predictions, audio_emb = self(mel_spec, dataset_id)
        
        # Main regression loss
        if self.use_weighted_loss and "weights" in batch:
            main_loss = masked_mae_weighted(predictions, targets, mask, batch["weights"])
        else:
            main_loss = masked_mae(predictions, targets, mask)
        
        # Distillation loss (if available)
        total_loss = main_loss
        if "distill_vec" in batch and self.distillation_weight > 0:
            distill_loss = cosine_distillation_loss(
                audio_emb, batch["distill_vec"], self.model.distill_proj
            )
            total_loss = main_loss + self.distillation_weight * distill_loss
            self.log(f"{stage}_distill_loss", distill_loss, prog_bar=True)
        
        # Logging
        self.log(f"{stage}_loss", total_loss, prog_bar=True)
        self.log(f"{stage}_main_loss", main_loss, prog_bar=True)
        
        # Compute per-dimension metrics (validation only)
        if stage == "val":
            self._log_dimension_metrics(predictions, targets, mask, stage)
        
        return total_loss
    
    def _log_dimension_metrics(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        mask: torch.Tensor,
        stage: str
    ):
        """Log per-dimension correlation metrics."""
        with torch.no_grad():
            # Convert to numpy for correlation calculation
            pred_np = predictions.detach().cpu().numpy()
            target_np = targets.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()
            
            correlations = []
            for dim in range(predictions.shape[1]):
                # Get valid predictions for this dimension
                valid_mask = mask_np[:, dim].astype(bool)
                if valid_mask.sum() > 1:  # Need at least 2 points for correlation
                    pred_dim = pred_np[valid_mask, dim]
                    target_dim = target_np[valid_mask, dim]
                    
                    # Compute Pearson correlation
                    corr = np.corrcoef(pred_dim, target_dim)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                        self.log(f"{stage}_corr_dim_{dim}", corr)
            
            # Log mean correlation
            if correlations:
                mean_corr = np.mean(correlations)
                self.log(f"{stage}_mean_correlation", mean_corr, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step for inference."""
        mel_spec = batch["mel_spec"]
        dataset_id = batch["dataset_id"]
        
        with torch.no_grad():
            predictions, embeddings = self(mel_spec, dataset_id)
        
        return {
            "predictions": predictions,
            "embeddings": embeddings,
            "mel_spec": mel_spec,
            "dataset_id": dataset_id
        }