# training.py

import json
from datetime import datetime, timezone


import torch
import torch.nn as nn

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm


# ==================== TRAINING MANAGER ====================
class TrainingManager:
    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device
        self.training_log = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def _load_checkpoint(self, optimizer=None):
        """Load checkpoint to resume training"""
        checkpoint_path = self.config.MODELS_DIR / "best_model.pth"

        if not checkpoint_path.exists():
            print("📭 No checkpoint found - starting fresh training")
            return 1, 0, 0  # start_epoch, best_val_acc, patience_counter

        print(f"🔄 Resuming from checkpoint: {checkpoint_path.name}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if available
        if optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✅ Optimizer state loaded")
            except Exception as e:
                print(f"⚠️ Could not load optimizer state: {e}")

        # Load training log if available
        if self.config.LOSS_JSON_DRIVE.exists():
            try:
                self.training_log = json.loads(self.config.LOSS_JSON_DRIVE.read_text())
                print(f"✅ Training log loaded ({len(self.training_log['train_loss'])} epochs)")
            except Exception as e:
                print(f"⚠️ Could not load training log: {e}")

        start_epoch = checkpoint.get('epoch', 1) + 1
        best_val_acc = checkpoint.get('best_val_acc', 0)
        patience_counter = checkpoint.get('patience_counter', 0)

        print(f"🎯 Resuming from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
        return start_epoch, best_val_acc, patience_counter

    def _save_checkpoint(self, epoch, optimizer, best_val_acc, patience_counter, filename="best_model.pth"):
        """Save complete checkpoint for resuming"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "best_val_acc": best_val_acc,
            "patience_counter": patience_counter,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Always save locally
        local_path = self.config.MODELS_DIR / filename
        torch.save(checkpoint, local_path)
        print(f"💾 Checkpoint saved locally: {local_path}")

        # Save training log
        try:
            with open(self.config.LOSS_JSON, "w") as f:
                json.dump(self.training_log, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save training log: {e}")

        # Optional: if in Colab, also save to Drive folder (if separate)
        if self.config.IN_COLAB and hasattr(self.config, 'LOSS_JSON_DRIVE'):
            try:
                with open(self.config.LOSS_JSON_DRIVE, "w") as f:
                    json.dump(self.training_log, f, indent=2)
            except Exception as e:
                print(f"⚠️ Could not save training log to Drive: {e}")

    def train_and_evaluate(self, train_loader, val_loader, test_loader, num_epochs=10, tb_logger=None):
        """Main training loop with resume capability"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler('cuda', enabled=self.config.USE_AMP)

        # Load checkpoint to resume training
        start_epoch, best_val_acc, patience_counter = self._load_checkpoint(optimizer)

        patience = 5

        print(f"🎬 Starting training from epoch {start_epoch} to {num_epochs}")

        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\n📈 Epoch {epoch}/{num_epochs}")

            train_loss, train_acc = self._train_epoch(epoch, train_loader, optimizer, criterion, scaler)
            val_loss, val_acc = self._validate_epoch(epoch, val_loader, criterion)

            # Update learning rate
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"📉 Learning rate: {current_lr:.2e}")

            # Update training log
            self.training_log["train_loss"].append(train_loss)
            self.training_log["val_loss"].append(val_loss)
            self.training_log["train_acc"].append(train_acc)
            self.training_log["val_acc"].append(val_acc)

            # Log metrics to TensorBoard
            if tb_logger:
                tb_logger.log_metrics({
                    "Loss/train": train_loss,
                    "Loss/val": val_loss,
                    "Acc/train": train_acc,
                    "Acc/val": val_acc,
                    "LearningRate": current_lr
                }, epoch)

            # Early stopping and checkpointing
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self._save_checkpoint(epoch, optimizer, best_val_acc, patience_counter, "best_model.pth")
                print(f"💾 New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"⏳ No improvement - Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch}")
                break

        # Final evaluation
        print("\n🎯 Final Evaluation:")
        self._evaluate_final(test_loader)

        # Save final checkpoint
        self._save_checkpoint(num_epochs, optimizer, best_val_acc, patience_counter, "final_model.pth")
        print(f"💾 Final model saved with best val acc: {best_val_acc:.2f}%")

    def _train_epoch(self, epoch, train_loader, optimizer, criterion, scaler):
        """Train one epoch"""
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train")
        for X, y in pbar:
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.config.USE_AMP):
                out = self.model(X)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * X.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)

            pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{100*correct/total:.1f}%")

        return running_loss / total, 100.0 * correct / total

    def _validate_epoch(self, epoch, val_loader, criterion):
        """Validate one epoch"""
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)

                with torch.amp.autocast('cuda', enabled=self.config.USE_AMP):
                    out = self.model(X)
                    loss = criterion(out, y)

                running_loss += loss.item() * X.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += X.size(0)

        val_acc = 100.0 * correct / total
        print(f"📊 Validation - Loss: {running_loss/total:.4f}, Acc: {val_acc:.2f}%")
        return running_loss / total, val_acc

    def _evaluate_final(self, test_loader):
        """Final evaluation"""
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X, y in tqdm(test_loader, desc="Final Test"):
                X, y = X.to(self.device), y.to(self.device)
                out = self.model(X)
                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        print("📊 Test Results:")
        print(classification_report(all_labels, all_preds, target_names=["non_drone", "drone"]))

        # Save confusion matrix
        if self.config.IN_COLAB:
            try:
                cm = confusion_matrix(all_labels, all_preds)
                plt.figure(figsize=(5,4))
                sns.heatmap(cm, annot=True, fmt='d', xticklabels=["non_drone","drone"], yticklabels=["non_drone","drone"])
                plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (Test)")
                cm_path = self.config.DRIVE_ROOT / "confusion_matrix.png"
                plt.tight_layout(); plt.savefig(cm_path, dpi=200); plt.close()
                print(f"📈 Confusion matrix saved: {cm_path}")
            except Exception as e:
                print(f"⚠️ Could not save confusion matrix: {e}")