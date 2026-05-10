import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────

def get_transforms():
    """Define image transforms for training and testing."""
    train_transform_orig = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_transform_seg = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform_orig = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform_seg = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform_orig, train_transform_seg, test_transform_orig, test_transform_seg


# ──────────────────────────────────────────────
# Freeze helpers
# ──────────────────────────────────────────────

def freeze_except_kan_head(model):
    """Freeze every parameter except the KAN classification head."""
    for param in model.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = True

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Freeze] Total: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.2f}%) | "
          f"Frozen: {total-trainable:,}")
    return model


def unfreeze_last_blocks(model, num_blocks=2):
    """Unfreeze the last `num_blocks` transformer blocks for phase-2 fine-tuning."""
    for param in model.blocks[-num_blocks:].parameters():
        param.requires_grad = True

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Unfreeze last {num_blocks} blocks] Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    return model


# ──────────────────────────────────────────────
# One epoch helpers
# ──────────────────────────────────────────────

def _run_epoch_train(model, loader, criterion, optimizer, device):
    """Run one training epoch; return (avg_loss, accuracy)."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        images, labels, _ = batch
        inputs = {
            'original':  images['original'].to(device),
            'segmented': images['segmented'].to(device),
        }
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping — important for KAN spline stability
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


def _run_epoch_val(model, loader, criterion, device):
    """Run one validation epoch; return (avg_loss, accuracy)."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in loader:
            images, labels, _ = batch
            inputs = {
                'original':  images['original'].to(device),
                'segmented': images['segmented'].to(device),
            }
            labels  = labels.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


# ──────────────────────────────────────────────
# Main training function — two-phase strategy
# ──────────────────────────────────────────────

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, config):
    """
    Train the AT-ViT model using a two-phase strategy:

      Phase 1  — KAN head only (backbone fully frozen).
                 Fast convergence, prevents corrupting pretrained weights.

      Phase 2  — KAN head + last N transformer blocks (partial fine-tuning).
                 Allows the backbone to adapt to the new head.

    Config keys used:
        num_epochs_phase1   (int,   default 20)
        num_epochs_phase2   (int,   default 10)
        lr_phase1           (float, default 1e-3)
        lr_phase2_head      (float, default 5e-4)
        lr_phase2_backbone  (float, default 1e-5)
        weight_decay        (float, default 1e-4)
        unfreeze_blocks     (int,   default 2)
        device              (str)
        results_dir         (Path)
    """
    metrics = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
        'phase': [],                          # 1 or 2 — useful for plotting
    }
    final_model_path = config['results_dir'] / 'KAN_final_model.pth'
    best_val_acc     = 0.0
    best_model_path  = config['results_dir'] / 'KAN_best_model.pth'

    device           = config['device']
    num_epochs_p1    = config.get('num_epochs_phase1',  20)
    num_epochs_p2    = config.get('num_epochs_phase2',  10)
    unfreeze_blocks  = config.get('unfreeze_blocks',     2)
    lr_p1            = config.get('lr_phase1',         1e-3)
    lr_p2_head       = config.get('lr_phase2_head',    5e-4)
    lr_p2_backbone   = config.get('lr_phase2_backbone',1e-5)
    weight_decay     = config.get('weight_decay',      1e-4)

    # ── Phase 1: KAN head only ─────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  PHASE 1 — KAN head only ({num_epochs_p1} epochs)")
    print("="*60)

    model = freeze_except_kan_head(model)

    optimizer_p1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_p1,
        weight_decay=weight_decay,
    )
    scheduler_p1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p1, T_max=num_epochs_p1, eta_min=1e-5
    )

    for epoch in range(num_epochs_p1):
        train_loss, train_acc = _run_epoch_train(model, train_loader, criterion, optimizer_p1, device)
        val_loss,   val_acc   = _run_epoch_val(model, test_loader, criterion, device)
        scheduler_p1.step()

        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['phase'].append(1)

        print(f"[P1] Epoch [{epoch+1:>3}/{num_epochs_p1}]  "
              f"Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"           ↳ New best val acc: {best_val_acc:.2f}% — model saved.")

    # ── Phase 2: KAN head + last N blocks ─────────────────────────────────
    print("\n" + "="*60)
    print(f"  PHASE 2 — KAN head + last {unfreeze_blocks} blocks ({num_epochs_p2} epochs)")
    print("="*60)

    model = unfreeze_last_blocks(model, num_blocks=unfreeze_blocks)

    optimizer_p2 = optim.AdamW([
        {"params": model.blocks[-unfreeze_blocks:].parameters(), "lr": lr_p2_backbone},
        {"params": model.head.parameters(),                      "lr": lr_p2_head},
    ], weight_decay=weight_decay)

    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=num_epochs_p2, eta_min=1e-6
    )

    for epoch in range(num_epochs_p2):
        train_loss, train_acc = _run_epoch_train(model, train_loader, criterion, optimizer_p2, device)
        val_loss,   val_acc   = _run_epoch_val(model, test_loader, criterion, device)
        scheduler_p2.step()

        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['phase'].append(2)

        print(f"[P2] Epoch [{epoch+1:>3}/{num_epochs_p2}]  "
              f"Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"           ↳ New best val acc: {best_val_acc:.2f}% — model saved.")

    # ── Save final model ───────────────────────────────────────────────────
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model  → {final_model_path}")
    print(f"Best model   → {best_model_path}  (val acc: {best_val_acc:.2f}%)")

    return final_model_path, best_model_path, metrics