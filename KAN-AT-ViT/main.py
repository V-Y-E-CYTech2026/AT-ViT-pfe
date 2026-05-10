import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm

from model import ATViT
from dataset import DualInputPlantTraitDataset
from train import train_model, get_transforms
from test import evaluate_model, evaluate_noisy_datasets
from visualize import (
    visualize_both_branches,
    visualize_original_branch_with_iou,
    summarize_attention_patterns,
    generate_gradcam_visualizations,
)
from utils import setup_environment, load_config


def main():
    # ── Environment & config ───────────────────────────────────────────────
    setup_environment(seed=42)
    config = load_config()
    print(f"Using device: {config['device']}")

    # ── Two-phase training hyper-parameters ───────────────────────────────
    # These extend (or override) whatever load_config() already provides.
    # Adjust values here or move them into your config file / YAML.
    config.setdefault('num_epochs_phase1',  20)   # KAN head only
    config.setdefault('num_epochs_phase2',  10)   # head + last N blocks
    config.setdefault('unfreeze_blocks',     2)   # how many blocks to unfreeze in phase 2
    config.setdefault('lr_phase1',         1e-3)  # aggressive LR — backbone is frozen
    config.setdefault('lr_phase2_head',    5e-4)  # head LR for phase 2
    config.setdefault('lr_phase2_backbone',1e-5)  # very small LR for unfrozen blocks
    config.setdefault('weight_decay',      1e-4)

    # ── Dataset ───────────────────────────────────────────────────────────
    df = pd.read_csv(config['csv_path'])
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("train_test_set value counts:\n", df["train_test_set"].value_counts())
    print(f"{config['target_variable']} distribution in each set:")
    print("Train:\n", df[df["train_test_set"] == "train"][config['target_variable']].value_counts(normalize=True))
    print("Test:\n",  df[df["train_test_set"] == "test"][config['target_variable']].value_counts(normalize=True))

    train_df = df[df["train_test_set"] == "train"].copy()
    test_df  = df[df["train_test_set"] == "test"].copy()
    print(f"\nTrain size: {len(train_df)} | Test size: {len(test_df)}")

    # ── Transforms & loaders ──────────────────────────────────────────────
    train_transform_orig, train_transform_seg, test_transform_orig, test_transform_seg = get_transforms()

    train_dataset = DualInputPlantTraitDataset(
        train_df, config['original_img_dir'], config['segmented_img_dir'],
        transform_orig=train_transform_orig, transform_seg=train_transform_seg,
        subset='train', target_variable=config['target_variable'],
    )
    test_dataset = DualInputPlantTraitDataset(
        test_df, config['original_img_dir'], config['segmented_img_dir'],
        transform_orig=test_transform_orig, transform_seg=test_transform_seg,
        subset='test', target_variable=config['target_variable'],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True,  pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=False, pin_memory=torch.cuda.is_available(),
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset:  {len(test_dataset)} samples")

    # ── Model ─────────────────────────────────────────────────────────────
    num_classes = len(train_dataset.classes)
    print(f"\nTraining model for {num_classes} classes: {train_dataset.classes}")

    base_model = timm.create_model(
        "hf_hub:timm/crossvit_base_240.in1k",
        pretrained=True,
        num_classes=num_classes,
    )
    model = ATViT(base_model, num_classes).to(config['device'])
    print(f"Model architecture:\n{model}")

    # ── Loss ──────────────────────────────────────────────────────────────
    # optimizer and scheduler are now built inside train_model per phase;
    # we pass None as placeholders so the signature stays compatible with
    # any external code that still forwards them.
    criterion = nn.CrossEntropyLoss()

    # ── Training (two-phase) ──────────────────────────────────────────────
    final_model_path, best_model_path, metrics = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=None,   # built internally per phase
        scheduler=None,   # built internally per phase
        config=config,
    )
    print(f"\nFinal model path : {final_model_path}")
    print(f"Best  model path : {best_model_path}")

    # ── Evaluation — use best checkpoint ──────────────────────────────────
    print("\n--- Evaluating best model on test set ---")
    model.load_state_dict(torch.load(best_model_path, map_location=config['device']))
    test_metrics, _, _ = evaluate_model(model, test_loader, criterion, config)
    print(f"Test Loss      : {test_metrics['test_loss']:.4f}")
    print(f"Test Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"Test Precision : {test_metrics['precision']:.4f}")
    print(f"Test Recall    : {test_metrics['recall']:.4f}")
    print(f"Test F1        : {test_metrics['f1_score']:.4f}")

    # ── Noisy evaluation ──────────────────────────────────────────────────
    evaluate_noisy_datasets(model, test_df, criterion, config)

    # ── Visualisations ────────────────────────────────────────────────────
    print("\nCreating attention visualizations for both branches...")
    visualize_both_branches(model, test_loader, num_samples=453, config=config)

    print("Creating attention visualizations with IoU for original branch...")
    visualize_original_branch_with_iou(
        model, test_loader, num_samples=453,
        segmented_img_dir=config['segmented_img_dir'], config=config,
    )

    print("Creating attention summary...")
    summarize_attention_patterns(model, test_loader, config=config)

    print("Generating Grad-CAM visualizations...")
    generate_gradcam_visualizations(model, test_loader, num_samples=453, config=config)

    print("\nAll tasks completed!")


if __name__ == "__main__":
    main()