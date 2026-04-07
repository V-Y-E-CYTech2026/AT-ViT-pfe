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
from visualize import visualize_both_branches, visualize_original_branch_with_iou, summarize_attention_patterns, generate_gradcam_visualizations
from utils import setup_environment, load_config

def main():
    # Setup environment
    setup_environment(seed=42)
    
    # Load configuration
    config = load_config()
    print(f"Using device: {config['device']}")

    # Load dataset
    df = pd.read_csv(config['csv_path'])
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("train_test_set value counts:\n", df["train_test_set"].value_counts())
    print(f"{config['target_variable']} distribution in each set:")
    print("Train:\n", df[df["train_test_set"] == "train"][config['target_variable']].value_counts(normalize=True))
    print("Test:\n", df[df["train_test_set"] == "test"][config['target_variable']].value_counts(normalize=True))

    train_df = df[df["train_test_set"] == "train"].copy()
    test_df = df[df["train_test_set"] == "test"].copy()
    print("\nTrain size:", len(train_df))
    print("Test size:", len(test_df))

    # Get transforms
    train_transform_orig, train_transform_seg, test_transform_orig, test_transform_seg = get_transforms()

    # Create datasets
    train_dataset = DualInputPlantTraitDataset(
        train_df, config['original_img_dir'], config['segmented_img_dir'],
        transform_orig=train_transform_orig, transform_seg=train_transform_seg, subset='train',
        target_variable=config['target_variable']
    )
    test_dataset = DualInputPlantTraitDataset(
        test_df, config['original_img_dir'], config['segmented_img_dir'],
        transform_orig=test_transform_orig, transform_seg=test_transform_seg, subset='test',
        target_variable=config['target_variable']
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")

    # Initialize model
    num_classes = len(train_dataset.classes)
    print(f"Training model for {num_classes} classes: {train_dataset.classes}")
    base_model = timm.create_model(
        "hf_hub:timm/crossvit_base_240.in1k",
        pretrained=True,
        num_classes=num_classes
    )
    model = ATViT(base_model, num_classes).to(config['device'])
    print(f"Model architecture:\n{model}")

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Train model
    final_model_path, metrics = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, config)
    print(f"Final model saved to: {final_model_path}")

    # Evaluate on test dataset
    model.load_state_dict(torch.load(final_model_path))
    test_metrics, _, _ = evaluate_model(model, test_loader, criterion, config)
    print(f"Test Loss: {test_metrics['test_loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")

    # Evaluate on noisy datasets
    evaluate_noisy_datasets(model, test_df, criterion, config)

    # Generate visualizations
    print("Creating attention visualizations for both branches...")
    visualize_both_branches(model, test_loader, num_samples=453, config=config)

    print("Creating attention visualizations with IoU for original branch...")
    visualize_original_branch_with_iou(model, test_loader, num_samples=453, segmented_img_dir=config['segmented_img_dir'], config=config)
    
    print("Creating attention summary...")
    summarize_attention_patterns(model, test_loader, config=config)
    
    print("Generating Grad-CAM visualizations...")
    generate_gradcam_visualizations(model, test_loader, num_samples=453, config=config)
    
    print("All tasks completed!")

if __name__ == "__main__":
    main()