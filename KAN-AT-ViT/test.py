import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dataset import DualInputPlantTraitDataset
from train import get_transforms

def evaluate_model(model, test_loader, criterion, config, dataset_name="test"):
    """Evaluate the model on a dataset and save metrics and predictions."""
    model.eval()
    test_loss = 0.0
    predictions = []
    true_labels = []
    codes = []

    with torch.no_grad():
        for batch, labels, img_codes in tqdm(test_loader, desc=f"Evaluating {dataset_name}"):
            images = {
                'original': batch['original'].to(config['device']),
                'segmented': batch['segmented'].to(config['device'])
            }
            labels = labels.to(config['device']).long()
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images['original'].size(0)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            codes.extend(img_codes)

    test_loss /= len(test_loader.dataset)
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)

    class_names = [str(cls) for cls in test_loader.dataset.classes]
    clf_report = classification_report(true_labels, predictions, target_names=class_names, output_dict=True)

    cm = confusion_matrix(true_labels, predictions)

    metrics = {
        "dataset_name": dataset_name,
        "test_loss": test_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": clf_report
    }

    # Save metrics
    metrics_path = config['results_dir'] / f"{dataset_name}_KAN_results.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics for {dataset_name} saved to: {metrics_path}")

    # Save predictions
    pred_df = pd.DataFrame({
        'code': codes,
        'true_label': true_labels,
        'predicted_label': predictions,
        'true_class': [test_loader.dataset.classes[label] for label in true_labels],
        'predicted_class': [test_loader.dataset.classes[pred] for pred in predictions]
    })
    pred_csv_path = config['results_dir'] / f"{dataset_name}_KAN_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"Predictions for {dataset_name} saved to: {pred_csv_path}")

    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_plot_path = config['results_dir'] / f"{dataset_name}_KAN_confusion_matrix.png"
    plt.savefig(cm_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix for {dataset_name} saved to: {cm_plot_path}")

    return metrics, cm, pred_df

def evaluate_noisy_datasets(model, test_df, criterion, config):
    """Evaluate the model on noisy datasets."""
    train_transform_orig, train_transform_seg, test_transform_orig, test_transform_seg = get_transforms()

    noisy_datasets = {
        "background_noise": DualInputPlantTraitDataset(
            test_df, config['background_noise_dir'], config['background_noise_seg_dir'],
            transform_orig=test_transform_orig, transform_seg=test_transform_seg, subset='test'
        ),
        "plant_noise": DualInputPlantTraitDataset(
            test_df, config['plant_noise_dir'], config['plant_noise_seg_dir'],
            transform_orig=test_transform_orig, transform_seg=test_transform_seg, subset='test'
        )
    }

    noisy_loaders = {
        name: DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        ) for name, dataset in noisy_datasets.items()
    }

    for dataset_name, loader in noisy_loaders.items():
        print(f"\nEvaluating on {dataset_name} dataset...")
        metrics, _, _ = evaluate_model(model, loader, criterion, config, dataset_name)
        print(f"{dataset_name} - Test Loss: {metrics['test_loss']:.4f}, "
              f"Accuracy: {metrics['accuracy']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, "
              f"F1 Score: {metrics['f1_score']:.4f}")