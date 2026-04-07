import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

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

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, config):
    """Train the AT-ViT model and track metrics."""
    metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    final_model_path = config['results_dir'] / 'KAN_final_model.pth'

    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            images, labels, _ = batch
            inputs = {
                'original': images['original'].to(config['device']),
                'segmented': images['segmented'].to(config['device'])
            }
            labels = labels.to(config['device'])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        metrics['train_loss'].append(epoch_loss)
        metrics['train_acc'].append(train_acc)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                images, labels, _ = batch
                inputs = {
                    'original': images['original'].to(config['device']),
                    'segmented': images['segmented'].to(config['device'])
                }
                labels = labels.to(config['device'])
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(test_loader)
        val_acc = 100 * correct / total
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)

        print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # Save the final model
    torch.save(model.state_dict(), final_model_path)
    return final_model_path, metrics