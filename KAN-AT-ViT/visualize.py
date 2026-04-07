import os
import json
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class DualInputWrapper(torch.nn.Module):
    """Wrapper for DualInputCrossVit to process only the original image for Grad-CAM."""
    def __init__(self, model):
        super(DualInputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        batch_size = x.shape[0]
        dummy_seg = torch.zeros(batch_size, 3, 224, 224, device=x.device)
        inputs = {'original': x, 'segmented': dummy_seg}
        return self.model(inputs)

def extract_attention_maps(model, inputs):
    """
    Extract attention maps from the final transformer layer of both branches.

    Args:
        model: Trained DualInputCrossVit model.
        inputs: Dictionary with 'original' and 'segmented' image tensors.

    Returns:
        tuple: (orig_attention, seg_attention) as numpy arrays (batch_size, 20, 20) and (batch_size, 14, 14).
    """
    model.eval()
    with torch.no_grad():
        orig_x = inputs['original']
        seg_x = inputs['segmented']
        batch_size = orig_x.shape[0]
        device = orig_x.device

        orig_tokens = model.patch_embed_orig(orig_x)
        seg_tokens = model.patch_embed_seg(seg_x)

        cls_tokens_orig = model.cls_token_orig.expand(batch_size, -1, -1)
        cls_tokens_seg = model.cls_token_seg.expand(batch_size, -1, -1)

        orig_tokens = torch.cat([cls_tokens_orig, orig_tokens], dim=1)
        seg_tokens = torch.cat([cls_tokens_seg, seg_tokens], dim=1)

        orig_tokens = orig_tokens + model.pos_embed_orig
        seg_tokens = seg_tokens + model.pos_embed_seg
        orig_tokens = model.pos_drop(orig_tokens)
        seg_tokens = model.pos_drop(seg_tokens)

        for block in model.blocks:
            orig_tokens, seg_tokens = block([orig_tokens, seg_tokens])

        orig_patch_features = model.norm[0](orig_tokens[:, 1:])
        seg_patch_features = model.norm[1](seg_tokens[:, 1:])

        orig_attention = torch.norm(orig_patch_features, dim=-1)
        seg_attention = torch.norm(seg_patch_features, dim=-1)

        orig_attention = (orig_attention - orig_attention.min(dim=1, keepdim=True)[0]) / \
                         (orig_attention.max(dim=1, keepdim=True)[0] - orig_attention.min(dim=1, keepdim=True)[0] + 1e-8)
        seg_attention = (seg_attention - seg_attention.min(dim=1, keepdim=True)[0]) / \
                        (seg_attention.max(dim=1, keepdim=True)[0] - seg_attention.min(dim=1, keepdim=True)[0] + 1e-8)

        orig_attention = orig_attention.view(batch_size, 20, 20)
        seg_attention = seg_attention.view(batch_size, 14, 14)

    return orig_attention.cpu().numpy(), seg_attention.cpu().numpy()

def compute_iou_score(attention_map, plant_mask, threshold=0.5):
    """
    Compute IoU between attention map and plant mask.

    Args:
        attention_map (np.array): Attention map (20x20 or resized).
        plant_mask (np.array): Binary mask of plant pixels (240x240).
        threshold (float): Threshold for binarizing attention map.

    Returns:
        float: IoU score between 0 and 1.
    """
    attention_map_resized = cv2.resize(attention_map, (240, 240), interpolation=cv2.INTER_LINEAR)
    attention_binary = (attention_map_resized > threshold).astype(np.uint8)

    if attention_binary.shape != plant_mask.shape:
        raise ValueError(f"Shape mismatch: attention_binary {attention_binary.shape} vs plant_mask {plant_mask.shape}")

    intersection = np.logical_and(attention_binary, plant_mask).sum()
    union = np.logical_or(attention_binary, plant_mask).sum()
    return intersection / union if union > 0 else 0.0

def overlay_heatmap(image, heatmap, alpha=0.4):
    """
    Overlay heatmap on image with a custom colormap.

    Args:
        image: PIL Image or numpy array (RGB or BGR).
        heatmap: 2D numpy array (attention map).
        alpha: Transparency of heatmap overlay.

    Returns:
        np.array: Overlayed image.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = image
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (21, 21), 0)

    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    heatmap_norm = np.where(heatmap_norm > 0.3, heatmap_norm, 0)
    if heatmap_norm.max() > 0:
        heatmap_norm = heatmap_norm / heatmap_norm.max()

    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    attention_mask = heatmap_norm
    mask_3d = np.stack([attention_mask] * 3, axis=-1)
    local_alpha = alpha * mask_3d

    overlayed = image_rgb.astype(np.float32) * (1 - local_alpha) + heatmap_colored.astype(np.float32) * local_alpha
    return np.clip(overlayed, 0, 255).astype(np.uint8)

def denormalize_image_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor for visualization.

    Args:
        tensor: Input tensor (C, H, W).
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.

    Returns:
        np.array: Denormalized image as uint8.
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    denorm = tensor.cpu() * std + mean
    denorm = torch.clamp(denorm, 0, 1)
    return (denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

def visualize_both_branches(model, test_loader, num_samples=453, save_dir=None, config=None):
    """
    Visualize attention maps for both original and segmented branches.

    Args:
        model: Trained DualInputCrossVit model.
        test_loader: DataLoader for test dataset.
        num_samples: Number of samples to visualize.
        save_dir: Directory to save visualizations.
        config: Configuration dictionary with 'device' key.
    """
    save_dir = save_dir or os.path.join(config.get('results_dir', 'results'), "kan_at_vit_both_branches_visualizations")
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    samples_processed = 0

    with torch.no_grad():
        for batch_idx, (images, labels, codes) in enumerate(test_loader):
            if samples_processed >= num_samples:
                break

            inputs = {
                'original': images['original'].to(config['device']),
                'segmented': images['segmented'].to(config['device'])
            }
            labels = labels.to(config['device'])

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            orig_attention, seg_attention = extract_attention_maps(model, inputs)
            batch_size = inputs['original'].shape[0]

            for i in range(min(batch_size, num_samples - samples_processed)):
                orig_img = denormalize_image_tensor(inputs['original'][i])
                seg_img = denormalize_image_tensor(inputs['segmented'][i])

                orig_overlay = overlay_heatmap(orig_img, orig_attention[i])
                seg_overlay = overlay_heatmap(seg_img, seg_attention[i])

                true_class = test_loader.dataset.classes[labels[i].cpu().item()]
                pred_class = test_loader.dataset.classes[predicted[i].cpu().item()]

                fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                axes[0].imshow(orig_img)
                axes[0].set_title('Original Image')
                axes[0].axis('off')

                axes[1].imshow(orig_overlay)
                axes[1].set_title('Original + Attention')
                axes[1].axis('off')

                axes[2].imshow(seg_img)
                axes[2].set_title('Segmented Image')
                axes[2].axis('off')

                axes[3].imshow(seg_overlay)
                axes[3].set_title('Segmented + Attention')
                axes[3].axis('off')

                fig.suptitle(f'Code: {codes[i]} | True: {true_class} | Predicted: {pred_class}',
                             fontsize=12, fontweight='bold')

                plt.tight_layout()
                save_path = os.path.join(save_dir, f'{codes[i]}.png')
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close()

                samples_processed += 1
                if samples_processed >= num_samples:
                    break

    print(f"Saved {samples_processed} attention visualizations to {save_dir}")

def summarize_attention_patterns(model, test_loader, save_dir=None, config=None):
    """
    Summarize attention patterns across correct and incorrect predictions for both branches.

    Args:
        model: Trained DualInputCrossVit model.
        test_loader: DataLoader for test dataset.
        save_dir: Directory to save summary visualization.
        config: Configuration dictionary with 'results_dir' key.
    """
    save_dir = save_dir or os.path.join(config.get('results_dir', 'results'), "kan_at_vit_attention_summaries")
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    correct_attentions_orig, correct_attentions_seg = [], []
    incorrect_attentions_orig, incorrect_attentions_seg = [], []

    with torch.no_grad():
        for images, labels, codes in tqdm(test_loader, desc="Computing attention summary"):
            inputs = {
                'original': images['original'].to(config['device']),
                'segmented': images['segmented'].to(config['device'])
            }
            labels = labels.to(config['device'])

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            orig_attention, seg_attention = extract_attention_maps(model, inputs)

            for i in range(inputs['original'].shape[0]):
                is_correct = predicted[i].cpu().item() == labels[i].cpu().item()
                if is_correct:
                    correct_attentions_orig.append(orig_attention[i])
                    correct_attentions_seg.append(seg_attention[i])
                else:
                    incorrect_attentions_orig.append(orig_attention[i])
                    incorrect_attentions_seg.append(seg_attention[i])

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    colors = ['#000080', '#00CCFF', '#00FFFF', '#FFFF00', '#FF9900', '#FF0000']
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)

    if correct_attentions_orig:
        avg_correct_orig = np.mean(correct_attentions_orig, axis=0)
        avg_correct_seg = np.mean(correct_attentions_seg, axis=0)
        im1 = axes[0, 0].imshow(avg_correct_orig, cmap=custom_cmap)
        axes[0, 0].set_title('Avg Attention - Correct (Original)')
        plt.colorbar(im1, ax=axes[0, 0])

        im2 = axes[0, 1].imshow(avg_correct_seg, cmap=custom_cmap)
        axes[0, 1].set_title('Avg Attention - Correct (Segmented)')
        plt.colorbar(im2, ax=axes[0, 1])

    if incorrect_attentions_orig:
        avg_incorrect_orig = np.mean(incorrect_attentions_orig, axis=0)
        avg_incorrect_seg = np.mean(incorrect_attentions_seg, axis=0)
        im3 = axes[1, 0].imshow(avg_incorrect_orig, cmap=custom_cmap)
        axes[1, 0].set_title('Avg Attention - Incorrect (Original)')
        plt.colorbar(im3, ax=axes[1, 0])

        im4 = axes[1, 1].imshow(avg_incorrect_seg, cmap=custom_cmap)
        axes[1, 1].set_title('Avg Attention - Incorrect (Segmented)')
        plt.colorbar(im4, ax=axes[1, 1])

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    summary_path = os.path.join(save_dir, 'kan_at_vit_attention_summary.png')
    plt.savefig(summary_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"Attention summary saved to {summary_path}")
    print(f"Processed {len(correct_attentions_orig)} correct and {len(incorrect_attentions_orig)} incorrect predictions")

def visualize_original_branch_with_iou(model, test_loader, num_samples=453, save_dir=None, segmented_img_dir=None, config=None):
    """
    Visualize attention maps for the original branch with IoU calculation.

    Args:
        model: Trained DualInputCrossVit model.
        test_loader: DataLoader for test dataset.
        num_samples: Number of samples to visualize.
        save_dir: Directory to save visualizations.
        segmented_img_dir: Directory containing segmented images.
        config: Configuration dictionary with 'device' and 'results_dir' keys.
    """
    save_dir = os.path.join(config.get('results_dir', 'results'), "kan_at_vit_original_branch_iou_visualizations")
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    iou_results = []
    samples_processed = 0

    with torch.no_grad():
        for batch_idx, (images, labels, codes) in enumerate(test_loader):
            if samples_processed >= num_samples:
                break

            inputs = {
                'original': images['original'].to(config['device']),
                'segmented': images['segmented'].to(config['device'])
            }
            labels = labels.to(config['device'])

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            orig_attention, _ = extract_attention_maps(model, inputs)
            batch_size = inputs['original'].shape[0]

            for i in range(min(batch_size, num_samples - samples_processed)):
                orig_img = denormalize_image_tensor(inputs['original'][i])
                orig_attn = orig_attention[i]
                orig_overlay = overlay_heatmap(orig_img, orig_attn)

                true_class = test_loader.dataset.classes[labels[i].cpu().item()]
                pred_class = test_loader.dataset.classes[predicted[i].cpu().item()]
                code = codes[i]

                seg_path = os.path.join(segmented_img_dir, f"{code}.jpg")
                seg_img_mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
                if seg_img_mask is None:
                    print(f"Warning: Failed to load segmented image for {code}, skipping IoU.")
                    continue
                plant_mask = (seg_img_mask > 0).astype(np.uint8)
                plant_mask = cv2.resize(plant_mask, (240, 240), interpolation=cv2.INTER_NEAREST)

                iou_score = compute_iou_score(orig_attn, plant_mask)
                iou_results.append({"code": code, "iou": iou_score})

                fig, axes = plt.subplots(1, 2, figsize=(8, 3))
                axes[0].imshow(orig_img)
                axes[0].set_title('Original Image')
                axes[0].axis('off')

                axes[1].imshow(orig_overlay)
                axes[1].set_title('Original + Attention')
                axes[1].axis('off')

                fig.suptitle(f'Code: {code} | True: {true_class} | Predicted: {pred_class} | IoU: {iou_score:.4f}',
                             fontsize=10, fontweight='bold')

                plt.tight_layout()
                save_path = os.path.join(save_dir, f'{code}.png')
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close()

                samples_processed += 1
                if samples_processed >= num_samples:
                    break

    avg_iou = np.mean([result["iou"] for result in iou_results])
    print(f"Average IoU across {samples_processed} images: {avg_iou:.4f}")
    iou_path = os.path.join(save_dir, "kan_at_vit_iou_results.json")
    iou_output = {"samples": iou_results, "average_iou": float(avg_iou)}
    with open(iou_path, "w") as f:
        json.dump(iou_output, f, indent=4)
    print(f"IoU results saved to: {iou_path}")

def generate_gradcam_visualizations(model, test_loader, num_samples=453, save_dir=None, output_size=(240, 240), config=None):
    """
    Generate Grad-CAM visualizations for the original branch at the patch embedding level.

    Args:
        model: Trained DualInputCrossVit model.
        test_loader: DataLoader for test dataset.
        num_samples: Number of samples to process.
        save_dir: Directory to save visualizations.
        output_size: Tuple of (width, height) for output images.
        config: Configuration dictionary with 'device' and 'results_dir' keys.
    """
    save_dir = save_dir or os.path.join(config.get('results_dir', 'results'), "kan_at_vit_gradcam_visualizations")
    os.makedirs(save_dir, exist_ok=True)

    wrapped_model = DualInputWrapper(model).to(config['device'])
    target_layers = [model.patch_embed_orig.proj]
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    model.eval()
    wrapped_model.eval()
    processed = 0

    for batch, labels, codes in tqdm(test_loader, desc="Generating Grad-CAM"):
        images = {
            'original': batch['original'].to(config['device']),
            'segmented': batch['segmented'].to(config['device'])
        }
        labels = labels.to(config['device'])

        for i in range(images['original'].size(0)):
            if processed >= num_samples:
                break

            input_tensor = images['original'][i:i+1]
            seg_tensor = images['segmented'][i:i+1]
            code = codes[i]
            true_label = labels[i].item()
            true_class = test_loader.dataset.classes[true_label]

            with torch.no_grad():
                input_dict = {'original': input_tensor, 'segmented': seg_tensor}
                output = model(input_dict)
                pred_label = torch.argmax(output, dim=1).item()
                pred_class = test_loader.dataset.classes[pred_label]

            input_tensor = input_tensor.requires_grad_(True)
            target = [ClassifierOutputTarget(pred_label)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=target)[0, :]

            seg_weight_scalars = model.compute_patch_weights_from_segmented(seg_tensor, config['device'])
            orig_weight_scalars = model.upsample_weight_scalars(seg_weight_scalars, config['device'])
            weight_map = orig_weight_scalars.view(1, 20, 20).detach().cpu().numpy()[0]
            weight_map = (weight_map - weight_map.min()) / (weight_map.max() - weight_map.min() + 1e-8)
            weight_map = cv2.resize(weight_map, (240, 240), interpolation=cv2.INTER_LINEAR)

            modulated_cam = grayscale_cam * weight_map
            modulated_cam = (modulated_cam - modulated_cam.min()) / (modulated_cam.max() - modulated_cam.min() + 1e-8)

            original_img = input_tensor.detach().cpu()
            original_img = inv_normalize(original_img[0]).permute(1, 2, 0).numpy()
            original_img = np.clip(original_img, 0, 1)

            cam_image = show_cam_on_image(original_img, modulated_cam, use_rgb=True)
            cam_image = cv2.resize(cam_image, output_size, interpolation=cv2.INTER_LINEAR)

            text_area_height = 40
            canvas_height = output_size[1] + text_area_height
            canvas = np.zeros((canvas_height, output_size[0], 3), dtype=np.uint8)
            canvas[:text_area_height, :] = [255, 255, 255]
            canvas[text_area_height:, :] = cam_image

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (0, 0, 0)
            thickness = 1
            line_type = cv2.LINE_AA

            cv2.putText(canvas, f"True: {true_class}", (5, 20), font, font_scale, font_color, thickness, line_type)
            cv2.putText(canvas, f"Pred: {pred_class}", (5, 35), font, font_scale, font_color, thickness, line_type)

            save_path = os.path.join(save_dir, f"{code}.png")
            cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            processed += 1

        if processed >= num_samples:
            break

    print(f"Saved {processed} Grad-CAM visualizations to {save_dir}")