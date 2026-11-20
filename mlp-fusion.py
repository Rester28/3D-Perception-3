import argparse
import random

import torch
import numpy as np

import matplotlib.pyplot as plt

#CLASS_MAPPING = {"Car": (1, 0, 0), "Pedestrian": (0, 1, 0), "Cyclist": (0, 0, 1)}
CLASS_MAPPING = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

class MLPFusion(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def load_training_set(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    x = []
    y = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split('|')
        if len(parts) != 3:
            raise ValueError(
                f"{file_path}: expected 3 pipe-separated fields, got {len(parts)}"
            )

        tmp = []
        gt_class = parts[0].strip()
        if gt_class not in CLASS_MAPPING:
            raise ValueError(
                f"{file_path}: unknown class {gt_class!r}"
            )

        for part in parts[1:]:
            tmp.extend(float(value) for value in part.strip().split())

        if len(tmp) != 6:
            raise ValueError(
                f"{file_path}: expected 6 feature values, got {len(tmp)}"
            )

        y.append(CLASS_MAPPING[gt_class])
        x.append(tmp)
    return x, y

def macro_f1_score(predictions, targets, num_classes):
    f1_scores = []
    for class_id in range(num_classes):
        predicted_class = predictions == class_id
        target_class = targets == class_id
        true_positive = (predicted_class & target_class).sum().float()
        false_positive = (predicted_class & ~target_class).sum().float()
        false_negative = (~predicted_class & target_class).sum().float()
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(2 * true_positive / denominator.clamp_min(1.0))

    return torch.stack(f1_scores).mean().item()

def confusion_matrix(predictions, targets, num_classes):
    encoded = targets * num_classes + predictions
    return torch.bincount(
        encoded,
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)

def accuracy(predictions, targets):
    if predictions.numel() == 0:
        return float("nan")
    return (predictions == targets).float().mean().item()

def fuse_detector_predictions(features):
    """Choose the class predicted with the larger detector maximum confidence.

    Features are ordered as [pp_car, pp_ped, pp_cyc,
                              yolo_car, yolo_ped, yolo_cyc].
    Returns predicted classes and the mean of the two maximum confidences.
    """
    features = torch.as_tensor(features, dtype=torch.float32)
    pp_confidence, pp_predictions = features[:, :3].max(dim=1)
    yolo_confidence, yolo_predictions = features[:, 3:].max(dim=1)
    predictions = torch.where(
        pp_confidence >= yolo_confidence,
        pp_predictions,
        yolo_predictions,
    )
    mean_max_confidence = (pp_confidence + yolo_confidence) / 2.0
    return predictions, mean_max_confidence

def plot_confusion_matrix(predictions, targets, num_classes, class_names, title):
    matrix = confusion_matrix(predictions, targets, num_classes).cpu().numpy()

    fig, ax = plt.subplots()
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=range(num_classes),
        yticks=range(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title=title,
    )

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(num_classes):
        for column in range(num_classes):
            ax.text(
                column,
                row,
                int(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    fig.tight_layout()

def train_model(x, y, x_val, y_val, input_dim, hidden_dim, output_dim,
                epochs=500, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPFusion(input_dim, hidden_dim, output_dim).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long, device=device)
    epochs_list = []
    training_losses = []
    validation_losses = []
    validation_f1_scores = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            training_loss = criterion(model(x_tensor), y_tensor)
            val_outputs = model(x_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_predictions = val_outputs.argmax(dim=1)
            val_f1 = macro_f1_score(val_predictions, y_val_tensor, output_dim)
            epochs_list.append(epoch + 1)
            training_losses.append(training_loss.item())
            validation_losses.append(val_loss.item())
            validation_f1_scores.append(val_f1)
            print(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"Training Loss: {training_loss.item():.4f}, "
                f"Validation Loss: {val_loss.item():.4f}, "
                f"Validation Macro-F1: {val_f1:.4f}"
            )

    plt.figure("Training and Validation Loss")
    plt.plot(epochs_list, training_losses, label="Training Loss")
    plt.plot(epochs_list, validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.figure("Validation Macro-F1")
    plt.plot(epochs_list, validation_f1_scores, label="Validation Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1 score")
    plt.ylim(0.0, 1.0)
    plt.title("Validation Macro-F1")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default="F:\\3D perception\\3 progetto\\trainingset.txt", help="Path to the training set")
    ap.add_argument("--valid", type=str, default="F:\\3D perception\\3 progetto\\validationset.txt", help="Path to the unblurred validation set")
    args = ap.parse_args()

    seed = 12345
    random.seed(seed)
    torch.manual_seed(seed)

    x, y = load_training_set(args.train)
    x_val, y_val = load_training_set(args.valid)
    model = train_model(
        x, y, x_val, y_val,
        input_dim=6, hidden_dim=32, output_dim=3,
        epochs=500, learning_rate=0.001,
    )
    model.eval()

    with torch.no_grad():
        val_ids = [line.strip() for line in open("F:\\3D perception\\3 progetto\\val.txt", 'r').readlines()]
        blur_k_size = [3, 5, 7, 9]
        blur_type = ["box", "gaussian"]
        device = next(model.parameters()).device
        class_names = [
            name
            for name, _ in sorted(CLASS_MAPPING.items(), key=lambda item: item[1])
        ]

        x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long, device=device)
        gt_counts = torch.bincount(y_val_tensor, minlength=3).cpu().numpy()

        mlp_val_predictions = model(x_val_tensor).argmax(dim=1)
        mlp_val_confusion = confusion_matrix(
            mlp_val_predictions, y_val_tensor, 3
        ).cpu().numpy()
        fusion_val_predictions, _ = fuse_detector_predictions(x_val_tensor)
        fusion_val_confusion = confusion_matrix(
            fusion_val_predictions, y_val_tensor, 3
        ).cpu().numpy()

        print("\nUnblurred validation confusion matrix - MLP:")
        print(mlp_val_confusion)
        print("\nUnblurred validation confusion matrix - max-confidence fusion:")
        print(fusion_val_confusion)
        plot_confusion_matrix(
            mlp_val_predictions, y_val_tensor, 3, class_names,
            "Unblurred Validation - MLP",
        )
        plot_confusion_matrix(
            fusion_val_predictions, y_val_tensor, 3, class_names,
            "Unblurred Validation - Max-confidence fusion",
        )

        modes = ("MLP", "Max-confidence fusion")
        for mode in modes:
            plt.figure(f"{mode} Macro-F1 vs. blur strength")
            plt.figure(f"{mode} Miss percentage vs. blur strength")

            for blur in blur_type:
                f1_by_kernel = []
                miss_by_kernel = {name: [] for name in class_names}

                for kernel in blur_k_size:
                    all_predictions = []
                    all_targets = []

                    for sample_id in val_ids:
                        sample_id = sample_id.strip()
                        if not sample_id:
                            continue
                        val_path = (
                            "F:\\3D perception\\3 progetto"
                            f"\\data_object_triple_conf_blur\\{blur}_{kernel}"
                            f"\\{sample_id}.txt"
                        )
                        features, targets = load_training_set(val_path)
                        if not features:
                            continue

                        features_tensor = torch.tensor(
                            features, dtype=torch.float32, device=device
                        )
                        targets_tensor = torch.tensor(
                            targets, dtype=torch.long, device=device
                        )
                        if mode == "MLP":
                            predictions = model(features_tensor).argmax(dim=1)
                        else:
                            predictions, _ = fuse_detector_predictions(features_tensor)

                        all_predictions.append(predictions)
                        all_targets.append(targets_tensor)

                    if not all_predictions:
                        f1_by_kernel.append(float("nan"))
                        for name in class_names:
                            miss_by_kernel[name].append(float("nan"))
                        continue

                    predictions = torch.cat(all_predictions)
                    targets = torch.cat(all_targets)
                    f1_by_kernel.append(
                        macro_f1_score(predictions, targets, 3)
                    )
                    blur_confusion = confusion_matrix(
                        predictions, targets, 3
                    ).cpu().numpy()

                    print(
                        f"\n{mode} - {blur.capitalize()} blur, "
                        f"kernel {kernel} confusion matrix:"
                    )
                    print(blur_confusion)
                    for class_id, class_name in enumerate(class_names):
                        correctly_found = int(blur_confusion[class_id, class_id])
                        gt_count = int(gt_counts[class_id])
                        miss_percentage = (
                            100.0 * (gt_count - correctly_found) / gt_count
                            if gt_count > 0
                            else float("nan")
                        )
                        miss_by_kernel[class_name].append(miss_percentage)
                        print(
                            f"{class_name}: miss {miss_percentage:.2f}% "
                            f"(validation GT {gt_count}, "
                            f"blurred correct {correctly_found})"
                        )

                    plot_confusion_matrix(
                        predictions, targets, 3, class_names,
                        f"{mode} - {blur.capitalize()} blur, kernel {kernel}",
                    )

                plt.figure(f"{mode} Macro-F1 vs. blur strength")
                plt.plot(
                    blur_k_size, f1_by_kernel,
                    marker="o", label=f"{blur} blur",
                )
                plt.figure(f"{mode} Miss percentage vs. blur strength")
                for class_name in class_names:
                    plt.plot(
                        blur_k_size, miss_by_kernel[class_name],
                        marker="o", label=f"{blur} blur - {class_name}",
                    )

            plt.figure(f"{mode} Macro-F1 vs. blur strength")
            plt.title(f"{mode} Macro-F1 vs. blur strength")
            plt.xlabel("Blur kernel size")
            plt.ylabel("Macro-F1 score")
            plt.xticks(blur_k_size)
            plt.ylim(0.0, 1.0)
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.figure(f"{mode} Miss percentage vs. blur strength")
            plt.title(f"{mode} Miss percentage vs. blur strength")
            plt.xlabel("Blur kernel size")
            plt.ylabel("Miss percentage")
            plt.xticks(blur_k_size)
            plt.ylim(0.0, 100.0)
            plt.grid(True, alpha=0.3)
            plt.legend()

    plt.show()