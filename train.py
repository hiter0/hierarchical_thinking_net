import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import shufflenet_v2_x1_0, mobilenet_v3_large
from torchvision.datasets import ImageFolder
import logging
from datetime import datetime
import argparse
import os
from utils import *

def main(args):

    set_random_seed(args.seed_number)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 加载数据集
    train_dataset = ImageFolder(args.data_dir + 'train', transform=data_transforms['train'])
    val_dataset = ImageFolder(args.data_dir + 'val', transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    num_classes = len(train_dataset.classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = mobilenet_v3_large(weights=True)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model.to(device)
    # print_model_info(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # log
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = logging.getLogger("training_log")
    logger.setLevel(logging.INFO)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    log_filename = f"{args.output_path}/log_{current_time}.txt"
    file_handler = logging.FileHandler(log_filename)
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    args_str = ', '.join(f'{arg}={getattr(args, arg)}' for arg in vars(args))
    logger.info(f"Arguments: {args_str}")

    # 训练模型
    best_accuracy = 0
    best_weights_path = ''
    for epoch in range(args.num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 20 == 0:
                print(
                    f'Epoch [{epoch + 1}/{args.num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        scheduler.step()

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f'Epoch [{epoch + 1}/{args.num_epochs}], Test Accuracy: {accuracy:.2f}%')

        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            new_weights_path = f"{args.output_path}/{current_time}_{best_accuracy:.2f}.pth"
            torch.save(model.state_dict(), new_weights_path)
            if os.path.exists(best_weights_path):
                os.remove(best_weights_path)
            best_weights_path = new_weights_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_number', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--data_dir', type=str, default='../data/PlantDoc/')
    parser.add_argument('--output_path', type=str, default='output/weights')
    opt = parser.parse_args()

    main(opt)
