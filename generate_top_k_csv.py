import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import argparse
import os
from torchvision.models import mobilenet_v3_large
import torch.nn as nn
import csv

def main(args):
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载测试数据集
    test_dataset = ImageFolder(args.data_dir + 'val', transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    num_classes = len(test_dataset.classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型列表
    model_list = []

    for weights_file in os.listdir(args.weights_dir):
        if weights_file.endswith('.pth'):
            model = mobilenet_v3_large(weights=None)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            model.load_state_dict(torch.load(os.path.join(args.weights_dir, weights_file)))
            model_list.append((model, weights_file))

    for model, weights_file in model_list:
        model = model.to(device)
        model.eval()

        with open(f'output/top_k/{weights_file}.csv', 'w', newline='') as csvfile:
            fieldnames = ['top1_class', 'top2_class']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for i, (images, labels) in enumerate(test_loader):
                images = images.to(device)

                with torch.no_grad():
                    outputs = model(images)
                    top2_prob, top2_classes = torch.topk(outputs, 2)
                    top2_classes = top2_classes.tolist()

                    for top2_class in top2_classes:
                        writer.writerow({'top1_class': top2_class[0], 'top2_class': top2_class[1]})

        print(f'Generate a csv file, Total Num:{len(model_list)}')

if __name__ == '__main__':
    # 确保您在这里添加了新的命令行参数，例如 '--weights_path'
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_dir', type=str, default='../data/PlantDoc/')
    parser.add_argument('--weights_dir', type=str, default='output/weights')
    opt = parser.parse_args()

    main(opt)
