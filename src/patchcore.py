import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from dataset import load_data, WaferDataset

def run_patchcore(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 데이터 로드
    train_df, test_df = load_data(args.data_path)

    # Dataset 생성
    train_dataset = WaferDataset(train_df, method=args.preprocess)
    test_dataset = WaferDataset(test_df, method=args.preprocess)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # ResNet18 Feature Extractor
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()

    # 정상 feature 추출
    print("정상 feature 추출 중...")
    normal_features = []
    with torch.no_grad():
        for x in train_loader:
            x = x.to(device)
            feat = resnet(x)
            normal_features.append(feat.cpu().numpy())
    normal_features = np.concatenate(normal_features, axis=0)
    print(f"normal feature shape: {normal_features.shape}")

    # kNN 학습
    print("kNN 학습 중...")
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(normal_features)

    # 테스트 feature 추출 및 이상 점수 계산
    print("테스트 평가 중...")
    scores = []
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            feat = resnet(x).cpu().numpy()
            dist, _ = knn.kneighbors(feat)
            scores.extend(dist.mean(axis=1))

    scores = np.array(scores)
    labels = np.ones(len(test_df))  # 불량 = 1

    # 정상 테스트 데이터 추가
    normal_test_df = train_df.sample(n=2000, random_state=42).reset_index(drop=True)
    normal_test_dataset = WaferDataset(normal_test_df, method=args.preprocess)
    normal_test_loader = DataLoader(normal_test_dataset, batch_size=64, shuffle=False)

    normal_scores = []
    with torch.no_grad():
        for x in normal_test_loader:
            x = x.to(device)
            feat = resnet(x).cpu().numpy()
            dist, _ = knn.kneighbors(feat)
            normal_scores.extend(dist.mean(axis=1))

    all_scores = np.concatenate([np.array(normal_scores), scores])
    all_labels = np.concatenate([np.zeros(len(normal_scores)), labels])

    # AUROC 계산
    auroc = roc_auc_score(all_labels, all_scores)
    print(f"AUROC: {auroc:.4f}")

    # 결과 시각화
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.hist(np.array(normal_scores), bins=50, alpha=0.7, label='Normal')
    plt.hist(scores, bins=50, alpha=0.7, label='Anomaly')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.title(f'PatchCore Score Distribution ({args.preprocess})')
    plt.legend()
    plt.savefig(f'results/patchcore_dist_{args.preprocess}.png')
    print(f"결과 저장: results/patchcore_dist_{args.preprocess}.png")

    return auroc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/LSWMD.pkl')
    parser.add_argument('--preprocess', type=str, default='baseline',
                        choices=['baseline', 'clahe', 'binary', 'edge'])
    args = parser.parse_args()
    run_patchcore(args)
