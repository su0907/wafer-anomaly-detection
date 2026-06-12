import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
from dataset import load_data, WaferDataset
from model import ConvAutoEncoder

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 데이터 로드
    train_df, test_df = load_data(args.data_path)

    # 정상 데이터 일부를 테스트에 추가
    normal_test_df = train_df.sample(n=2000, random_state=42).reset_index(drop=True)

    # 테스트 데이터셋 구성 (정상 2000 + 불량 전체)
    import pandas as pd
    full_test_df = pd.concat([normal_test_df, test_df]).reset_index(drop=True)
    labels = [0] * len(normal_test_df) + [1] * len(test_df)  # 0=정상, 1=불량

    test_dataset = WaferDataset(full_test_df)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 모델 로드
    model = ConvAutoEncoder().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    criterion = nn.MSELoss(reduction='none')

    # 재구성 오차 계산
    scores = []
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            recon = model(x)
            loss = criterion(recon, x)
            loss = loss.mean(dim=[1, 2, 3])
            scores.extend(loss.cpu().numpy())

    scores = np.array(scores)
    labels = np.array(labels)

    # AUROC 계산
    auroc = roc_auc_score(labels, scores)
    print(f"AUROC: {auroc:.4f}")

    # 재구성 오차 분포 시각화
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.hist(scores[labels == 0], bins=50, alpha=0.7, label='Normal')
    plt.hist(scores[labels == 1], bins=50, alpha=0.7, label='Anomaly')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Count')
    plt.title(f'Reconstruction Error Distribution ({args.preprocess})')
    plt.legend()
    plt.savefig(f'results/error_dist_{args.preprocess}.png')
    print(f"결과 저장: results/error_dist_{args.preprocess}.png")

    return auroc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/LSWMD.pkl')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--preprocess', type=str, default='baseline',
                        choices=['baseline', 'clahe', 'binary', 'edge'])
    args = parser.parse_args()
    evaluate(args)
