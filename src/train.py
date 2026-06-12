import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import argparse
from dataset import load_data, WaferDataset
from preprocess import preprocess
from model import ConvAutoEncoder

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 데이터 로드
    train_df, test_df = load_data(args.data_path)

    # Dataset 생성 (전처리 방법 적용)
    train_dataset = WaferDataset(train_df, method=args.preprocess)
    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_set, val_set = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # 모델 초기화
    model = ConvAutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 학습 로그
    log_path = f'logs/train_{args.preprocess}.log'
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    best_val_loss = float('inf')

    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        for x in train_loader:
            x = x.to(device)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                recon = model(x)
                loss = criterion(recon, x)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        # 로그 저장
        with open(log_path, 'a') as f:
            f.write(f'{epoch+1},{train_loss:.4f},{val_loss:.4f}\n')

        # Best 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'checkpoints/best_{args.preprocess}.pth')
            print(f"  Best model saved!")

    print(f"학습 완료! Best Val Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/LSWMD.pkl')
    parser.add_argument('--preprocess', type=str, default='baseline',
                        choices=['baseline', 'clahe', 'binary', 'edge'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
