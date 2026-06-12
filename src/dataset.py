import os
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset
import torch
from preprocess import preprocess

class WaferDataset(Dataset):
    def __init__(self, data, method='baseline', transform=None):
        self.data = data
        self.method = method
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wafer_map = self.data.iloc[idx]['waferMap']
        img = np.array(wafer_map, dtype=np.float32)

        # 0~1 정규화
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # 64x64 리사이즈
        img = cv2.resize(img, (64, 64))

        # 전처리 적용
        img = preprocess(img, method=self.method)

        # 3채널로 변환
        img = np.stack([img, img, img], axis=0)

        return torch.tensor(img, dtype=torch.float32)


def load_data(pkl_path):
    df = pd.read_pickle(pkl_path)

    # failureType 추출 (numpy.ndarray 형식)
    df['failureType'] = df['failureType'].apply(
        lambda x: x[0][0] if isinstance(x, np.ndarray) and len(x) > 0 and len(x[0]) > 0 else 'none'
    )

    # 정상 데이터 10,000개만 학습용으로
    train_df = df[df['failureType'] == 'none'].sample(n=10000, random_state=42).reset_index(drop=True)

    # 테스트: 라벨링된 불량 데이터 전체
    test_df = df[df['failureType'] != 'none'].reset_index(drop=True)

    print(f"Train (정상만): {len(train_df)}개")
    print(f"Test (불량 전체): {len(test_df)}개")
    print(f"불량 클래스 분포:\n{test_df['failureType'].value_counts()}")

    return train_df, test_df
