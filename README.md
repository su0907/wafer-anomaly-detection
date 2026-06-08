# WM-811K 웨이퍼 불량 패턴 분류 | 전처리 방법 비교 실험

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Dataset](https://img.shields.io/badge/Dataset-WM--811K-green)
![Task](https://img.shields.io/badge/Task-Classification-orange)

## 프로젝트 개요

WM-811K 데이터셋을 활용하여 반도체 웨이퍼 맵의 불량 패턴을 분류하는 딥러닝 모델을 구현합니다.
전처리 방법(CLAHE, 이진화, Canny Edge)에 따른 분류 성능 변화를 비교 실험하는 것을 목표로 합니다.

## 데이터셋

- 출처: [WM-811K (Kaggle)](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
- 전체 811,457개 중 라벨링된 172,950개 사용
- 9개 클래스: None, Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-Full, Random, Scratch

### 클래스별 데이터 분포

| 클래스 | 데이터 수 | 비율 |
|--------|-----------|------|
| None | 147,431 | 85.25% |
| Edge-Ring | 9,682 | 5.60% |
| Edge-Loc | 5,199 | 3.00% |
| Center | 4,296 | 2.48% |
| Loc | 3,597 | 2.08% |
| Scratch | 1,194 | 0.69% |
| Random | 866 | 0.50% |
| Donut | 555 | 0.32% |
| Near-Full | 149 | 0.09% |

### 데이터 분할

- Train : Validation : Test = 6 : 2 : 2
- 분할 후 Train 데이터에만 증강 적용 (Data Leakage 방지)

## 전처리 비교 실험

웨이퍼 이미지 특성에 맞는 전처리 방법을 비교하여 분류 성능에 미치는 영향을 분석합니다.

| 실험 | 전처리 방법 | 설명 |
|------|------------|------|
| Exp-A | Baseline | 리사이즈 + 정규화만 |
| Exp-B | CLAHE | 적응형 히스토그램 평활화로 대비 강화 |
| Exp-C | 이진화 (Binary) | 픽셀값 임계값 기준으로 흑백 변환 |
| Exp-D | Canny Edge | 엣지 검출로 불량 경계선 강조 |

### 공통 전처리

- 리사이즈: 64×64
- 정규화: mean=[0.5], std=[0.5]
- 증강: 랜덤 수평 플립, 밝기·대비 조절 (Train only)

## 모델

- ResNet18 (Transfer Learning, ImageNet pretrained)
- FC layer만 교체하여 9개 클래스 분류

## 성능 평가

| 실험 | 전처리 | Accuracy | F1-Score | Inference Time |
|------|--------|----------|----------|----------------|
| Exp-A | Baseline | - | - | - |
| Exp-B | CLAHE | - | - | - |
| Exp-C | 이진화 | - | - | - |
| Exp-D | Canny Edge | - | - | - |

> 주요 지표: Accuracy, F1-Score

## 프로젝트 구조

    wafer-defect-classification/
    ├── data/
    │   └── LSWMD.pkl
    ├── src/
    │   ├── dataset.py
    │   ├── preprocess.py
    │   ├── model.py
    │   ├── train.py
    │   └── evaluate.py
    ├── notebooks/
    │   └── experiment.ipynb
    ├── logs/
    ├── checkpoints/
    └── README.md

## 실행 방법

```bash
# 환경 설정 (AWS EC2)
conda create -n wafer_env python=3.10 -y
conda activate wafer_env
pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm opencv-python

# 데이터 다운로드
kaggle datasets download -d qingyi/wm811k-wafer-map

# 학습
python src/train.py --preprocess baseline
python src/train.py --preprocess clahe
python src/train.py --preprocess binary
python src/train.py --preprocess edge

# 백그라운드 학습
nohup python src/train.py --preprocess clahe > logs/train.log 2>&1 &
tail -f logs/train.log
```

## 개발 환경

| 항목 | 내용 |
|------|------|
| 언어 | Python 3.10 |
| 프레임워크 | PyTorch 2.x |
| 학습 환경 | AWS EC2 (g4dn.xlarge, T4 GPU) |
| 버전 관리 | Git / GitHub |

## 진행 계획

| 단계 | 내용 | 상태 |
|------|------|------|
| 1 | 주제 선정 및 데이터셋 확보 | ✅ 완료 |
| 2 | 데이터 구조 파악 및 전처리 파이프라인 구현 | 🔲 예정 |
| 3 | Baseline 모델 학습 | 🔲 예정 |
| 4 | 전처리 방법별 비교 실험 | 🔲 예정 |
| 5 | 결과 시각화 및 최종 정리 | 🔲 예정 |

## 참고 자료

- [WM-811K Dataset (Kaggle)](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
