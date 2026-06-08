# WM-811K 웨이퍼 이상탐지 | 전처리 방법 비교 실험

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Dataset](https://img.shields.io/badge/Dataset-WM--811K-green)
![Task](https://img.shields.io/badge/Task-Anomaly%20Detection-orange)

## 프로젝트 개요

WM-811K 데이터셋을 활용하여 반도체 웨이퍼 맵에서 정상 웨이퍼와 불량 웨이퍼를 탐지하는 이상탐지 모델을 구현합니다.
정상 데이터만으로 학습하여 불량 패턴을 탐지하는 비지도 이상탐지 방식을 채택하며,
전처리 방법(CLAHE, 이진화, Canny Edge)과 모델(AutoEncoder, PatchCore)에 따른 성능 변화를 비교 실험합니다.

## 데이터셋

- 출처: [WM-811K (Kaggle)](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
- 전체 811,457개 중 라벨링된 172,950개 사용
- 학습: None(정상) 데이터만 사용
- 테스트: 정상 + 불량 패턴 전체 사용

### 클래스별 데이터 분포

| 클래스 | 데이터 수 | 비율 | 용도 |
|--------|-----------|------|------|
| None | 147,431 | 85.25% | 학습 + 테스트 |
| Edge-Ring | 9,682 | 5.60% | 테스트만 |
| Edge-Loc | 5,199 | 3.00% | 테스트만 |
| Center | 4,296 | 2.48% | 테스트만 |
| Loc | 3,597 | 2.08% | 테스트만 |
| Scratch | 1,194 | 0.69% | 테스트만 |
| Random | 866 | 0.50% | 테스트만 |
| Donut | 555 | 0.32% | 테스트만 |
| Near-Full | 149 | 0.09% | 테스트만 |

### 데이터 분할

- Train : Validation : Test = 6 : 2 : 2
- Train, Validation: None(정상) 데이터만 사용
- Test: 정상 + 불량 전체 사용

## 전처리 비교 실험

웨이퍼 이미지 특성에 맞는 전처리 방법을 비교하여 이상탐지 성능에 미치는 영향을 분석합니다.

| 실험 | 전처리 방법 | 설명 |
|------|------------|------|
| Exp-A | Baseline | 리사이즈 + 정규화만 |
| Exp-B | CLAHE | 적응형 히스토그램 평활화로 대비 강화 |
| Exp-C | 이진화 (Binary) | 픽셀값 임계값 기준으로 흑백 변환 |
| Exp-D | Canny Edge | 엣지 검출로 불량 경계선 강조 |

### 공통 전처리

- 리사이즈: 64×64
- 정규화: mean=[0.5], std=[0.5]

## 모델 및 실험 설계

| 실험 | 모델 | 전처리 | 비고 |
|------|------|--------|------|
| Exp-A | AutoEncoder | Baseline | 기준선 |
| Exp-B | AutoEncoder | CLAHE | 전처리 효과 확인 |
| Exp-C | AutoEncoder | Canny Edge | 전처리 효과 확인 |
| Exp-D | PatchCore (ResNet18) | Baseline | 모델 비교 |
| Exp-E | PatchCore (ResNet18) | CLAHE | 핵심 실험 |

### AutoEncoder 구조

    입력 이미지
        ↓
    [Encoder] Conv2d → BN → ReLU → MaxPool (반복)
        ↓
    Latent Vector
        ↓
    [Decoder] ConvTranspose2d → BN → ReLU (반복)
        ↓
    재구성 이미지
        ↓
    Reconstruction Error (MSE) → 정상/이상 판단

### PatchCore 구조

    입력 이미지
        ↓
    Pretrained ResNet18 (feature 추출)
        ↓
    정상 feature Memory Bank 저장
        ↓
    kNN으로 거리 계산
        ↓
    Anomaly Score → 정상/이상 판단

## 성능 평가

| 실험 | 모델 | 전처리 | AUROC | Inference Time |
|------|------|--------|-------|----------------|
| Exp-A | AutoEncoder | Baseline | - | - |
| Exp-B | AutoEncoder | CLAHE | - | - |
| Exp-C | AutoEncoder | Canny Edge | - | - |
| Exp-D | PatchCore | Baseline | - | - |
| Exp-E | PatchCore | CLAHE | - | - |

> 주요 지표: AUROC (임계값에 독립적인 이상탐지 평가 지표)

## 프로젝트 구조

    wafer-anomaly-detection/
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

    # 환경 설정 (AWS EC2)
    conda create -n anomaly_env python=3.10 -y
    conda activate anomaly_env
    pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm opencv-python

    # 데이터 다운로드
    kaggle datasets download -d qingyi/wm811k-wafer-map

    # 학습 (AutoEncoder)
    python src/train.py --model autoencoder --preprocess baseline
    python src/train.py --model autoencoder --preprocess clahe

    # 학습 (PatchCore)
    python src/train.py --model patchcore --preprocess baseline

    # 백그라운드 학습
    nohup python src/train.py --model autoencoder --preprocess clahe > logs/train.log 2>&1 &
    tail -f logs/train.log

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
| 2 | 데이터 구조 파악 및 전처리 구현 | 🔲 예정 |
| 3 | AutoEncoder 학습 (Baseline) | 🔲 예정 |
| 4 | 전처리 방법별 비교 실험 | 🔲 예정 |
| 5 | PatchCore 구현 및 비교 실험 | 🔲 예정 |
| 6 | 결과 시각화 및 최종 정리 | 🔲 예정 |

## 참고 자료

- [WM-811K Dataset (Kaggle)](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [교수님 실습 코드 (anomaly)](https://github.com/inhatcmin/anomaly.git)
