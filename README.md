# dfu_multimodal

DFU(당뇨발궤양) 이미지 + 임상 데이터 멀티모달 모델. `--fusion-strategy` 인자 하나로
**concat / gated / gbdt** fusion 방식을 바꿔가며 같은 데이터·같은 태스크·같은 평가 지표
위에서 비교할 수 있도록 만들어졌다.

- 예측 태스크: Wagner grade(6등급 ordinal, CORAL head) + SINBAD 6개 component(binary) — 총 7개.
  `amputation_risk`는 라벨링 근거가 애매하고 예측 실익도 불확실해 의도적으로 제외했다.
- 이미지 인코더: DINOv3 ViT-S/16, **항상 frozen** (forward pass는 학습 루프 밖에서 한 번만
  실행되고 embedding은 parquet에 캐싱됨).
- 환자(`id`) 단위 split만 사용한다 — 같은 환자의 여러 방문·여러 사진이 train/val에 걸쳐
  섞이지 않는다. 세 fusion 전략 모두 **같은 seed, 같은 val-ratio면 완전히 같은 환자 집합**으로
  나뉘도록 보장되어 있어(`trainers/common.py: split_by_group`), 전략 간 성능 비교가 공정하다.
- CSV·이미지·에셋까지 `dfu_multimodal/` 한 폴더 안에 다 들어있는 자기완결 프로젝트다
  (다른 `dfu_*` 패키지를 import하지도 않는다).
- 설계 배경/단계별 진행 기록은 [Project.md](./Project.md) 참고.

## 목차

- [디렉토리 구조](#디렉토리-구조)
- [설치](#설치)
- [테스트 (설치 후 가장 먼저 할 것)](#테스트-설치-후-가장-먼저-할-것)
- [데이터 준비](#데이터-준비)
- [DINOv3 에셋 준비](#dinov3-에셋-준비)
- [실행](#실행)
  - [concat / gated (neural 경로)](#concat--gated-neural-경로)
  - [gbdt (완전 별도 파이프라인)](#gbdt-완전-별도-파이프라인)
  - [세 전략 결과 비교하기](#세-전략-결과-비교하기)
- [출력물](#출력물)
- [새 fusion 전략 추가하기](#새-fusion-전략-추가하기)
- [결측치 처리 정책](#결측치-처리-정책)

## 디렉토리 구조

```
dfu_multimodal/
├── README.md                   
├── requirements.txt
├── train_fusion.py                 # 학습 진입점 — --fusion-strategy로 neural/gbdt 분기
│
├── dataset/                        # ★ clinical CSV + 이미지. git에는 CSV만
│   ├── model_input_variables_full.csv
│   ├── model_input_variables_required.csv
│   └── toy_image_dataset/          # "데이터 준비" 절의 다운로드 링크로 받을 것
│
├── assets/
│   └── dinov3-hf/                  # 로컬 DINOv3 HF 스냅샷 (config.json, model.safetensors, ...)
│                                    # "DINOv3 에셋 준비" 절 참고
│
├── config.py                       
│
├── cli/
│   └── fusion_args.py              
│
├── data/
│   ├── clinical_loader.py          # raw CSV 로딩 
│   ├── clinical_transform.py       # clinical feature 표준화 + median 대체 + 결측 flag
│   │                         
│   ├── image_dataset.py            # image_root/{id}/{visit}/{파일명} 구조 Dataset
│   ├── embedding_cache.py          # DINOv3 embedding을 (id, visit, img2d) 키로 parquet 캐싱
│   ├── fusion_dataset.py           # neural 경로(concat/gated) 전용 Dataset/collate
│   └── (gbdt 경로는 clinical_transform을 거치지 않고 raw 값을 그대로 씀 — 아래 참고)
│
├── models/
│   ├── dinov3_backbone.py          # frozen DINOv3ViTModel wrapper 
│   ├── image_projection.py         
│   ├── clinical_mlp.py            
│   ├── coral_head.py               # Wagner용 rank-consistent ordinal head 
│   ├── fusion_model.py             # DFUMultimodalModel — image/clinical branch + fusion + heads
│   └── fusion_strategies/          # ★ fusion 방식 레지스트리
│       ├── base.py                 
│       ├── concat_fusion.py        
│       ├── gated_fusion.py        
│       └── __init__.py            
│
├── trainers/
│   ├── common.py                   # 전략 공통 진입점
│   ├── losses.py                 
│   ├── metrics.py                  # classification_metrics — neural/gbdt 공통 평가 스키마
│   └── neural_trainer.py           # concat/gated 학습 루프
│
├── gbdt/                           # GBDT 경로 — neural과 완전히 별도인 파이프라인
│   ├── tasks.py                   
│   ├── tabular_builder.py         #   clinical raw feature + image embedding -> flat feature 목록
│   └── gbdt_trainer.py            #   CatBoost/XGBoost/LightGBM 공통 인터페이스, 태스크별 독립 학습
│
└── tests/                          # 모의 test 코드 - 실제 가중치/데이터 없이 실행 가능
    ├── mock_backbone.py           
    ├── synthetic_data.py          
    ├── test_concat_fusion_smoke.py  
    ├── test_gated_fusion_smoke.py   
    └── test_gbdt_fusion_smoke.py    
```

## 설치

```powershell
cd D:\dfu\multimodal\dfu_multimodal
..\venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt`는 neural 경로에 필요한 패키지(`torch`, `transformers` 등)에 더해
`--fusion-strategy gbdt`용 백엔드 3종(`catboost`/`xgboost`/`lightgbm`)과 테스트용
`pytest`까지 포함한다. GBDT 백엔드를 하나만 쓸 계획이면 나머지 둘은 지워도 된다.

## 테스트 (설치 후 진행 권장)

`tests/`는 실제 DINOv3 가중치나 실 환자 데이터 없이 mock encoder + 합성 데이터로 전체
파이프라인(이미지 로딩, embedding 캐시, 환자 단위 split, concat/gated/gbdt 학습까지)을
검증한다. 그래서 **아래 "데이터 준비"/"DINOv3 에셋 준비"를 하기 전에, `pip install`만
끝낸 상태에서 먼저 돌려보는 걸 권장한다** — torch/transformers 버전 문제 같은 환경 이슈를
462MB 이미지를 받거나 실제 학습을 시작하기 전에 몇 초 만에 잡아낼 수 있다.

```powershell
cd D:\dfu\multimodal\dfu_multimodal
..\venv\Scripts\python.exe -m pytest tests -v
```

9개 테스트가 전부 통과하면 환경은 문제없다는 뜻이고, 그다음 "데이터 준비"로 넘어가면 된다.

## 데이터 준비

`dataset/`에 두 가지가 들어있다.

1. **clinical CSV** (git에 포함됨) — `model_input_variables_full.csv` /
   `model_input_variables_required.csv`. `id, visit, img2d, <clinical feature 컬럼들>,
   <라벨 컬럼들>`을 포함하는 raw 상태 CSV다. **NaN이 유지된 원본이어야 한다** — median 대체나
   표준화가 이미 적용된 CSV를 넣으면 그 통계가 전체 데이터 기준으로 계산된 것이라 train/val
   분리 이전에 leakage가 생긴다.
   - 라벨 컬럼: `wag`(Wagner 0~5), `snb_site`, `snb_isc`, `snb_neuro`, `snb_bac`, `snb_area`,
     `snb_depth`(SINBAD 6개 component, 각 0/1).
   - `img2d`는 bare 파일명이든 `{id}/{visit}/{파일명}` 상대경로든 상관없다 — 내부적으로
     파일명만 뽑아 `image_root/{id}/{visit}/{파일명}`을 다시 조립하므로 안전하다.
2. **이미지**(`dataset/toy_image_dataset/`, **git에는 포함되지 않음** 
   아래 Google Drive 링크에서 받아 `dataset/toy_image_dataset/`에
   그대로 풀어놓으면 된다. 실제 파일은 반드시 `{image-root}/{id}/{visit}/{파일명}` 구조여야
   하고, 환자 한 명이 여러 방문(visit)을, 한 방문이 여러 장의 사진을 가질 수 있다.

   > 📎 **다운로드**: `https://drive.google.com/file/d/13YEhjZ8KvkU7stoZ3j7IPzp9_a2j9p7b/view?usp=sharing`
   >
   > 다운로드한 압축을 풀면 `dfu_multimodal/dataset/toy_image_dataset/0001/0/...` 형태가 되도록
   > `dataset/` 바로 아래에 풀 것.

`--clinical-csv`/`--image-root`의 기본값이 이미 `dataset/model_input_variables_full.csv`,
`dataset/toy_image_dataset`을 가리키므로, 이 구조 그대로면 CLI에서 생략해도 된다.

## DINOv3 에셋 준비

`assets/dinov3-hf/`에 로컬 Hugging Face 스냅샷(`config.json`, `model.safetensors`,
`preprocessor_config.json` 등)을 준비해야 한다(이것도 git에는 포함되지 않음). `--weights-path`
기본값이 이 경로를 가리킨다. 이미 변환된 스냅샷이 있다면 그대로 복사해 넣으면 되고, Meta 원본
`.pth` 체크포인트만 있다면 별도 변환 스크립트로 먼저 HF 포맷으로 바꿔야 한다(이 저장소에는
포함돼 있지 않음 — 이미 한 번 끝난 변환 결과물을 재사용하는 것을 전제로 한다).

다른 위치의 스냅샷을 쓰려면 `--weights-path <경로>`로 덮어쓰면 된다.

## 실행

모든 명령어는 `dfu_multimodal/` 디렉토리 **안에서** 실행한다.

### concat / gated (neural 경로)

```powershell
cd D:\dfu\multimodal\dfu_multimodal
..\venv\Scripts\python.exe -m train_fusion `
  --resolution 224 `
  --fusion-strategy concat `
  --run-name concat_v1 `
  --epochs 30 --batch-size 32 --lr 5e-4 `
  --val-ratio 0.2 --seed 42 `
  --device cuda
```

`--clinical-csv`/`--image-root`를 생략하면 `dataset/` 안의 toy 데이터를 그대로 쓴다. 체크포인트는
`.\checkpoints\concat_v1\`에 저장된다(`--checkpoint-dir`로 위치 변경 가능).

`--fusion-strategy gated`만 바꾸면 게이트 기반 fusion으로 그대로 재실행된다. 두 경로 모두
같은 인자들을 쓴다:

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--clinical-csv` | `dataset/model_input_variables_full.csv` | raw clinical CSV 경로 |
| `--image-root` | `dataset/toy_image_dataset` | 이미지 루트 디렉토리 |
| `--resolution` | (필수) | DINOv3 입력 해상도. `patch_size`(16)의 배수면 임의 값 가능 |
| `--weights-path` | `assets/dinov3-hf` | 로컬 DINOv3 HF 스냅샷 경로 |
| `--embedding-cache` | 없음 | embedding parquet 캐시 경로 (지정하면 재실행 시 재추출 안 함) |
| `--fusion-strategy` | `concat` | `concat` \| `gated` \| `gbdt` |
| `--run-name` | timestamp | 체크포인트 하위 폴더명 |
| `--checkpoint-dir` | `checkpoints` | 체크포인트 루트 디렉토리 |
| `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--early-stopping-patience` | - | 학습 하이퍼파라미터 |
| `--val-ratio`, `--test-ratio`, `--seed` | `0.2` / `0.0` / `42` | 환자 단위 split 설정 |
| `--device` | `cpu` | `cpu` \| `cuda` |

### gbdt (완전 별도 파이프라인)

```powershell
cd D:\dfu\multimodal\dfu_multimodal
..\venv\Scripts\python.exe -m train_fusion `
  --resolution 224 `
  --fusion-strategy gbdt --gbdt-backend catboost `
  --run-name gbdt_catboost_v1 `
  --val-ratio 0.2 --seed 42
```

GBDT는 신경망 학습 루프를 타지 않는다 — clinical raw feature + image embedding을 하나의
tabular 행으로 합쳐서, 태스크(Wagner/SINBAD 6개)마다 완전히 독립적인 모델을 학습한다.
`--epochs`, `--lr` 같은 neural 전용 인자는 무시되고 대신 아래 인자를 쓴다.

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--gbdt-backend` | `catboost` | `catboost` \| `xgboost` \| `lightgbm` |
| `--gbdt-iterations` | `300` | 트리 개수 |
| `--gbdt-depth` | `4` | 트리 깊이 |
| `--gbdt-learning-rate` | `0.05` | 학습률 |
| `--gbdt-min-samples` | `10` | 태스크별 최소 라벨 표본 수 (미만이면 스킵) |
| `--gbdt-cat-features` | `age_band` | categorical로 취급할 clinical feature 컬럼 목록 |
| `--gbdt-min-child-samples` | `5` | LightGBM 전용(`min_data_in_leaf`), 소규모 데이터 대응 |

### 세 전략 결과 비교하기

같은 `--seed`, `--val-ratio`로 세 번 실행하면 완전히 같은 환자 집합으로 나뉘므로
`summary.json`을 그대로 나란히 비교할 수 있다.

```powershell
cd D:\dfu\multimodal\dfu_multimodal
$python = "..\venv\Scripts\python.exe"
$common = "--resolution 224 --seed 42 --val-ratio 0.2"

Invoke-Expression "$python -m train_fusion $common --fusion-strategy concat --run-name concat_v1"
Invoke-Expression "$python -m train_fusion $common --fusion-strategy gated  --run-name gated_v1"
Invoke-Expression "$python -m train_fusion $common --fusion-strategy gbdt   --run-name gbdt_v1"
```

## 출력물

`{checkpoint-dir}/{run-name}/` 아래:

- **neural(concat/gated)**: `best.pt`(최저 val loss 시점), `last.pt`(마지막 epoch), `summary.json`
- **gbdt**: `{task_name}_{backend}.pkl` (태스크별 모델, 7개), `summary.json`

`summary.json` 스키마는 두 경로가 최대한 동일하게 맞춰져 있다:

```jsonc
{
  "fusion_strategy": "concat",       // "concat" | "gated" | "gbdt"
  "n_total": 164,                    // 전체 표본 수 (라벨 유무 무관)
  "n_train": 125,                    // 환자 단위 split 기준 train 표본 수
  "n_val": 39,                       // val 표본 수
  "best_val_loss": 7.09,             // neural 전용 (gbdt에는 없음 — 태스크별 독립 모델이라
                                      // 하나의 loss로 합산할 대상이 없음)
  "tasks": {
    "wagner": {
      "n_val_labeled": 36,           // 이 태스크 라벨이 있는 val 표본 수
      "accuracy": 0.19,
      "balanced_accuracy": 0.2,
      "macro_f1": 0.065,
      "roc_auc": null                // multiclass라 항상 null
    },
    "sinbad_site": { "...": "..." },
    "...": "..."
  }
}
```

체크포인트 로딩(`dfu_multimodal/` 안에서 실행하는 스크립트/REPL 기준):

```python
from trainers.neural_trainer import load_checkpoint

model, checkpoint = load_checkpoint("checkpoints/concat_v1/best.pt", device="cpu")
```

## 새 fusion 전략 추가하기

cross-attention, FiLM 같은 새 신경망 fusion 방식을 추가하려면:

1. `models/fusion_strategies/`에 `FusionStrategy`(`base.py`)를 상속한 클래스를 하나 추가한다.
   `forward(image_repr, clinical_repr) -> shared_repr` 규격만 지키면 된다.
2. `models/fusion_strategies/__init__.py`의 `FUSION_STRATEGIES` 딕셔너리에 한 줄 등록한다.
3. `cli/fusion_args.py`의 `--fusion-strategy` choices에 이름을 추가한다.

`fusion_model.py`, `neural_trainer.py`는 전혀 수정할 필요가 없다. (GBDT처럼 미분 불가능하고
shared trunk 구조 자체가 성립하지 않는 방식이라면 `gbdt/`처럼 별도 경로를 새로 만드는 쪽이
맞다 — 자세한 배경은 [Project.md](./Project.md) 4.3절 참고.)

## 결측치 처리 정책

- **이미지 없음**: `img2d`가 비어있거나(`NaN`) 문자열 `"none"`/`"null"`/`"nan"`이거나
  실제로 그 경로에 파일이 없으면, 그 행은 이미지 매칭 단계에서 제외된다.
- **라벨 결측은 태스크별로 독립 처리된다**: Wagner 라벨이 없는 행은 Wagner 학습/평가에서만
  빠지고, 다른 SINBAD component 학습에는 영향을 주지 않는다 (neural 경로는 mask 기반, gbdt
  경로는 태스크마다 `notna()` 필터링).
- **clinical feature 결측**(neural 경로만 해당): continuous는 train split의 median으로,
  binary는 0.5로 채우고, 원래 결측 여부를 나타내는 `_missing_flag` 컬럼을 별도로 추가한다.
  GBDT 경로는 이 전처리를 거치지 않고 raw 값을 그대로 넘긴다 — CatBoost/XGBoost/LightGBM은
  결측치를 자체적으로 처리한다.
