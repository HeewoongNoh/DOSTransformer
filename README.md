# krict_2021

### 데이터셋 빌드
데이터셋은 용량 문제로 업로드하지 못했습니다.

연구원님께서 공유해주셨던 [링크](https://drive.google.com/drive/folders/1zMstskysqr6pT-Fk5j2Apc6ORLr5z2ER?usp=sharing)에 있는 파일들을 다운받아주시고 `data\raw` 디렉토리에 넣어주시면 됩니다.

`create_pickle.py` 파일은 raw 파일들을 pkl 파일로 변환시켜주는 역할을 합니다.

`mat2graph.py` 파일은 연구원님께서 초반에 보내주셨던 변환 파일을 기반으로 위에서 만들어진 pickle 파일을 소재 구조 그래프 형태로 변환해주고 DoS 데이터와 연결시켜주는 역할을 합니다. 결과물은 processed 폴더에 저장이 됩니다.

`mat2graph_mp_only.py` 파일은 pretraining 에서 사용할 mp 데이터를 만들어주는 역할을 합니다.

### 모델 학습
모델은 최종 발표에서 말씀드렸듯이 node-level pretraining / graph-level pretraining / finetuning 순서로 학습을 시켜주시면 됩니다.

`pretrain_masking.py` 파일은 노드들을 마스킹하고 어떤 원자인지를 맞추는 태스크를 수행합니다.

`pretrain_efermi.py` 파일은 각각의 소재구조의 fermi energy 를 맞추는 태스크를 수행합니다.

`finetune.py` 파일은 pretrain 된 모델을 기반으로 본격적인 dos 를 맞추는 태스크를 수행합니다.

이와 별개로 다른 태스크들을 실험하실때 사용하실 수 있도록 `train_joint.py` 파일을 만들어 놓았습니다. 이 파일은 pretrain 없이 그냥 multi-task learning 만 적용을 한 모델이라고 생각하시면 됩니다.  

### Hyperparameters
`--layers:` GNTransformer 모델에서 GNN layer의 개수 입니다.

`--transformer:` GNTransformer 모델에서 Transformer layer의 개수 입니다.

`--masking_ratio:` GNTransformer 모델에서 node-level pretraining을 할 때 노드를 얼마나 masking 할지 결정하는 하이퍼파라미터 입니다.

`--spark:` GNTransformer 모델에서 joint training 할 때 두 연속된 에너지 레벨 사이에서 "튄다"라는 개념을 정의합니다. 예를 들어 spark = 0.1 일 경우 두 연속된 에너지 레벨의 dos 차이가 0.1 이상일 경우 "튄다"라고 학습을 하게 됩니다. `mategraph.py` 파일에서 spark 값은 0.01, 0.1, 0.2, 0.3, 0.4, 0.5 를 데이터셋에 저장을 해놓았습니다.

`--preepoch:` GNTransformer 모델에서 finetune 을 할 때 몇 epoch 를 돌린 pretrained model 을 가져올지 결정하는 역할을 합니다.# DOSTransformer
