# DPO2-E 재현용 공통 경로 (실습실 노드, 모든 것을 /shared_dir 아래에 둔다)
# 원본 스크립트의 /home/latuca/... 경로를 이 값으로 바꿔 쓴다.
ROOT=/shared_dir/MMDL_solo
EVAL=/shared_dir/mmdl_test                      # mmdl_test @ 7740786a1b6b173db86285a00f06e93a44d8a225
TRAIN_PY=/shared_dir/venv/bin/python            # 학습 venv (torch 2.8.0, transformers 4.57.0, peft 0.17.1)
EVAL_PY=$EVAL/.venv/bin/python                  # 평가·샘플링 venv (mmdl_test/requirements.txt 고정 버전)

export HF_HOME=/shared_dir/hf_cache
export MMMU_MODEL_CACHE=$HF_HOME/hub            # mmdl_test prepare/evaluate가 ~/.cache 대신 이 경로를 쓰게 한다
export MMMU_DATASET_CACHE=$HF_HOME/datasets
export PIP_CACHE_DIR=/shared_dir/.pip_cache
export VLLM_CACHE_ROOT=/shared_dir/.vllm_cache     # vLLM 컴파일 캐시도 /shared_dir 아래 (다음 실행부터 적용)

BASE_REPO=Qwen/Qwen3-VL-4B-Instruct
BASE_REV=ebb281ec70b05090aa6165b016eac8ec08e71b17
BASE=$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/$BASE_REV

SFT_DATA=$ROOT/runs/sft_general_vic2000_e5/data/train.jsonl
OLD=$ROOT/runs/dpo_ep4_mix_b0.1_lr5e-6/data     # 베이스 샘플·채점 (원래 위치 유지)
IMAGES=$ROOT/data/vic_era_2000
VCFG=$ROOT/config/training_settings_only.json
RUN=$ROOT/runs/dpo2
D=$RUN/data
FR=/shared_dir/friend_dpo_r4/x                  # 팀원 루프 쌍 (datasets/dpo_r4_loop/train.jsonl + 이미지)
EXP_E=$ROOT/experiments/dpo2_E                  # 원래 결과 (비교 기준)
