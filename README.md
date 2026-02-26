# LLMPostTraining

## Description
LLMPostTraining is a framework for post-training and fine-tuning Large Language Models (LLMs): Supervised Fine-Tuning (SFT), Reinforcement Learning from Human Feedback (RLHF), distillation, quantization, and MoE/MLM training. It uses DeepSpeed ZeRO, multi-GPU parallelism, and shared config/training utilities.

## Installation

### Prerequisites
- Python 3.12+
- CUDA-capable GPU (recommended)
- uv or pip for dependencies

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd llm_training
```

2. Install uv (optional):
```bash
pip install uv
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt --no-build-isolation --index-strategy unsafe-best-match
```
or with pip:
```bash
pip install -r requirements.txt
```

4. Run from repo root with `PYTHONPATH` set:
```bash
export PYTHONPATH=/path/to/llm_training   # or . when already in repo root
```

**GCP VM with local SSD (optional):**
```sh
sudo lsblk -o NAME,SIZE,TYPE,MOUNTPOINT | grep nvme0n1
sudo mkfs.ext4 -F /dev/nvme0n1
sudo mkdir -p /mnt/disks/local-ssd
sudo mount /dev/nvme0n1 /mnt/disks/local-ssd
sudo chmod a+w /mnt/disks/local-ssd
UUID=$(sudo blkid -s UUID -o value /dev/nvme0n1)
echo "UUID=$UUID /mnt/disks/local-ssd ext4 discard,defaults,nofail 0 2" | sudo tee -a /etc/fstab
```

## Usage

### Model Inference
```bash
python main.py
```

### Seqorth SFT
From repo root:
```bash
PYTHONPATH=. bash training/seqorth_sft/run_seqorth.sh [config] [num_gpus]
```
Configs live under `config/seqorth/` (e.g. `config/seqorth/seqorth_qwen_config.json`).

### Supervised Fine-Tuning (SFT / G3MoE)
```bash
cd training/sft
# Configs: config/sft/*.json
python custom_module_sft.py   # or use run_g3moe_config.sh with config/sft/...
```

### Lightning Trainer
```bash
cd training/lightning_trainer
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export WANDB_API_KEY=<your_wandb_api_key>
export HF_SECRET_KEY=<your_huggingface_token>
export HF_DATASETS_CACHE=<your_cache_directory>

huggingface-cli login --token $HF_SECRET_KEY
wandb login --relogin $WANDB_API_KEY

python trainer.py fit \
    --trainer.fast_dev_run false \
    --trainer.max_epochs 5 \
    --model.learning_rate 3e-3 \
    --data.train_batch_size 4 \
    --data.eval_batch_size 4
```

### RLHF Training
```bash
cd training/rlhf
export CUDA_VISIBLE_DEVICES="0,1"
export WANDB_API_KEY=<your_wandb_api_key>
export HF_SECRET_KEY=<your_huggingface_token>
export HF_DATASETS_CACHE=<your_cache_directory>

huggingface-cli login --token $HF_SECRET_KEY
wandb login --relogin $WANDB_API_KEY

accelerate launch --config_file "accelerate_config.yaml" train.py
```

### MoRA
Install the MoRA (peft-mora) package from the repo, then run training:
```bash
pip install -e ./models/mora
cd training/mora
# Use train.py with your config (see training/mora/README.md)
```

## Features
- **Training methods**: SFT (Seqorth, G3MoE), RLHF (GRPO, TTC, DPO/SimPO), MoRA, distillation, Lightning
- **Models**: Seqorth MoE, G3MoE, MoRA, Qwen3 MoE fused; shared code under `core/` and `models/`
- **Config**: Single `config/` tree for seqorth, sft, rlhf, eval
- **Distributed**: DeepSpeed ZeRO-2/3, multi-GPU, NVMe offload
- **Monitoring**: W&B, eval callbacks, routing benchmarks

## Memory & ZeRO

### ZeRO (ZeRO-2/3 + optional ZenFlow)
- **ZeRO-2 + ZenFlow**: Recommended for performance
- **ZeRO-3**: Max memory efficiency; parameter/gradient/optimizer partitioning, NVMe offload
- **ZenFlow**: Async gradient updates, selective updates, communication overlap (configurable in DeepSpeed config)

### Environment
```bash
# Disable ZenFlow if RAM OOM
export DISABLE_ZENFLOW=1
```

### Tips
- Use ZeRO-2 + ZenFlow for best throughput; ZeRO-3 for largest models
- Reduce `per_device_train_batch_size` or enable `gradient_checkpointing` if OOM
- Configs under `config/seqorth/`, `config/sft/` reference DeepSpeed JSONs in the same tree

## Dependencies
- PyTorch 2.4+ (CUDA)
- Transformers (with trust_remote_code for Qwen/Seqorth)
- DeepSpeed, Accelerate
- Lightning AI (for `training/lightning_trainer`)
- Hugging Face (datasets, tokenizers, etc.)

## License
MIT License.
