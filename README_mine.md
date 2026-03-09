
## Step 0：建立環境

```bash
cd /root/KVQuant
python3.11 -m venv venv
source venv/bin/activate

# 安裝基本套件
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install wheel datasets

# 安裝 quant 套件
cd quant
pip install -e .
cd ..

# 安裝 deployment 用的 custom transformers（支援 KV cache 量化）
pip install -e deployment/transformers/

# 編譯 quant_cuda CUDA extension（deployment 推理用）
cd deployment/kvquant
python setup_cuda.py install
cd ../..

# NOTE: flash-attn 與 torch ABI 不相容，改用 sdpa，不需安裝 flash-attn
```

 # Huggingface
 huggingface-cli login
 hf_MsunjexXeNDaolHSNtbppwsykJCmytScVcab

 # Copy Dataset
 scp -P 40655 -r -i ~/.ssh/id_ed25519_evelyn_r76134115 ./longbench_v1/ root@38.147.83.26:/root/KVQuant/data/
 scp -P 40655 -r -i ~/.ssh/id_ed25519_evelyn_r76134115 ./longbench_v1/musique.jsonl root@194.68.245.81:/root/KVQuant/data/longbench_v1/musique.jsonl
 ssh root@38.147.83.26 -p 40655 -i ~/.ssh/id_ed25519

 # 拿掉 GitHub 最新 commit，但保留內容在 local
 git reset --soft HEAD~1
 git push --force-with-lease

 # Setup name and gmail
 git config user.name Evelyn
 git config user.email chia20010709@gmail.com

 # Download Claude Code Extension

## Step 1：計算 Fisher Information（可選，但效果更好）
cd ..
cd gradients
CUDA_VISIBLE_DEVICES=0 python run-fisher.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --dataset c4 \
  --num_examples 16 \
  --seqlen 2048 \
  --output_dir /tmp/fisher_output

## Step 2：量化校準（產生 quantizers.pickle）
cd ..
cd quant
CUDA_VISIBLE_DEVICES=0 python llama_simquant.py meta-llama/Meta-Llama-3.1-8B-Instruct \
  --abits 4 \
  --nsamples 16 \
  --seqlen 2048 \
  --nuq \
  --fisher /tmp/fisher_output \
  --quantize \
  --include_sparse \
  --sparsity-threshold 0.99 \
  --first_few_fp16 5 \
  --quantizer-path quantizers.pickle

## Step 3：驗證（跑 Perplexity 確認量化正確）
CUDA_VISIBLE_DEVICES=0 python llama_simquant.py meta-llama/Meta-Llama-3.1-8B-Instruct \
  --abits 4 \
  --nuq \
  --include_sparse \
  --sparsity-threshold 0.99 \
  --first_few_fp16 5 \
  --quantizer-path quantizers.pickle \
  --seqlen 2048

** 7.多是正常的

## Step 4：LongBench Evaluation
cd ../deployment

# 確認 quantizers.pickle 在這個資料夾（從 quant/ 複製過來）
cp ../quant/quantizers.pickle .

# 跑單一 task（4-bit 量化）
CUDA_VISIBLE_DEVICES=0 python longbench_eval.py \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --task narrativeqa \
  --bits 4 \
  --quantizer-path quantizers.pickle \
  --include_sparse \
  --sparsity-threshold 0.99 \
  --first_few_fp16 5 \
  --output-len 64 \
  --chunk-size 512 \
  --n-warmup 2 \
  --output-path results/narrativeqa_4bit.json

# 跑 baseline（fp16，不量化）
CUDA_VISIBLE_DEVICES=0 python longbench_eval.py \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --task narrativeqa \
  --bits 16 \
  --output-len 64 \
  --chunk-size 512 \
  --n-warmup 2 \
  --output-path results/narrativeqa_fp16.json

# 支援的 task：
# narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique,
# gov_report, qmsum, multi_news, trec, triviaqa, samsum,
# passage_count, passage_retrieval_en, lcc, repobench-p

# 結果存在 deployment/results/<task>.json（格式同 longbench_sample_output.json）

