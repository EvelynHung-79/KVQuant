
## Step 0：建立環境
python3.10 -m venv venv
source venv/bin/activate

cd quant
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .
pip install flash-attn --no-build-isolation
pip install datasets

## Step 1：計算 Fisher Information（可選，但效果更好）
cd gradients
python run-fisher.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --dataset c4 \
  --num_examples 16 \
  --seqlen 2048 \
  --output_dir /tmp/fisher_output

## Step 2：量化校準（產生 quantizers.pickle）
cd quant
CUDA_VISIBLE_DEVICES=0 python llama_simquant.py meta-llama/Llama-3.1-8B-Instruct \
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
CUDA_VISIBLE_DEVICES=0 python llama_simquant.py meta-llama/Llama-3.1-8B-Instruct \
  --abits 4 \
  --nuq \
  --include_sparse \
  --sparsity-threshold 0.99 \
  --first_few_fp16 5 \
  --quantizer-path quantizers.pickle

## Step 4：LongBench Evaluation（需要新增）
# 安裝 LongBench
pip install rouge_score jieba fuzz


