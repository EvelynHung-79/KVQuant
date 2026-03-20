
## Step 0：建立環境（Step 1–4 統一用同一個 venv）

```bash
cd /root/KVQuant
python3.11 -m venv venv
source venv/bin/activate

# 基本工具
pip install --upgrade pip
pip install wheel ninja

# torch（必須先裝，quant_cuda 編譯會用到）
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 基本套件
pip install datasets huggingface_hub==0.36.0

# quant 套件（Step 1–3 calibration 用）
pip install -e quant/

# custom transformers（Step 4 inference 用，必須在 quant/ 之後裝才能覆蓋）
pip install -e deployment/transformers/

# deployment 主套件
pip install -e deployment/ --no-build-isolation

# 評分套件（ROUGE-L 和 edit_sim 用）
pip install rouge fuzzywuzzy python-Levenshtein

# 編譯 CUDA extension（需要 nvcc）
cd deployment/kvquant && python setup_cuda.py install && cd ../..

# 讓 quant_cuda 每次都能找到 torch shared libraries
echo 'export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"' >> venv/bin/activate

# 重新 activate 讓 LD_LIBRARY_PATH 生效
source venv/bin/activate
```
 _EWXDELkcgPsWjrlEIpUInlBMYXDFDZbnba

 # Copy Dataset
 scp -r ./longbench_v1/ pod2:/root/KVQuant/data/

 # 拿掉 GitHub 最新 commit，但保留內容在 local
 git reset --soft HEAD~1
 git push --force-with-lease

 # Setup name and gmail
 git config user.name Evelyn
 git config user.email chia20010709@gmail.com

 # Download Claude Code Extension

## Step 1：計算 Fisher Information（可選，但效果更好）
```bash
# 安裝 quant 套件 for Step 1 to 3
cd quant
pip install -e .

cd ../gradients
CUDA_VISIBLE_DEVICES=0 python run-fisher.py \
  --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset c4 \
  --num_examples 16 \
  --seqlen 2048 \
  --output_dir /tmp/fisher_output
```

## Step 2：量化校準（產生 quantizers.pickle）
```bash
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
```

## Step 3：驗證（跑 Perplexity 確認量化正確）
```bash
CUDA_VISIBLE_DEVICES=0 python llama_simquant.py meta-llama/Meta-Llama-3.1-8B-Instruct \
  --abits 4 \
  --nuq \
  --include_sparse \
  --sparsity-threshold 0.99 \
  --first_few_fp16 5 \
  --quantizer-path quantizers.pickle \
  --seqlen 2048

# ** 7.多是正常的
```

## Step 4：LongBench Evaluation

```bash
# 確認在 venv 裡
source ~/KVQuant/venv/bin/activate

# 確認 quantizers.pickle 路徑（run.sh 裡用絕對路徑指定）
# 預設位置：~/KVQuant/quant/quantizers_llama3.pickle

nohup bash ~/KVQuant/run.sh > ~/KVQuant/run.log 2>&1 &
tail -f ~/KVQuant/run.log
```

# 支援的 task：
# narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique,
# gov_report, qmsum, multi_news, trec, triviaqa, samsum,
# passage_count, passage_retrieval_en, lcc, repobench-p

# 結果存在 deployment/results/<task>.json（格式同 longbench_sample_output.json）

