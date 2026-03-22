# 跑壓縮過的 task（4-bit 量化）
set -e
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 跑 baseline（fp16，不量化）
# 每個 task type 各抽一個代表：
#   Single-doc QA  → narrativeqa
#   Multi-doc QA   → hotpotqa
#   Summarization  → gov_report
#   Few-shot       → trec,samsum
#   Code           → lcc
# python deployment/longbench_eval.py --task qasper --bits 16
# python deployment/longbench_eval.py --task 2wikimqa --bits 16
# python deployment/longbench_eval.py --task multi_news --bits 16
# python deployment/longbench_eval.py --task samsum --bits 16
# python deployment/longbench_eval.py --task passage_count --bits 16
# python deployment/longbench_eval.py --task repobench-p --bits 16

# Single-doc QA
# python deployment/longbench_eval.py --task narrativeqa
# python deployment/longbench_eval.py --task qasper
# python deployment/longbench_eval.py --task multifieldqa_en
# # Multi-doc QA
# python deployment/longbench_eval.py --task hotpotqa
# python deployment/longbench_eval.py --task 2wikimqa
# python deployment/longbench_eval.py --task musique
# # Summarization
# python deployment/longbench_eval.py --task gov_report
# python deployment/longbench_eval.py --task qmsum
# python deployment/longbench_eval.py --task multi_news
# # Few-shot
# python deployment/longbench_eval.py --task trec
# python deployment/longbench_eval.py --task triviaqa
# python deployment/longbench_eval.py --task samsum
# # Synthetic
# python deployment/longbench_eval.py --task passage_count
# python deployment/longbench_eval.py --task passage_retrieval_en
# # Code
# python deployment/longbench_eval.py --task lcc
# python deployment/longbench_eval.py --task repobench-p

# Longbench_v2 (4-bit 量化)
python deployment/longbench_v2_eval.py --task single-doc
# python deployment/longbench_v2_eval.py --task multi-doc
# python deployment/longbench_v2_eval.py --task long-context
# python deployment/longbench_v2_eval.py --task dialogue
# python deployment/longbench_v2_eval.py --task code
# python deployment/longbench_v2_eval.py --task structured

# nohup bash ./run.sh > run.log 2>&1 &
echo "Done"