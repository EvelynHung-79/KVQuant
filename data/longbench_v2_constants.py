# LongBench v2 – domain map, task list, metrics, output lengths

V2_DOMAIN_MAP = {
    "single-doc": "Single-Document QA",
    "multi-doc": "Multi-Document QA",
    "long-context": "Long In-context Learning",
    "dialogue": "Long-dialogue History Understanding",
    "code": "Code Repository Understanding",
    "structured": "Long Structured Data Understanding"
}

# All v2 task names (jsonl filenames without extension)
V2_TASKS = [
    "single-doc",
    "multi-doc",
    "long-context",
    "dialogue",
    "code",
    "structured",
]

# v2 is all MCQ → accuracy metric, short output
TASK_METRICS_V2 = {task: "accuracy" for task in V2_TASKS}
TASK_OUTPUT_LEN_V2 = {task: 4 for task in V2_TASKS}
