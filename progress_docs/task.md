重要提醒: 在做事的时候, 记得做笔记, 实习期结束前, 需要将笔记整合成报告的形式作为最后的交付.

任务1
了解模型评估方式. 学会常用的模型评估框架
需要学习的模型评估框架: opencompass
1. GitHub - open-compass/opencompass: OpenCompass is an LLM evaluation platform, supporting a wide rang

任务2
了解主流的医学评估benchmark.
需要学习的医学评估benchmark: healthbench
1. HealthBench: Evaluating Large Language Models Towards Improved Human Health
2. Introducing HealthBench

任务3(最重要)
使用opencompass测评模型Qwen3-4B-Thinking-2507在healthbench的效果.
参考opencompass的教程完成该测评, 你可以在下面任务中2选1, 或者都做:
1. 加速这个测评, 该测评如果不做改动, 实际耗时高达4天. 你需要了解opencompass的代码, 提高其在infer和eval的并发或调用速度, 你可以制作较小的healthbench数据集来测试.
2. 参考healthbench的论文, 从零编写测评healthbench代码. 从而可控的加速infer和eval.
opencompass参考代码(该代码使用模型API进行测评, 你可以使用任何喜欢的方式改写):
from opencompass.models import OpenAISDK
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

num_worker = 100
work_dir = "/root/hospital/temp/opencompass"

data_dir = "/mnt/datasets/opencompass"

models = [
    dict(
        abbr="qwen3-80b",
        type=OpenAISDK,
        
        path="/mnt/temp/qwen3-80b",
        tokenizer_path="/mnt/temp/qwen3-80b",
        key="YOUR_API_KEY",
        openai_api_base="http://YOUR_API_BASE/v1",
        
        meta_template=dict(
            round=[
                dict(role='HUMAN', api_role='HUMAN'),
                dict(role='BOT', api_role='BOT', generate=True),
            ],
        ),
        
        max_seq_len=16384,
        max_out_len=16384,
        batch_size=480,
        temperature=0.001,
        max_workers=num_worker,
        
        query_per_second=num_worker,
        verbose=True,
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    ),
]

with read_base():
    from opencompass.configs.datasets.HealthBench.healthbench_gen_831613 import healthbench_datasets

for item in healthbench_datasets:
    item["path"] = data_dir + "/" + item["path"]
for item in healthbench_datasets:
    item["eval_cfg"]["evaluator"]["n_threads"] = num_worker

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

verifier_cfg = dict(
    abbr="qwen3-235b-a22b-thinking-2507",
    type=OpenAISDK,
    
    path="qwen3-235b-a22b-thinking-2507",
    tokenizer_path="/mnt/models/Qwen3-235B-A22B-Thinking-2507",
    key="YOUR_API_KEY",
    openai_api_base="http://YOUR_API_BASE/v1",
    
    meta_template=dict(
        round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ],
    ),
    
    max_seq_len=65536,
    max_out_len=65536,
    batch_size=480,
    temperature=0.001,
    max_workers=num_worker,
    
    query_per_second=num_worker,
    verbose=True,
    pred_postprocessor=dict(type=extract_non_reasoning_content)
)

for item in datasets:
    if 'judge_cfg' in item['eval_cfg']['evaluator']:
        item['eval_cfg']['evaluator']['judge_cfg'] = verifier_cfg
    if 'llm_evaluator' in item['eval_cfg']['evaluator'].keys() and 'judge_cfg' in item['eval_cfg']['evaluator']['llm_evaluator']:
        item['eval_cfg']['evaluator']['llm_evaluator']['judge_cfg'] = verifier_cfg
    item["eval_cfg"]["pred_postprocessor"] = dict(type=extract_non_reasoning_content)
    if "infer_cfg" in item and "inferencer" in item["infer_cfg"] and "max_out_len" in item["infer_cfg"]["inferencer"]:
        item["infer_cfg"]["inferencer"]["max_out_len"] = 16384
    

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner, 
        num_worker=num_worker
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=num_worker,
        retry=1,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(
        type=NaivePartitioner, 
        n=10
    ),
    runner=dict(type=LocalRunner,
        max_num_workers=num_worker,
        retry=1,
        task=dict(type=OpenICLEvalTask)
    ),
)

summarizer = dict(
    dataset_abbrs=[
        ["healthbench", "accuracy"],
        ["healthbench_hard", "accuracy"],
        ["healthbench_consensus", "accuracy"],
        ["healthbench_all", "naive_average"],
    ],
    summary_groups=[
        {"name": "healthbench_all", "subsets": ["healthbench", "healthbench_hard", "healthbench_consensus"]}
    ],
)
opencompass参考启动命令:
OC_JUDGE_MODEL=qwen3-235b-a22b-thinking-2507 OC_JUDGE_API_BASE=http://YOUR_API_BASE/v1 OC_JUDGE_API_KEY=YOUR_API_KEY opencompass opencompass_eval.py

附录
博大A40服务器

IP
YOUR_SERVER_IP
Port
YOUR_PORT
Username
YOUR_USERNAME
Password
YOUR_PASSWORD
Model Path
/data1/models/Qwen3-4B-Thinking-2507
Judge Model
qwen3-235b-a22b-thinking-2507 
Judge Model API
http://YOUR_API_BASE/v1
Judge Model API Key
YOUR_API_KEY
