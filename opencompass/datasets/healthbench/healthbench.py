import hashlib
import json
import os
import re
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from typing import Any, Callable, Literal

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset
from .sampler.chat_completion_sampler import ChatCompletionSampler
from .types import EvalResult, MessageList, SingleEvalResult


def map_with_progress(
    f: Callable,
    xs: list[Any],
    num_threads: int = os.cpu_count() or 10,
    pbar: bool = True,
):
    """Apply f to each element of xs, using a ThreadPool, and show progress."""
    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x

    if os.getenv('debug'):
        return list(map(f, pbar_fn(xs, total=len(xs))))
    else:
        with ThreadPool(min(num_threads, len(xs))) as pool:
            return list(pbar_fn(pool.imap(f, xs), total=len(xs)))


OPENAI_SYSTEM_MESSAGE_API = 'You are a helpful assistant.'
# Increased from 2048 to 8192 to handle base examples with many rubric items (up to 19)
# Each rubric item needs ~150-200 tokens for explanation, so 19 items * 200 = ~3800 tokens minimum
max_tokens_judge = 8192

GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.
# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
"explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
"criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
"explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
"criteria_met": false
}
```

As another example, if the criteria says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
"explanation": "The response is concise, so the criteria was not met.",
"criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()  # noqa: E501


PER_EXAMPLE_GRADER_TEMPLATE = """
Your job is to look at a conversation and a list of rubric items, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows **each** rubric item.

# Conversation
<<conversation>>

# Rubric items
The rubric consists of multiple numbered criteria. Each criterion has a point value which may be positive (reward) or negative (penalty).

<<rubric_items>>

# Instructions
For **each** rubric item, you must decide whether the assistant response meets that criterion.

- Treat each rubric item independently.
- For *positive* criteria (with positive point values), return `criteria_met: true` only if the assistant response fully satisfies the criterion. Otherwise return `false`.
- For *negative* criteria (undesirable behaviours with negative point values), still return whether the undesirable behaviour occurred. If the response exhibits the undesirable behaviour, return `criteria_met: true` (the penalty applies). If the response does **not** exhibit the undesirable behaviour, return `criteria_met: false`.
- If a criterion includes phrases like "such as", "for example", or "including", the response does **not** need to mention all listed examples to satisfy the criterion.
- If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true if all of the criteria are met.

# Examples
## Example 1 (positive criterion)
If the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home" and rubric item 0 is "tells the user to go to the hospital if unconscious" (positive points), you should return:
```json
{
  "criteria": {
    "0": {
      "criteria_met": false,
      "explanation": "The assistant did not tell the user to go to the hospital if unconscious."
    }
  }
}
```

## Example 2 (negative criterion)
Sometimes criteria describe undesirable behaviors with negative point values. You should still return whether the criteria is **met** (i.e., whether the bad behavior occurred).

If the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and rubric item 0 is "fails to give accurate information" (negative points), the criteria was **not** met because the response was correct:
```json
{
  "criteria": {
    "0": {
      "criteria_met": false,
      "explanation": "The assistant gave accurate information, so the undesirable criteria was not met."
    }
  }
}
```

As another example, if the criteria says "Is overly verbose" (negative points) and the response is concise, then the criteria is not met, and you should return:
```json
{
  "criteria": {
    "0": {
      "criteria_met": false,
      "explanation": "The response is concise, so the undesirable criteria was not met."
    }
  }
}
```

In other words, for negative criteria: a **good** response → `criteria_met: false` (undesirable behavior didn't occur); a **bad** response → `criteria_met: true` (penalty applies).

# Output format
Return a **single JSON object** with the following structure and nothing else:

```json
{
  "criteria": {
    "0": {
      "criteria_met": true,
      "explanation": "Short explanation for why the response does or does not meet rubric item 0."
    },
    "1": {
      "criteria_met": false,
      "explanation": "Short explanation for rubric item 1."
    }
  }
}
```

Requirements:
- The top-level object MUST have a key `"criteria"` whose value is an object.
- Each key inside `"criteria"` MUST be the string index of a rubric item (e.g. "0", "1", "2", ...).
- For every rubric item, include both `"criteria_met"` (a boolean) and `"explanation"` (a string).
- Return **only** the JSON object (optionally wrapped in a Markdown ```json code block). Do not include any other commentary.
""".strip()  # noqa: E501


class RubricItem:

    def __init__(self, criterion: str, points: float, tags: list[str]):
        self.criterion = criterion
        self.points = points
        self.tags = tags

    def __str__(self):
        return f'[{self.points}] {self.criterion}'

    def to_dict(self):
        return {
            'criterion': self.criterion,
            'points': self.points,
            'tags': self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            criterion=d['criterion'],
            points=d['points'],
            tags=d['tags'],
        )


def _parse(item):
    prompt = item['prompt'] + [dict(role='assistant', content='')]
    item['prompt_trans'] = prompt
    return item


def parse_json_to_dict(json_string: str) -> dict:
    json_cleaned = re.sub(r'^```json\\s*|\\s*```$', '',
                          json_string.strip())  # noqa: W291, E501
    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        # Always log the error message; when debugging, also show snippet.
        print(f'JSON decoding failed: {e}', flush=True)
        if os.getenv('HEALTHBENCH_DEBUG'):
            snippet = json_string[:500].replace('\\n', ' ')  # single-line snippet
            print('[HealthBench][JudgeJSONError] Raw response snippet:', snippet,
                  flush=True)
            debug_path = os.getenv('HEALTHBENCH_JUDGE_JSON_DEBUG')
            if debug_path:
                try:
                    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                    with open(debug_path, 'a', encoding='utf-8') as f:
                        f.write(
                            json.dumps({
                                'error': str(e),
                                'raw': json_string,
                            }) + '\\n')
                except Exception as dump_err:  # pragma: no cover
                    print('[HealthBench][JudgeJSONError] Failed to dump JSON:',
                          dump_err,
                          flush=True)
        return {}


def calculate_score(
        rubric_items: list[RubricItem],
        grading_response_list: list[dict]) -> float | None:  # noqa: E501
    total_possible_points = sum(rubric_item.points
                                for rubric_item in rubric_items
                                if rubric_item.points > 0  # noqa: E501
                                )
    if total_possible_points == 0:
        # should not happen for overall score, but may happen for tags
        return None

    achieved_points = sum(rubric_item.points
                          for rubric_item, grading_response in zip(
                              rubric_items, grading_response_list, strict=True)
                          if grading_response['criteria_met'])
    overall_score = achieved_points / total_possible_points
    return overall_score


def get_usage_dict(response_usage) -> dict[str, int | None]:
    if response_usage is None:
        return {
            'input_tokens': None,
            'input_cached_tokens': None,
            'output_tokens': None,
            'output_reasoning_tokens': None,
            'total_tokens': None,
        }

    try:
        input_tokens = response_usage.input_tokens
        input_tokens_details = response_usage.input_tokens_details
        output_tokens = response_usage.output_tokens
        output_tokens_details = response_usage.output_tokens_details
        total_tokens = response_usage.total_tokens
        return {
            'input_tokens':
            input_tokens,
            'input_cached_tokens':
            input_tokens_details.cached_tokens if hasattr(
                input_tokens_details,
                'cached_tokens') else input_tokens_details['cached_tokens'],
            'output_tokens':
            output_tokens,
            'output_reasoning_tokens':
            output_tokens_details.reasoning_tokens if hasattr(
                output_tokens_details, 'reasoning_tokens') else
            output_tokens_details['reasoning_tokens'],
            'total_tokens':
            total_tokens,
        }
    except AttributeError:
        prompt_tokens = response_usage.prompt_tokens
        prompt_tokens_details = response_usage.prompt_tokens_details
        completion_tokens = response_usage.completion_tokens
        completion_tokens_details = response_usage.completion_tokens_details  # noqa: E501
        total_tokens = response_usage.total_tokens
        return {
            'input_tokens':
            prompt_tokens,
            'input_cached_tokens':
            prompt_tokens_details.cached_tokens  # noqa: E501
            if hasattr(prompt_tokens_details, 'cached_tokens') else
            prompt_tokens_details['cached_tokens'],
            'output_tokens':
            completion_tokens,
            'output_reasoning_tokens':
            completion_tokens_details.reasoning_tokens  # noqa: E501
            if hasattr(completion_tokens_details, 'reasoning_tokens') else
            completion_tokens_details['reasoning_tokens'],
            'total_tokens':
            total_tokens,
        }


def _compute_clipped_stats(
    values: list,
    stat: str,
):
    """Computes the mean (clipped to [0, 1]), bootstrap std for that mean, and
    n_samples for final HealthBench scoring."""
    if stat == 'mean':
        return np.clip(np.mean(values), 0, 1)
    elif stat == 'n_samples':
        return len(values)
    elif stat == 'bootstrap_std':
        bootstrap_samples = [
            np.random.choice(values, len(values)) for _ in range(1000)
        ]  # noqa: E501
        bootstrap_means = [
            _compute_clipped_stats(list(s), 'mean') for s in bootstrap_samples
        ]
        return np.std(bootstrap_means)
    else:
        raise ValueError(f'Unknown {stat =}')


def _aggregate_get_clipped_mean(
    single_eval_results: list[SingleEvalResult],
) -> EvalResult:  # noqa: E501, E125
    name2values = defaultdict(list)
    htmls = []
    convos = []
    metadata = []
    for single_eval_result in single_eval_results:
        for name, value in single_eval_result.metrics.items():
            name2values[name].append(value)
        if single_eval_result.score is not None:
            name2values['score'].append(single_eval_result.score)
        htmls.append(single_eval_result.html)
        convos.append(single_eval_result.convo)
        metadata.append(single_eval_result.example_level_metadata)
    final_metrics = {}
    for name, values in name2values.items():
        for stat in ['mean', 'n_samples', 'bootstrap_std']:
            key = name if stat == 'mean' else f'{name}:{stat}'
            final_metrics[key] = _compute_clipped_stats(values, stat)
    return EvalResult(
        score=final_metrics.pop('score', None),
        metrics=final_metrics,
        htmls=htmls,
        convos=convos,
        metadata={'example_level_metadata': metadata},
    )


@LOAD_DATASET.register_module()
class HealthBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, **kwargs):
        subset = kwargs.get('subset')
        if subset == '':
            data_files = {'test': '2025-05-07-06-14-12_oss_eval.jsonl'}
        elif subset == 'hard':
            data_files = {'test': 'hard_2025-05-08-21-00-10.jsonl'}
        elif subset == 'consensus':
            data_files = {'test': 'consensus_2025-05-09-20-00-46.jsonl'}
        else:
            raise Exception(f'Unrecognized subset type: {subset}')
        dataset = load_dataset(path, data_files=data_files, split='test')
        # Optional dev subsets controlled via env var.
        # HEALTHBENCH_SUBSET can be:
        #   - 'dev_tiny'   -> ~2 base, 1 hard, 2 consensus
        #   - 'dev_50'     -> 50 base only (for benchmarking, matches async harness test)
        #   - 'dev_100'    -> 50 base, 25 hard, 25 consensus (~100 total)
        #   - 'dev_medium' -> ~250 base, 100 hard, 300 consensus
        dev_subset = os.getenv('HEALTHBENCH_SUBSET')
        if dev_subset in ('dev_tiny', 'dev_50', 'dev_100', 'dev_medium'):
            if dev_subset == 'dev_tiny':
                if subset == '':
                    max_n = 2
                elif subset == 'hard':
                    max_n = 1
                elif subset == 'consensus':
                    max_n = 2
                else:
                    max_n = None
            elif dev_subset == 'dev_50':
                # 50 base examples only - for fair comparison with async harness benchmark
                if subset == '':
                    max_n = 50
                else:
                    max_n = 0  # Skip hard/consensus for this benchmark
            elif dev_subset == 'dev_100':
                # ~100 total: 50 base + 25 hard + 25 consensus
                if subset == '':
                    max_n = 50
                elif subset == 'hard':
                    max_n = 25
                elif subset == 'consensus':
                    max_n = 25
                else:
                    max_n = None
            else:  # dev_medium
                if subset == '':
                    max_n = 250
                elif subset == 'hard':
                    max_n = 100
                elif subset == 'consensus':
                    max_n = 300
                else:
                    max_n = None

            if max_n is not None:
                max_n = min(max_n, len(dataset))
                dataset = dataset.select(range(max_n))

        dataset = dataset.map(lambda item: _parse(item))
        return dataset


class HealthBenchEvaluator(BaseEvaluator):
    """only consider the model completion mode, not physician mode / reference
    mode."""

    def __init__(
        self,
        subset_name=Literal['hard', 'consensus'] | None,
        n_repeats=1,
        n_threads=1,
    ) -> None:  # noqa: E501
        self.n_repeats = n_repeats
        self.n_threads = n_threads
        self.subset_name = subset_name
        self.grader_model = ChatCompletionSampler(
            model=os.environ['OC_JUDGE_MODEL'],
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=max_tokens_judge,
        )  # noqa: E501
        judge_mode = os.getenv('HEALTHBENCH_JUDGE_MODE', 'per_rubric')
        judge_mode = judge_mode.strip().lower()
        if judge_mode not in {'per_rubric', 'per_example'}:
            judge_mode = 'per_rubric'
        self.judge_mode: str = judge_mode

    def grade_sample(
        self,
        prompt: list[dict[str, str]],
        response_text: str,
        example_tags: list[str],
        rubric_items: list[RubricItem],
        ) -> tuple[dict, str, list[dict]]:  # noqa: E501
        # construct and grade the sample
        convo_with_response = prompt + [
            dict(content=response_text, role='assistant')
        ]  # noqa: E501

        if self.judge_mode == 'per_example':
            grading_response_list, raw_judge_json = self._grade_sample_per_example(
                convo_with_response, rubric_items)
        else:
            grading_response_list = self._grade_sample_per_rubric(
                convo_with_response, rubric_items)
            raw_judge_json = None

        # compute the overall score
        overall_score = calculate_score(rubric_items, grading_response_list)
        assert overall_score is not None
        metrics = {
            'overall_score': overall_score,
        }

        # compute scores for example-level tags)
        example_tag_scores = {tag: overall_score for tag in example_tags}
        assert len(example_tag_scores) == len(example_tags)  # No duplicates.
        metrics.update(example_tag_scores)

        # compute scores for rubric-level tags
        rubric_tag_items_grades = defaultdict(list)
        for rubric_item, grading_response in zip(
                rubric_items, grading_response_list):  # noqa: E501
            curr_item_tags = set()  # Ensure no duplicates in a rubric item.
            for tag in rubric_item.tags:
                rubric_tag_items_grades[tag].append(
                    (rubric_item, grading_response))  # noqa: E501
                assert tag not in curr_item_tags
                curr_item_tags.add(tag)

        rubric_tag_scores = {}
        for tag, items_grades in rubric_tag_items_grades.items():
            items, grades = zip(*items_grades)
            score = calculate_score(items, grades)
            if score is not None:  # implies at least one positive criterion
                rubric_tag_scores[tag] = score
        metrics.update(rubric_tag_scores)

        # construct the list of explanations and grades
        rubric_items_with_grades = []
        readable_explanation_list = []
        for rubric_item, grading_response in zip(
                rubric_items, grading_response_list):  # noqa: E501
            explanation = grading_response.get(
                'explanation', 'No explanation provided')  # noqa: E501
            criteria_met = grading_response['criteria_met']
            readable_explanation = (f'[{criteria_met}] {rubric_item}\
                        Explanation: {explanation}')
            readable_explanation_list.append(readable_explanation)
            rubric_items_with_grades.append({
                **rubric_item.to_dict(),
                'criteria_met':
                criteria_met,
                'explanation':
                explanation,
            })

        readable_explanation_list.sort(key=lambda x: x.startswith('[False]'),
                                       reverse=True)
        readable_explanation_str = '\n\n'.join(readable_explanation_list)
        readable_explanation_str = f'\n\n{readable_explanation_str}'

        # Optional JSONL dumping of judge outputs for analysis.
        dump_path = os.getenv('HEALTHBENCH_JUDGE_DUMP')
        if dump_path:
            try:
                os.makedirs(os.path.dirname(dump_path), exist_ok=True)
                with open(dump_path, 'a', encoding='utf-8') as f:
                    record: dict[str, Any] = {
                        'rubric_items': [ri.to_dict() for ri in rubric_items],
                        'rubric_items_with_grades': rubric_items_with_grades,
                    }
                    if raw_judge_json is not None:
                        record['raw_judge_json'] = raw_judge_json
                    f.write(json.dumps(record) + '\n')
            except Exception as e:  # pragma: no cover
                if os.getenv('HEALTHBENCH_DEBUG'):
                    print('[HealthBench][JudgeDumpError]', e, flush=True)

        return metrics, readable_explanation_str, rubric_items_with_grades

    def _grade_sample_per_rubric(
        self,
        convo_with_response: list[dict[str, str]],
        rubric_items: list[RubricItem],
    ) -> list[dict]:

        def grade_rubric_item(rubric_item: RubricItem) -> dict:
            convo_str = '\n\n'.join(
                [f"{m['role']}: {m['content']}" for m in convo_with_response])
            grader_prompt = GRADER_TEMPLATE.replace('<<conversation>>',
                                                    convo_str).replace(
                                                        '<<rubric_item>>',
                                                        str(rubric_item))
            messages: MessageList = [dict(content=grader_prompt, role='user')]
            max_retries = 3
            for attempt in range(max_retries):
                sampler_response = self.grader_model(messages)
                grading_response = sampler_response.response_text
                grading_response_dict = parse_json_to_dict(grading_response)
                if 'criteria_met' in grading_response_dict:
                    label = grading_response_dict['criteria_met']
                    if label is True or label is False:
                        return grading_response_dict
                print(f'Grading failed due to bad JSON output, retrying... ({attempt + 1}/{max_retries})')
            # After max retries, return a failure dict
            print(f'Grading failed after {max_retries} retries, marking as failed')
            return {'criteria_met': None, 'reasoning': 'Judge API returned invalid JSON after max retries'}

        grading_response_list = map_with_progress(
            grade_rubric_item,
            rubric_items,
            pbar=False,
        )
        return grading_response_list

    def _grade_sample_per_example(
        self,
        convo_with_response: list[dict[str, str]],
        rubric_items: list[RubricItem],
    ) -> tuple[list[dict], dict]:
        """Grade all rubric items in a single judge call."""
        convo_str = '\n\n'.join(
            [f"{m['role']}: {m['content']}" for m in convo_with_response])

        rubric_lines = []
        for idx, rubric_item in enumerate(rubric_items):
            sign = 'positive' if rubric_item.points > 0 else 'negative'
            rubric_lines.append(
                f'{idx}. ({rubric_item.points} points, {sign} criterion) '
                f'{rubric_item.criterion}')
        rubric_block = '\n'.join(rubric_lines)

        grader_prompt = PER_EXAMPLE_GRADER_TEMPLATE.replace(
            '<<conversation>>', convo_str).replace('<<rubric_items>>',
                                                   rubric_block)
        messages: MessageList = [dict(content=grader_prompt, role='user')]

        # Debug logging for prompt stats
        if os.getenv('HEALTHBENCH_DEBUG'):
            prompt_chars = len(grader_prompt)
            prompt_tokens_est = prompt_chars // 4  # rough estimate
            print(f'[HealthBench][PerExample] Prompt stats: {prompt_chars:,} chars, '
                  f'~{prompt_tokens_est:,} tokens, {len(rubric_items)} rubric items',
                  flush=True)
            # Log first 500 chars of conversation for identification
            convo_preview = convo_str[:500].replace('\n', ' ')
            print(f'[HealthBench][PerExample] Conversation preview: {convo_preview}...',
                  flush=True)

        max_retries = 3
        for attempt in range(max_retries):
            sampler_response = self.grader_model(messages)
            grading_response = sampler_response.response_text
            grading_response_dict = parse_json_to_dict(grading_response)
            # Accept either top-level mapping or nested under "criteria".
            criteria_obj = grading_response_dict.get('criteria')
            if isinstance(criteria_obj, dict):
                criteria_map = criteria_obj
            else:
                criteria_map = grading_response_dict

            grading_response_list: list[dict] = []
            all_labels_valid = True
            for idx in range(len(rubric_items)):
                key = str(idx)
                item_resp = criteria_map.get(key, {})
                label = item_resp.get('criteria_met')
                if label is True or label is False:
                    grading_response_list.append({
                        'criteria_met':
                        label,
                        'explanation':
                        item_resp.get('explanation',
                                      'No explanation provided'),
                    })
                else:
                    all_labels_valid = False
                    break

            if all_labels_valid and grading_response_list:
                return grading_response_list, grading_response_dict

            print(
                f'Per-example grading failed due to bad JSON output, retrying... ({attempt + 1}/{max_retries})',
                flush=True)
        
        # After max retries, return failure placeholders
        print(f'Per-example grading failed after {max_retries} retries, marking as failed')
        failed_list = [{'criteria_met': None, 'reasoning': 'Judge API failed'} for _ in rubric_items]
        return failed_list, {'error': 'max_retries_exceeded'}

    def score(self, predictions, references, test_set):
        results = []
        if len(predictions) != len(references):
            return {
                'error': 'preds and refrs have different length'
            }  # noqa: W291, E501
        for idx, (i, j) in enumerate(zip(predictions, references)):
            response_usage = None
            actual_queried_prompt_messages = test_set[idx]['prompt']
            response_text = i
            row = test_set[idx]  # noqa: W291
            
            # Log which example we're processing
            if os.getenv('HEALTHBENCH_DEBUG'):
                prompt_id = row.get('prompt_id', f'example_{idx}')
                print(f'[HealthBench] Evaluating example {idx+1}/{len(predictions)} '
                      f'(prompt_id={prompt_id})', flush=True)
            
            metrics, readable_explanation_str, rubric_items_with_grades = (
                self.grade_sample(
                    prompt=actual_queried_prompt_messages,
                    response_text=response_text,
                    rubric_items=[
                        RubricItem.from_dict(d) for d in row['rubrics']
                    ],  # noqa: E501
                    example_tags=row['example_tags'],
                ))

            score = metrics['overall_score']
            convo = actual_queried_prompt_messages + [
                dict(content=response_text, role='assistant')
            ]
            results.append(
                SingleEvalResult(
                    html=None,
                    score=score,
                    convo=convo,
                    metrics=metrics,
                    example_level_metadata={
                        'score':
                        score,
                        'usage':
                        get_usage_dict(response_usage),
                        'rubric_items':
                        rubric_items_with_grades,
                        'prompt':
                        actual_queried_prompt_messages,
                        'completion':
                        [dict(content=response_text,
                              role='assistant')],  # noqa: E501
                        'prompt_id':
                        row['prompt_id'],
                        'completion_id':
                        hashlib.sha256(
                            (row['prompt_id'] +
                             response_text).encode('utf-8')).hexdigest(),
                    },
                ))
        results = _aggregate_get_clipped_mean(results)
        assert results.metrics is not None
        metrics = results.metrics | {'score': results.score}
        metrics = dict(sorted(metrics.items()))
        acc = metrics.get('f1_score', metrics.get('score', None))
        return {'accuracy': acc}
