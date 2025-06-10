# Investigating Robustness of Parameter Allocation Strategies in Mix-and-Match Adapters

This project explores how parameter allocation between components of Mix-and-Match (MAM) Adapters affects downstream performance in parameter-efficient fine-tuning (PEFT). We run extensive ablation studies to evaluate whether the default allocation ratios hold across different parameter budgets and tasks in the SuperGLUE benchmark.

## 🧠 Key Features

- **PEFT-focused analysis**: Evaluates a top-performing Mixture-of-Adapters (MoA) architecture combining Prefix-Tuning and Scaled Parallel Adapters.
- **Allocation ratio ablation**: Tests different splits between Prefix-Tuning and Adapter modules.
- **Cross-scale comparison**: Examines robustness of optimal allocations across 1%, 5%, and 9.15% parameter budgets.
- **Benchmark coverage**: Uses RTE, BoolQ, and COPA tasks from SuperGLUE.

## 🎯 Motivation

While Mix-and-Match Adapters have shown strong performance with minimal parameters, their default parameter allocation strategy remains under-explored. This project challenges the assumption that a fixed allocation (e.g., 6% Prefix / 94% Adapter) is optimal across tasks and budget scales. We demonstrate that tuning this ratio can lead to significant performance gains and highlight the importance of budget-aware configuration.

## 🔍 Architecture Overview

The MAM Adapter combines:
- **Prefix-Tuning Module** (in attention layers): Adds learnable tokens to queries and keys.
- **Scaled Parallel Adapter** (in FFN layers): Uses LoRA-style injection of low-rank updates.

Allocation ratio = (Prefix-Tuning params) / (Total tunable params)  
Tunable budget = (Tunable params) / (Total model params)

       ┌────────────────────┐
       │ Pretrained BERT   │
       └─────────┬──────────┘
                 ↓
   ┌──────────────────────────────┐
   │ Add Prefix-Tuning in MHA     │
   └─────────┬─────────┬──────────┘
             ↓         ↓
 FFN ← Add Scaled Parallel Adapter
             ↓
         Output logits


## 📊 Evaluation Results

We report task accuracy for each allocation ratio and parameter budget:

| Budget | Prefix Ratio | RTE (%) | BoolQ (%) | COPA (%) |
|--------|---------------|---------|------------|-----------|
| 1%     | 0.50          | 63.9    | **73.5**   | 64.0      |
| 5%     | 0.50          | **69.0**| 72.8       | 64.0      |
| 9.15%  | 0.50          | 65.3    | 71.8       | **68.0**  |

- Default config (Prefix 6%, Budget 9.15%) is suboptimal for all tasks.
- Optimal ratios vary by task and scale, e.g., COPA prefers higher Adapter allocation.
- Accuracy variance is lower with larger budgets, indicating more stability at scale.

## 🧪 Experimental Setup

- **Backbone**: BERT-base-uncased
- **Tasks**: RTE (entailment), BoolQ (boolean QA), COPA (causal reasoning)
- **Tunable Budgets**: 1%, 5%, 9.15% of total model parameters
- **Allocation Ratios Tested**: 0%, 6%, 50%, 90%, 100%
- **Metrics**: Validation accuracy
- **Fine-tuning**: Grid search over learning rate, batch size, epochs, and regularization

## 📌 Key Findings

- The 6%/94% default MAM allocation is not optimal for any task or budget.
- Task-specific tuning of allocation ratios improves performance.
- Larger budgets reduce sensitivity to allocation changes, suggesting robustness at scale.

## ⚠️ Limitations

- Limited hyperparameter tuning may restrict observed performance peaks.
- Only BERT-base and three tasks are evaluated; results may not generalize to larger LLMs.
- Lack of test set labels restricts generalization evaluation beyond validation accuracy.
