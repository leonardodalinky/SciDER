"""Internal validation: idea search vs. baselines across four ablation modes.

For each query the eval runs four configurations and reports:
  baseline_novelty    — top seed by novelty_score (current single-pass behaviour)
  baseline_composite  — top seed by 4D composite rubric (score-only, no iterations)
  improve_only        — full search loop, Improve operator only (fraction=1.0)
  improve_combine     — full search loop, Improve + Combine (default config)

Lift is computed as (final_best − initial_best) / initial_best where initial_best
is the best composite score in the seed population immediately after the first
scoring pass — before any operators run.

Judge (critic model) runs three pairwise comparisons per query:
  A: baseline_novelty  vs B: improve_combine   → main claim
  A: baseline_composite vs B: improve_combine  → value of search iterations
  A: improve_only      vs B: improve_combine   → marginal value of Combine

Usage:
    python -m evals.eval_idea_search [--sequential] [--call-delay SECONDS]
                                     [--queries N [N ...]] [--output PATH]

Models are loaded from model_settings/role_defaults.yaml.
The judge uses the "critic" role to avoid self-confirmation.
"""

from __future__ import annotations

import copy
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from scider.agents.ideation_agent.idea_search import (
    IdeaNode,
    IdeaSearchResult,
    run_idea_search,
    score_population,
)
from scider.core.llms import ModelRegistry
from scider.core.types import Message
from scider.core.utils import parse_json_from_llm_response
from scider.default.models import register_defaults_from_yaml

# ---------------------------------------------------------------------------
# Seed data: 20 queries × 6 ideas each
# Ideas vary in quality so the search has meaningful signal to act on.
# ---------------------------------------------------------------------------

QUERIES: list[dict] = [
    # ── NLP ─────────────────────────────────────────────────────────────────
    {
        "query": "efficient transformers for long document understanding",
        "domain": "NLP",
        "ideas": [
            {"title": "Sliding Window Attention with Learned Window Sizes", "description": "Replace fixed sliding windows with learnable window boundaries that adapt per layer and document type. A small auxiliary network predicts sizes from input entropy.", "rationale": "Fixed windows ignore document structure; learned boundaries could reduce compute while preserving cross-section dependencies.", "experiment": "Compare to Longformer on SCROLLS; train auxiliary network end-to-end.", "contribution": "Parameter-efficient long-range attention that adapts to content.", "novelty_score": 7.0},
            {"title": "Hierarchical Chunking with Cross-Chunk Memory", "description": "Divide documents into semantically coherent chunks, run BERT-scale attention within chunks, and a smaller cross-chunk transformer on chunk representations.", "rationale": "Avoids quadratic attention across the full document while retaining global coherence.", "experiment": "Evaluate on QuALITY QA; ablate chunk granularity.", "contribution": "Practical two-stage architecture for book-length documents.", "novelty_score": 6.0},
            {"title": "Token Dropping via Importance Scoring Before Attention", "description": "Use a lightweight scorer to drop the least informative tokens before the attention layer, reducing sequence length dynamically. Dropped tokens are approximated via interpolation during decoding.", "rationale": "Not all tokens are equally important; early pruning can cut FLOPs while preserving accuracy.", "experiment": "Benchmark on QASPER; compare drop ratios of 20/40/60%.", "contribution": "Adaptive length reduction with bounded information loss.", "novelty_score": 6.5},
            {"title": "Linear Attention Kernels with Positional Bias Correction", "description": "Extend linear attention (e.g., Performer) with a lightweight additive bias term that compensates for relative-position signal lost in the kernel approximation.", "rationale": "Linear attention loses positional awareness; correcting for this could close the accuracy gap with full attention at linear cost.", "experiment": "Probe on Synthetic Reasoning Task and SCROLLS/SummScreenFD.", "contribution": "Drop-in fix for linear attention on long-context tasks requiring positional reasoning.", "novelty_score": 7.5},
            {"title": "State-Space Model Hybrid with Selective Full Attention", "description": "Use a Mamba-style recurrent backbone for most tokens and trigger local full-attention windows only at detected sentence boundaries or explicit query positions.", "rationale": "SSMs are fast globally but weak at precise retrieval; injecting full attention at key positions could give the best of both worlds.", "experiment": "Evaluate on LongBench; compare to pure Mamba and hybrid baselines.", "contribution": "Principled trigger mechanism for selective dense attention in SSM architectures.", "novelty_score": 8.0},
            {"title": "Attention Head Specialization for Long-Range vs. Local Patterns", "description": "Explicitly partition attention heads into long-range and short-range groups and penalize deviation from this assignment during training via an auxiliary regularizer.", "rationale": "Emergent head specialization is documented; encouraging it from the start may improve efficiency and interpretability.", "experiment": "Visualize head patterns on arXiv summarization; compare to vanilla transformer on SCROLLS.", "contribution": "Structured inductive bias for more interpretable long-document models.", "novelty_score": 5.5},
        ],
    },
    # ── FL/Privacy ───────────────────────────────────────────────────────────
    {
        "query": "privacy-preserving federated learning with heterogeneous data",
        "domain": "FL/Privacy",
        "ideas": [
            {"title": "Differentially Private Gradient Clipping Calibrated to Layer Sensitivity", "description": "Instead of uniform DP clipping, compute per-layer sensitivity estimates and allocate the privacy budget proportionally, giving low-sensitivity layers more signal.", "rationale": "Uniform clipping wastes budget on layers with low sensitivity; adaptive allocation should improve utility at the same epsilon.", "experiment": "Compare on CIFAR-10 and Shakespeare under (ε=1, δ=1e-5).", "contribution": "Layer-aware DP gradient mechanism with improved accuracy-privacy tradeoff.", "novelty_score": 7.5},
            {"title": "Personalized Federated Learning via Sparse Private Adapters", "description": "Each client learns a small sparse LoRA-style adapter privately; only the shared global backbone is aggregated across clients with DP guarantees.", "rationale": "Aggregating the full model under DP on heterogeneous data causes excessive noise; private adapters confine personalization locally.", "experiment": "Evaluate on LEAF benchmarks (FEMNIST, Reddit).", "contribution": "Modular architecture separating global knowledge (shared, DP) from local adaptation (private).", "novelty_score": 8.0},
            {"title": "Secure Aggregation with Byzantine-Robust Median", "description": "Replace mean aggregation with the geometric median under a secure multiparty computation protocol, making federation robust to both poisoning and privacy attacks.", "rationale": "Mean aggregation is vulnerable to poisoning; secure median is robust but computationally expensive — recent SMPC advances may make it tractable.", "experiment": "Measure accuracy, Byzantine tolerance (10-30% malicious clients), communication overhead.", "contribution": "Simultaneous robustness to privacy leakage and model poisoning.", "novelty_score": 6.5},
            {"title": "Communication-Efficient Federated Learning via Sketch Aggregation", "description": "Clients send compact Count-Sketch summaries of gradients; the server reconstructs the aggregate from sketches.", "rationale": "Bandwidth is often the bottleneck in cross-device FL; sketching reduces upload by 10-100× at a small reconstruction cost.", "experiment": "Benchmark on Cross-Device FL simulation; measure accuracy vs. compression ratio.", "contribution": "Practical communication reduction without trusted aggregator assumptions.", "novelty_score": 5.5},
            {"title": "Federated Domain Adaptation via Shared Feature Alignment", "description": "Clients with different data distributions align local feature spaces using a shared contrastive objective before local training, reducing distribution shift without sharing raw data.", "rationale": "Heterogeneous data hurts convergence; feature alignment may be more privacy-friendly than sharing samples.", "experiment": "DomainNet benchmark split across 6 clients with domain mismatch; compare to FedProx, MOON.", "contribution": "Contrastive federation objective for heterogeneous domain settings.", "novelty_score": 7.0},
            {"title": "Asynchronous Federated Learning with Staleness-Aware Weighting", "description": "Allow clients to submit updates asynchronously and down-weight stale gradients proportionally to the number of rounds elapsed since the update was computed.", "rationale": "Synchronous FL stalls on slow clients; asynchronous approaches risk divergence without staleness correction.", "experiment": "Simulate heterogeneous device speeds; compare convergence curves on CIFAR-100.", "contribution": "Staleness-aware aggregation rule for asynchronous FL.", "novelty_score": 6.0},
        ],
    },
    # ── Drug Discovery ───────────────────────────────────────────────────────
    {
        "query": "graph neural networks for molecular property prediction in drug discovery",
        "domain": "Chemistry",
        "ideas": [
            {"title": "3D Equivariant GNN with Torsion Angle Features", "description": "Extend E(3)-equivariant graph networks by explicitly encoding torsion angles as edge features, capturing conformational properties relevant to binding.", "rationale": "Torsion angles encode conformational flexibility not captured by bond angle alone; they are strongly predictive of binding affinity.", "experiment": "QM9 benchmark + PCBA ADMET tasks; ablate torsion features vs. distance-only.", "contribution": "Richer 3D molecular representation with torsion awareness.", "novelty_score": 7.5},
            {"title": "Hierarchical Molecular GNN: Atom, Fragment, and Scaffold Levels", "description": "Build a three-level graph hierarchy — atoms, functional groups, and scaffold — and propagate messages up and down, allowing reasoning at multiple scales.", "rationale": "Drug-likeness is often determined at the scaffold level; multi-scale reasoning should improve generalization across scaffold hops.", "experiment": "Scaffold-split benchmarks from MoleculeNet; compare to flat MPNNs.", "contribution": "Hierarchical message passing aligned with medicinal chemist intuition.", "novelty_score": 8.0},
            {"title": "Uncertainty-Aware GNN for Active Learning in Hit Expansion", "description": "Train a GNN with MC Dropout or Deep Ensemble for calibrated uncertainty estimates, then use these to select compounds for synthesis in an active learning loop.", "rationale": "Drug discovery iterates over expensive assays; uncertainty-guided selection reduces experiments needed to find active compounds.", "experiment": "Simulate active learning on ChEMBL bioactivity data.", "contribution": "Practical active learning pipeline for hit expansion using GNN uncertainty.", "novelty_score": 7.0},
            {"title": "Reaction-Aware GNN for Retrosynthesis-Driven Property Prediction", "description": "Augment molecular property prediction with a retrosynthesis feasibility score — molecules predicted to be hard to synthesize are penalized — learning to predict synthesizability jointly with property.", "rationale": "Many predicted drug candidates are never synthesized; co-optimizing for synthesizability makes predictions more actionable.", "experiment": "Train on USPTO reaction database + ADMET datasets; evaluate on virtual screening hit lists.", "contribution": "Multi-task GNN that jointly predicts target property and synthetic accessibility.", "novelty_score": 8.5},
            {"title": "Graph Transformer with Attention Biased by Fingerprint Similarity", "description": "Initialize attention weights using Morgan fingerprint similarity between atom neighborhoods, providing a pharmacophore-inspired prior.", "rationale": "Fingerprint similarity is a well-validated proxy for biological activity; using it as an attention prior could accelerate convergence.", "experiment": "Compare to GraphGPS and Graphormer on MoleculeNet.", "contribution": "Chemoinformatics-informed attention mechanism for molecular graph transformers.", "novelty_score": 6.5},
            {"title": "Protein-Ligand Interaction GNN for Target-Specific Property Prediction", "description": "Model the protein-ligand interaction graph jointly, placing the protein binding pocket as context nodes alongside the ligand graph.", "rationale": "Target-agnostic models miss target-specific SAR; joint modeling captures interaction geometry.", "experiment": "PDBbind benchmark; compare to DeepDTA, HOLOPROT.", "contribution": "Target-aware GNN for binding affinity prediction using 3D pocket-ligand graphs.", "novelty_score": 7.0},
        ],
    },
    # ── Causal Inference ─────────────────────────────────────────────────────
    {
        "query": "causal inference methods for observational healthcare data",
        "domain": "Statistics",
        "ideas": [
            {"title": "Doubly-Robust Estimator with Neural Network Nuisance Models", "description": "Replace parametric propensity score and outcome models in AIPW with flexible neural networks, maintaining double robustness while gaining nonparametric expressiveness.", "rationale": "Healthcare covariates are high-dimensional; neural networks reduce misspecification bias.", "experiment": "ACIC 2018 benchmark + MIMIC-IV sepsis treatment data; compare to linear DR, TMLE.", "contribution": "Scalable doubly-robust ATE estimator for high-dimensional EHR data.", "novelty_score": 6.5},
            {"title": "Instrumental Variable Estimation via Neural Proxy Variables", "description": "Learn proxy instruments from raw clinical notes using a contrastive language model, then apply 2SLS with these learned proxies to identify causal effects when direct instruments are unavailable.", "rationale": "Valid instruments are rare in healthcare; learning proxies from unstructured text may recover identifiability in confounded settings.", "experiment": "Simulate IV identification on semi-synthetic MIMIC data with known ground truth.", "contribution": "NLP-powered instrument mining for causal estimation from clinical text.", "novelty_score": 8.5},
            {"title": "Causal Discovery with Forbidden Edge Constraints from Clinical Guidelines", "description": "Incorporate known clinical constraints (e.g., drug → side effect is plausible; side effect → drug is not) as forbidden edges in a score-based causal discovery algorithm.", "rationale": "Standard causal discovery ignores domain knowledge; clinical constraints should improve accuracy.", "experiment": "Evaluate on synthetic DAGs from clinical ontologies; apply to Alzheimer's EHR data.", "contribution": "Knowledge-constrained causal discovery for clinical decision support.", "novelty_score": 7.0},
            {"title": "Sensitivity Analysis for Unmeasured Confounding in Time-Series EHR", "description": "Extend Rosenbaum-style sensitivity analysis to longitudinal EHR data with time-varying confounders, computing bounds on treatment effect estimates as a function of confounder strength.", "rationale": "Unmeasured confounding is endemic in observational studies; sensitivity bounds are critical for clinical decision-making.", "experiment": "Apply to CPRD/OPTUM diabetes medication cohort.", "contribution": "Tractable sensitivity analysis for marginal structural models with time-varying confounders.", "novelty_score": 7.5},
            {"title": "Heterogeneous Treatment Effects via Causal Forests on Genomic Data", "description": "Apply causal forests to high-dimensional genomic covariates to estimate individual-level treatment effect heterogeneity in cancer immunotherapy trials.", "rationale": "Average effects mask patient subgroups that respond differentially; causal forests identify biomarker-defined subgroups.", "experiment": "TCGA immunotherapy cohort; compare to LASSO-selected subgroup analysis.", "contribution": "Genomic subgroup identification for personalized immunotherapy using CATE estimation.", "novelty_score": 7.0},
            {"title": "Regression Discontinuity Designs for Treatment Threshold Studies", "description": "Apply sharp and fuzzy RDD to clinical threshold decisions (e.g., A1c > 7.0 triggers insulin) to estimate local average treatment effects without full randomization.", "rationale": "Clinical thresholds create natural quasi-experiments; RDD recovers near-causal estimates near the cutoff.", "experiment": "Apply to HbA1c → insulin initiation in UK Biobank.", "contribution": "Quasi-experimental toolkit for causal estimation around clinical decision thresholds.", "novelty_score": 6.0},
        ],
    },
    # ── Speech ───────────────────────────────────────────────────────────────
    {
        "query": "self-supervised learning for low-resource speech recognition",
        "domain": "Speech",
        "ideas": [
            {"title": "Masked Acoustic Modeling with Phoneme-Level Targets", "description": "Pre-train a speech encoder by predicting phoneme labels at masked positions rather than raw acoustic features, using a small phoneme recognizer as the target network.", "rationale": "Predicting discrete linguistic units gives a stronger training signal than reconstructing spectrograms.", "experiment": "Evaluate on FLEURS (25 languages) with 1h/10h fine-tuning; compare to wav2vec 2.0, HuBERT.", "contribution": "Linguistically-motivated pre-training objective for multilingual low-resource ASR.", "novelty_score": 7.0},
            {"title": "Cross-Lingual Transfer via Shared Phoneme Inventory Alignment", "description": "Map phonemes from high-resource languages to low-resource target language phonemes using a universal IPA phoneme space, then fine-tune only shared phoneme embeddings.", "rationale": "Many low-resource languages share phoneme inventories with higher-resource ones; aligning at the phoneme level should improve transfer.", "experiment": "CMU Wilderness + CommonVoice; evaluate on 10 typologically diverse languages with < 1h data.", "contribution": "Phoneme-aligned cross-lingual transfer for extreme low-resource ASR.", "novelty_score": 8.0},
            {"title": "Data Augmentation via TTS-Driven Acoustic Simulation", "description": "Use a multilingual TTS system to generate synthetic speech for low-resource languages by transferring prosody and phoneme-level acoustic parameters from a few recorded samples.", "rationale": "Lack of transcribed speech is the core bottleneck; TTS augmentation can multiply the effective training set.", "experiment": "Measure WER improvement on FLEURS with 0/1/5/10 seed recordings for TTS training.", "contribution": "Scalable low-cost data generation pipeline for under-resourced languages.", "novelty_score": 6.5},
            {"title": "Contrastive Disentanglement of Speaker Identity and Linguistic Content", "description": "Learn separate speaker identity and linguistic content representations using a contrastive loss that maximizes mutual information between content representations across speakers.", "rationale": "Speaker variation inflates apparent variability; disentangling it should improve representation quality for downstream recognition.", "experiment": "VCTK + FLEURS; evaluate ASR and speaker verification; measure disentanglement via probing classifiers.", "contribution": "Content-speaker disentangled pre-training for both ASR and speaker verification.", "novelty_score": 7.5},
            {"title": "Prompt-Based Adaptation of Whisper for Low-Resource Languages", "description": "Fine-tune only a small prompt token prepended to Whisper's encoder, learning a language-specific acoustic context without modifying model weights.", "rationale": "Full fine-tuning of Whisper on < 1h data typically overfits; prompt-only adaptation has far fewer parameters.", "experiment": "OpenSLR low-resource languages; compare to full fine-tuning, LoRA, zero-shot Whisper.", "contribution": "Parameter-efficient adaptation strategy for Whisper on unseen low-resource languages.", "novelty_score": 7.0},
            {"title": "Semi-Supervised Conformer Training with Noisy Student on Untranscribed Audio", "description": "Iteratively train a Conformer ASR model by generating pseudo-labels for untranscribed audio, then training a new student on the combined labeled + pseudo-labeled set with added noise.", "rationale": "Untranscribed audio is plentiful; noisy student training can leverage it to reduce WER even when transcribed data is scarce.", "experiment": "VoxPopuli + CommonVoice (10 languages); measure WER vs. amount of pseudo-labeled data.", "contribution": "Iterative noisy student pipeline for scaling low-resource conformer ASR.", "novelty_score": 6.0},
        ],
    },
    # ── CV ───────────────────────────────────────────────────────────────────
    {
        "query": "contrastive learning for visual representation without labels",
        "domain": "CV",
        "ideas": [
            {"title": "Multi-Crop Contrastive Learning with Resolution-Aware Projectors", "description": "Extend multi-crop augmentation by adding a lightweight resolution-aware projection head that conditions on the crop scale, preventing the model from relying on absolute resolution as a shortcut.", "rationale": "Multi-crop improves efficiency but can cause the model to learn scale shortcuts; resolution conditioning removes this signal.", "experiment": "Linear evaluation on ImageNet-1k; compare to DINO, iBOT with same backbone.", "contribution": "Shortcut-resistant multi-crop strategy for self-supervised ViT training.", "novelty_score": 7.0},
            {"title": "Asymmetric Momentum Encoders with Hard Negative Mining", "description": "Maintain a momentum encoder queue and actively mine hard negatives by selecting samples with high cosine similarity to the query but different semantic class.", "rationale": "Random negatives are mostly easy; hard negatives push the encoder to learn finer-grained distinctions.", "experiment": "MoCo v3 baseline + hard negative sampling on ImageNet; transfer to PASCAL VOC.", "contribution": "Hard negative curriculum for momentum contrastive learning.", "novelty_score": 7.5},
            {"title": "Contrastive Pre-training with Semantic Segment Consistency", "description": "Apply contrastive loss at the segment level by grouping pixels into semantic regions using off-the-shelf oversegmentation and requiring consistency across augmented views at the segment level.", "rationale": "Instance-level contrastive learning discards spatial structure needed for dense tasks like segmentation.", "experiment": "Evaluate on COCO instance segmentation and ADE20k after fine-tuning.", "contribution": "Spatially structured self-supervised pre-training for dense prediction tasks.", "novelty_score": 8.0},
            {"title": "Self-Supervised Pretext via Temporal Order Prediction in Video", "description": "Train a visual encoder to correctly order shuffled video frames as a pretext task, forcing temporal reasoning without labels.", "rationale": "Video provides free supervision via temporal structure; ordering prediction requires understanding motion and causality.", "experiment": "Evaluate on UCF-101, HMDB-51; compare to VideoMAE and temporal contrastive methods.", "contribution": "Causal temporal pretext task for video-based visual representation learning.", "novelty_score": 6.0},
            {"title": "Patch-Level Masked Feature Prediction with Cluster Targets", "description": "Predict cluster assignments of masked patches rather than raw features, using online k-means as the teacher target — removing the need for a pre-trained target network.", "rationale": "BEiT-style masked prediction requires a dVAE or DALL-E tokenizer; cluster targets are simpler and do not require pre-training the target.", "experiment": "ImageNet linear probing and fine-tuning; compare to MAE, BEiT, data2vec.", "contribution": "Tokenizer-free masked image modelling with online cluster targets.", "novelty_score": 7.5},
            {"title": "Equivariant Contrastive Learning for 3D Point Clouds", "description": "Apply SO(3)-equivariant contrastive learning to 3D point clouds, where positive pairs are rotation-augmented versions of the same object and the encoder is constrained to produce equivariant representations.", "rationale": "Standard contrastive learning ignores 3D symmetry; equivariant representations generalize better to unseen orientations.", "experiment": "ShapeNet55 shape classification; compare to PointMAE, Point-BERT.", "contribution": "Rotation-equivariant self-supervised 3D encoder for downstream shape tasks.", "novelty_score": 8.0},
        ],
    },
    # ── AI Safety ────────────────────────────────────────────────────────────
    {
        "query": "aligning large language models to human values through feedback",
        "domain": "AI Safety",
        "ideas": [
            {"title": "Constitutional AI with Automated Principle Refinement", "description": "Extend Constitutional AI by having the model propose revisions to its own constitutional principles after observing which principles conflict most in adversarial examples.", "rationale": "Static constitutions may be inconsistent or incomplete; self-refinement could produce more coherent value systems.", "experiment": "Measure harmlessness/helpfulness Pareto frontier on Anthropic HH-RLHF before and after refinement.", "contribution": "Dynamic self-updating constitution for Constitutional AI.", "novelty_score": 8.0},
            {"title": "Reward Model Ensembles to Reduce Overoptimization", "description": "Train an ensemble of reward models and use the minimum ensemble prediction rather than the mean to reduce reward hacking during RL fine-tuning.", "rationale": "Single reward models are easily hacked; using the minimum provides a conservative lower bound less susceptible to exploitation.", "experiment": "Measure KL divergence and reward model score on held-out preference data after PPO.", "contribution": "Ensemble-conservative reward signal for safer RLHF optimization.", "novelty_score": 7.0},
            {"title": "Debate as a Scalable Oversight Method for Complex Tasks", "description": "Have two agents argue opposing sides of a factual question and train a human judge to identify the winning side, then use the debate outcome as a reward signal for model training.", "rationale": "Human oversight fails when tasks exceed human ability; debate amplifies human judgment without requiring superhuman evaluators.", "experiment": "QuALITY comprehension questions where ground truth is known; measure judge accuracy with and without debate.", "contribution": "Scalable oversight mechanism that remains effective as task complexity increases.", "novelty_score": 8.5},
            {"title": "Process Reward Models for Step-Level Alignment", "description": "Train a reward model that scores individual reasoning steps rather than final outputs, providing denser learning signal and penalizing harmful intermediate reasoning.", "rationale": "Outcome-only rewards allow harmful reasoning chains that happen to produce correct answers; step-level supervision is more aligned.", "experiment": "Math reasoning benchmark; compare step-level vs. outcome reward on out-of-distribution problems.", "contribution": "Reasoning-chain aligned reward model for process-level supervision.", "novelty_score": 7.5},
            {"title": "Representation Engineering to Detect and Remove Deceptive States", "description": "Train a probing classifier to identify when a model is representing a deceptive intent in its residual stream, then subtract the deceptive direction during inference.", "rationale": "Models may learn to behave differently when evaluated; activating detection from internal representations is more robust than behavioral testing.", "experiment": "Generate a dataset of deceptive vs. honest generation pairs; measure classification accuracy and effect on downstream honesty benchmarks.", "contribution": "Mechanistic intervention for honesty alignment via representation engineering.", "novelty_score": 8.0},
            {"title": "Red-Teaming Automation via Diversity-Seeking Adversarial LLMs", "description": "Train a red-team model to generate adversarial prompts that are both effective and diverse using a diversity-seeking RL objective, replacing manual red-teaming effort.", "rationale": "Manual red-teaming is slow and tends to cluster around human-intuitive attacks; automated diversity incentivizes coverage of the full failure mode space.", "experiment": "Measure harm rate and prompt diversity on a target LLM; compare to uniform sampling and greedy attack baselines.", "contribution": "Scalable automated red-teaming pipeline with diversity coverage guarantees.", "novelty_score": 7.0},
        ],
    },
    # ── AutoML ───────────────────────────────────────────────────────────────
    {
        "query": "efficient neural architecture search for edge deployment",
        "domain": "AutoML",
        "ideas": [
            {"title": "Hardware-Aware NAS with Differentiable Latency Proxy", "description": "Incorporate a differentiable hardware latency proxy into the NAS loss function, penalizing architectures that exceed target latency on the specific edge device class.", "rationale": "FLOPs are a poor proxy for actual device latency; device-specific differentiable proxies produce deployable architectures directly.", "experiment": "Search on CIFAR-10, evaluate on ImageNet; measure latency on Raspberry Pi 4 and Jetson Nano.", "contribution": "Device-class-aware differentiable NAS with actual hardware latency optimization.", "novelty_score": 7.5},
            {"title": "Once-for-All Networks with Knowledge Distillation Across Subnets", "description": "Extend once-for-all training by distilling knowledge from larger subnets to smaller ones during training, improving accuracy of the smallest deployable subnets.", "rationale": "OFA training produces variable-size networks but smaller subnets underperform; subnet distillation closes this gap.", "experiment": "OFA baseline on ImageNet; compare subnet accuracy before/after distillation across depth/width configurations.", "contribution": "Distillation-enhanced once-for-all training for improved smallest-subnet accuracy.", "novelty_score": 7.0},
            {"title": "Predictor-Guided NAS with Architecture Encoding via GNN", "description": "Train a GNN-based performance predictor on a small set of evaluated architectures and use it to guide the NAS search, reducing the number of full evaluations.", "rationale": "Full training of each candidate architecture is prohibitively expensive; predictor-guided search achieves comparable results with far fewer evaluations.", "experiment": "NAS-Bench-201 and DARTS search space; compare predictor-guided vs. random search at equal query budget.", "contribution": "Query-efficient NAS via GNN predictor with <100 training examples.", "novelty_score": 7.0},
            {"title": "Evolutionary NAS with Multi-Objective Pareto Optimization", "description": "Apply NSGA-II evolutionary search over (accuracy, latency, memory) objectives simultaneously, maintaining a Pareto front of non-dominated architectures across the search.", "rationale": "Single-objective NAS discards the tradeoff surface; Pareto optimization gives practitioners a range of deployable options.", "experiment": "Search on MobileNet search space; plot Pareto front vs. single-objective search on ImageNet.", "contribution": "Multi-objective evolutionary NAS producing a deployable Pareto frontier.", "novelty_score": 6.5},
            {"title": "Zero-Cost Proxy NAS via Gradient-Free Signal at Initialization", "description": "Use zero-cost proxies (e.g., synflow, grad-norm, jacov) that require only a single forward/backward pass at initialization to rank architectures without any training.", "rationale": "Training-free proxies reduce NAS cost to minutes; recent benchmarks show they can achieve high rank correlation with final accuracy.", "experiment": "Evaluate proxy rank correlation on NAS-Bench-101 and NATS-Bench; compare combinations of proxies.", "contribution": "Training-free NAS via combined zero-cost proxy ensemble with strong rank correlation.", "novelty_score": 8.0},
            {"title": "Sparse Supernet Training with Lottery Ticket Extraction", "description": "Train a sparse supernet using a differentiable sparsity mask and extract candidate architectures by keeping the top-k weights per layer based on mask magnitude.", "rationale": "Standard supernet training allows weight coupling between subnets; sparse training with masks reduces coupling and produces cleaner candidate architectures.", "experiment": "DARTS and EfficientNet search spaces; compare to DARTS and single-path NAS.", "contribution": "Weight-decoupled sparse supernet training for cleaner one-shot NAS.", "novelty_score": 6.5},
        ],
    },
    # ── Medical AI ───────────────────────────────────────────────────────────
    {
        "query": "multimodal AI for automated radiology report generation",
        "domain": "Medical AI",
        "ideas": [
            {"title": "Anatomy-Guided Cross-Attention for Structured Report Generation", "description": "Use a pre-trained anatomy segmentation model to define anatomical regions as structured attention queries, ensuring the report generation model attends to each region systematically.", "rationale": "Free-form generation often omits findings in specific anatomical regions; structured anatomical attention ensures coverage.", "experiment": "IU X-Ray and MIMIC-CXR; measure ROUGE, ClinicalBERT-F1, and finding recall per anatomical region.", "contribution": "Anatomy-complete radiology report generation with structured cross-attention.", "novelty_score": 7.5},
            {"title": "Contrastive Pre-training on Image-Report Pairs with Negative Mining", "description": "Pre-train a radiology vision-language model using contrastive learning on image-report pairs with hard negatives drawn from radiologically similar but clinically different cases.", "rationale": "Random negatives are easy; hard negatives from radiologically similar cases force the model to distinguish subtle clinical differences.", "experiment": "Pre-train on MIMIC-CXR; evaluate on zero-shot classification and fine-tuned report generation.", "contribution": "Clinically discriminative vision-language pre-training for radiology.", "novelty_score": 8.0},
            {"title": "Fact-Checked Report Generation via Radiology Knowledge Graph", "description": "After generating a report draft, automatically verify each factual claim against a radiology knowledge graph and revise claims that contradict known anatomical or pathological relationships.", "rationale": "LLMs generate plausible but factually incorrect radiology findings; knowledge graph verification reduces clinical errors.", "experiment": "Measure factual accuracy on RadGraph annotations before/after fact-checking; compare to standard generation.", "contribution": "Knowledge-grounded radiology report generation with automatic factual verification.", "novelty_score": 8.5},
            {"title": "Progressive Report Generation from Coarse to Fine-Grained Findings", "description": "Generate the report in two passes: first a high-level impression section summarizing key findings, then a detailed findings section grounded in the impression.", "rationale": "Clinical reports follow a coarse-to-fine structure that mirrors radiologist workflow; enforcing this structure should improve coherence.", "experiment": "MIMIC-CXR; ablate two-pass vs. single-pass generation on coherence metrics.", "contribution": "Hierarchical generation approach that mirrors clinical reporting workflow.", "novelty_score": 6.5},
            {"title": "Uncertainty-Aware Report Generation with Confidence Calibration", "description": "Train the report generation model to express uncertainty explicitly (e.g., 'possible pneumonia' vs. 'pneumonia') and calibrate these confidence expressions against radiologist agreement rates.", "rationale": "Overconfident AI reports increase risk of clinical error; calibrated uncertainty expressions help radiologists appropriately weight AI findings.", "experiment": "Measure calibration of linguistic confidence markers against majority vote among 3 radiologists on CheXpert.", "contribution": "Calibrated uncertainty expression in AI-generated radiology reports.", "novelty_score": 7.0},
            {"title": "Multi-View Fusion Network for Chest X-Ray Report Generation", "description": "Explicitly fuse frontal and lateral X-ray views using cross-view attention before report generation, rather than treating each view independently.", "rationale": "Clinical practice requires integrating findings across views; single-view models miss information that is only visible from specific angles.", "experiment": "MIMIC-CXR lateral+frontal pairs; compare to frontal-only generation on finding-level recall.", "contribution": "Cross-view attention fusion for complete multi-view chest X-ray report generation.", "novelty_score": 7.0},
        ],
    },
    # ── Continual Learning ───────────────────────────────────────────────────
    {
        "query": "continual learning without catastrophic forgetting in neural networks",
        "domain": "ML",
        "ideas": [
            {"title": "Gradient Projection onto Orthogonal Memory Subspaces", "description": "Project gradient updates for new tasks onto the orthogonal complement of the subspace spanned by gradients of past tasks, preventing interference with previously learned representations.", "rationale": "Catastrophic forgetting is caused by gradient interference; orthogonal projection guarantees no degradation on past tasks.", "experiment": "Permuted MNIST, Split CIFAR-100; compare to EWC, GEM, A-GEM.", "contribution": "Provably non-forgetting gradient constraint for continual learning.", "novelty_score": 8.0},
            {"title": "Sparse Rehearsal via Coreset Selection with Coverage Guarantee", "description": "Select a minimal rehearsal memory by maximizing geometric coverage of the past task distribution, ensuring that no region of the past data space is left unrepresented.", "rationale": "Random rehearsal memory selection may leave regions unrepresented; coverage-based selection improves forgetting resistance with the same memory budget.", "experiment": "Split CIFAR-100 and CORe50; vary memory size from 100 to 2000 examples.", "contribution": "Coverage-optimal coreset selection for rehearsal-based continual learning.", "novelty_score": 7.0},
            {"title": "Task-Specific Adapters with Shared Feature Backbone", "description": "Freeze a pre-trained backbone and learn small task-specific adapters for each task, storing adapters rather than rehearsal data to prevent forgetting.", "rationale": "Adapters add minimal parameters per task and prevent all forgetting by construction; the frozen backbone provides shared transfer.", "experiment": "Split ImageNet benchmark with 100 tasks; compare to PackNet, HAT on memory efficiency vs. accuracy.", "contribution": "Zero-forgetting continual learning via parameter-isolated task adapters.", "novelty_score": 7.5},
            {"title": "Dynamic Architecture Expansion with Knowledge Distillation on Growth", "description": "Grow the network by adding new neurons for each new task, using knowledge distillation to transfer previously learned representations to the expanded network.", "rationale": "Fixed-capacity networks trade off plasticity and stability; dynamic expansion avoids this by adding capacity while preserving old knowledge via distillation.", "experiment": "Benchmark on CORe50 and CLOC with 39-class online learning; measure backward transfer and memory overhead.", "contribution": "Capacity-adaptive continual learner with distillation-based knowledge preservation on expansion.", "novelty_score": 7.0},
            {"title": "Generative Replay with Conditional VAE for Past Task Simulation", "description": "Train a conditional VAE as a generative model of past task data and use it to produce synthetic rehearsal examples, eliminating the need to store raw past examples.", "rationale": "Storing raw past examples raises privacy and memory concerns; generative replay approximates rehearsal without storing data.", "experiment": "Permuted MNIST, Sequential CUB; compare to real-data rehearsal and EWC on forgetting metric.", "contribution": "Privacy-preserving generative rehearsal for continual learning without stored past data.", "novelty_score": 6.5},
            {"title": "Meta-Learning Initialization for Fast Continual Adaptation", "description": "Use MAML-style meta-learning to find an initialization that can be quickly adapted to new tasks with few gradient steps while suffering minimal forgetting on prior tasks.", "rationale": "Standard meta-learning does not explicitly optimize for continual learning; adapting MAML with an anti-forgetting term could combine fast adaptation with retention.", "experiment": "Few-shot continual learning benchmark; measure adaptation speed and backward transfer.", "contribution": "Meta-learned initialization optimized jointly for fast adaptation and retention.", "novelty_score": 7.5},
        ],
    },
    # ── Code Generation ──────────────────────────────────────────────────────
    {
        "query": "large language models for automated code generation and repair",
        "domain": "SE/PL",
        "ideas": [
            {"title": "Execution-Guided Code Generation with Dynamic Test Synthesis", "description": "Generate candidate code, execute it against automatically synthesized test cases derived from the problem specification, and iteratively revise based on execution feedback.", "rationale": "Static generation without execution feedback produces plausible-looking but incorrect code; execution grounding closes the feedback loop.", "experiment": "HumanEval+ and SWE-bench Lite; measure pass@1 with and without execution feedback loop.", "contribution": "Execution-grounded iterative code generation with dynamic test synthesis.", "novelty_score": 7.5},
            {"title": "Retrieval-Augmented Code Repair with Bug Pattern Library", "description": "Maintain a library of common bug patterns and their fixes; retrieve the most similar past bug-fix pairs when repairing a new bug and condition generation on the retrieved examples.", "rationale": "LLMs repeat known bug patterns; retrieval from a bug library provides directly applicable repair templates.", "experiment": "Defects4J and QuixBugs; compare to standard APR baselines and zero-shot LLM repair.", "contribution": "Bug-pattern-grounded LLM code repair with structured retrieval.", "novelty_score": 7.0},
            {"title": "Type-Constrained Code Generation via Formal Type Inference Integration", "description": "Integrate a formal type checker into the generation loop; at each token step, mask out tokens that would produce a type error according to incremental type inference.", "rationale": "LLMs frequently generate type errors that would be caught by a compiler; constraining generation to type-correct outputs reduces these errors without post-hoc filtering.", "experiment": "TypeScript and Python type-annotated HumanEval; measure type error rate and pass@k.", "contribution": "Type-safe token-level constraint for LLM code generation.", "novelty_score": 8.5},
            {"title": "Program Synthesis via Hierarchical Task Decomposition", "description": "Decompose complex programming tasks into a hierarchy of subtasks using chain-of-thought, implement each leaf subtask independently, and compose the results.", "rationale": "Complex programs exceed LLM context and reasoning capacity; hierarchical decomposition brings complex tasks into range.", "experiment": "SWE-bench full; compare to zero-shot and ReAct baselines on issues requiring >200-line changes.", "contribution": "Hierarchical task decomposition framework for complex program synthesis.", "novelty_score": 7.5},
            {"title": "Test-Driven Development Loop with LLM as Both Coder and Tester", "description": "Have the LLM first write tests based on the specification, then generate code to pass those tests, with a refinement loop between the two roles.", "rationale": "Tests serve as a formal specification; separating test writing from code writing reduces the LLM's tendency to overfit to the prompt phrasing.", "experiment": "HumanEval and MBPP; compare to direct generation and ReAct on pass@k.", "contribution": "LLM-driven test-code separation loop for more robust program synthesis.", "novelty_score": 7.0},
            {"title": "Fault Localization via Causal Tracing in LLM Code Representations", "description": "Apply causal tracing to identify which specific LLM activations correspond to known faulty code patterns, then target repairs to the causally implicated components.", "rationale": "Black-box LLM repair is uninterpretable; causal tracing provides mechanistic understanding of where faults are represented.", "experiment": "Manually inject known bug types into correct code; measure localization accuracy and repair success.", "contribution": "Mechanistic LLM interpretability for fault localization and targeted code repair.", "novelty_score": 8.0},
        ],
    },
    # ── Climate ──────────────────────────────────────────────────────────────
    {
        "query": "deep learning for climate model downscaling and extreme weather prediction",
        "domain": "Climate",
        "ideas": [
            {"title": "Physics-Informed Neural Network for Statistical Downscaling", "description": "Incorporate known atmospheric physics constraints (e.g., conservation of mass, energy balance) as soft loss terms when training a neural downscaling model, ensuring outputs respect physical laws.", "rationale": "Unconstrained neural networks can produce physically implausible local extremes; physics-informed constraints improve reliability at tail events.", "experiment": "ERA5 → 1km downscaling over Europe; compare to BCSD and unconstrained CNN on CRPS and physical consistency metrics.", "contribution": "Physics-constrained statistical downscaling network with improved extreme value reliability.", "novelty_score": 8.0},
            {"title": "Diffusion Model for Probabilistic Weather Downscaling", "description": "Train a diffusion model to sample from the conditional distribution of high-resolution weather fields given low-resolution model output, producing calibrated uncertainty ensembles.", "rationale": "Deterministic downscaling cannot capture spatial variability of extremes; probabilistic outputs are more useful for impact models.", "experiment": "Compare ensemble spread calibration and CRPS to BCSD ensemble and post-processed NWP on precipitation over the Alps.", "contribution": "Calibrated probabilistic downscaling via conditional diffusion model.", "novelty_score": 8.5},
            {"title": "Graph Neural Network for Mesoscale Convective System Tracking", "description": "Represent thunderstorm cells as nodes in a spatio-temporal graph and use a GNN to predict cell merging, splitting, and intensification trajectories for 0-6 hour lead times.", "rationale": "Convective systems have complex nonlinear interactions that are poorly captured by grid-based approaches; graph representations capture cell topology naturally.", "experiment": "NEXRAD composite reflectivity; compare to optical flow and convolutional baselines on cell lifecycle prediction.", "contribution": "Graph-based mesoscale convective system tracking for sub-6h severe weather prediction.", "novelty_score": 7.5},
            {"title": "Transfer Learning from Global Climate Models to Regional Impact Assessment", "description": "Pre-train a transformer on global climate model output and fine-tune on regional observational records for impact-relevant variables (crop yield, flooding events).", "rationale": "Regional observational records are too short for direct training; global model pre-training provides physical priors that transfer to regional impact tasks.", "experiment": "Fine-tune on US county-level crop yield data; compare to direct training and CMIP6-based statistical downscaling.", "contribution": "Climate model pre-training for data-efficient regional impact assessment.", "novelty_score": 7.0},
            {"title": "Conditional Normalizing Flow for Multivariate Extreme Event Generation", "description": "Train a normalizing flow conditioned on large-scale atmospheric circulation patterns to generate physically consistent multivariate extreme events (concurrent heat and drought).", "rationale": "Compound extremes are driven by specific circulation patterns; conditioning on these patterns produces more realistic event scenarios.", "experiment": "ERA5 heatwave dataset; compare generated compound events to observed frequency and spatial coherence.", "contribution": "Circulation-conditioned compound extreme event generator for climate risk assessment.", "novelty_score": 8.0},
            {"title": "Temporal Convolutional Network for Sub-Seasonal Forecast Skill Enhancement", "description": "Train a TCN to identify persistent atmospheric patterns in extended-range NWP output and amplify their predictable signal while suppressing noise at 3-6 week lead times.", "rationale": "Sub-seasonal prediction skill is low due to chaos; persistent modes (MJO, blocking) carry predictable information that ML can extract from noisy NWP.", "experiment": "ECMWF ENS sub-seasonal forecasts; compare anomaly correlation of temperature at weeks 3-4 to raw NWP and climatological baselines.", "contribution": "TCN-based sub-seasonal forecast calibrator targeting predictable atmospheric modes.", "novelty_score": 6.5},
        ],
    },
    # ── Knowledge Graphs ─────────────────────────────────────────────────────
    {
        "query": "knowledge graph embedding methods for link prediction",
        "domain": "KR",
        "ideas": [
            {"title": "Temporal Knowledge Graph Embedding with Periodic Relation Dynamics", "description": "Model time-varying relations as periodic functions (Fourier basis) over entity embeddings, capturing seasonal and cyclical patterns in temporal knowledge graphs.", "rationale": "Linear time models miss periodic dynamics common in real-world KGs (e.g., annual events); periodic basis functions model these naturally.", "experiment": "ICEWS14, YAGO11k; compare to TComplEx, DE-SimplE on filtered MRR.", "contribution": "Periodic temporal KG embedding with improved modelling of cyclical relational dynamics.", "novelty_score": 7.5},
            {"title": "Hyperbolic Poincaré Embeddings for Hierarchical KG Reasoning", "description": "Embed entities in the Poincaré ball model of hyperbolic space, exploiting its tree-like metric structure to represent hierarchical ontologies with exponentially fewer dimensions.", "rationale": "Euclidean space requires exponentially many dimensions to represent hierarchies; hyperbolic space embeds them in low dimensions with small distortion.", "experiment": "FB15k-237 and YAGO3-10 (high hierarchy depth); compare to TransE, RotatE on MRR and hits@10.", "contribution": "Dimensionally efficient hierarchical KG embedding via Poincaré space.", "novelty_score": 7.0},
            {"title": "Relational Graph Convolutional Network with Edge-Type Decomposition", "description": "Decompose relation-specific weight matrices into a shared basis plus relation-specific coefficients, drastically reducing parameters while retaining expressiveness for multi-relational graphs.", "rationale": "Full relation-specific matrices scale quadratically with relation count; basis decomposition enables scalable multi-relational GNNs.", "experiment": "FB15k-237 and WN18RR; compare to R-GCN and CompGCN on link prediction MRR.", "contribution": "Parameter-efficient relational GCN via matrix basis decomposition.", "novelty_score": 7.0},
            {"title": "Reasoning over KG Paths via Differentiable Rule Learning", "description": "Learn first-order logic rules over KG paths differentiably (Neural LP / DRUM style) and use inferred rules to explain and improve link prediction.", "rationale": "Black-box KG embeddings are not interpretable; rule learning provides explicit logical explanations for predicted links.", "experiment": "FB15k-237, NELL-995; compare to Neural LP and RotatE on MRR and rule quality metrics.", "contribution": "Differentiable rule learning for interpretable and accurate KG link prediction.", "novelty_score": 7.5},
            {"title": "Multi-Hop Reasoning via Reinforcement Learning on KG Paths", "description": "Train a RL agent to walk multi-hop paths in the KG to answer queries, using the path itself as an explanation for the predicted answer entity.", "rationale": "Single-hop embedding models miss multi-hop reasoning chains; RL path-walking produces both accurate predictions and explicit reasoning paths.", "experiment": "WebQSP, MetaQA-3hop; compare to EmbedKGQA and TransE path baselines.", "contribution": "Explainable multi-hop KG reasoning via RL path walking with natural language path justifications.", "novelty_score": 8.0},
            {"title": "Zero-Shot KG Completion via Pre-trained Language Model Entity Descriptions", "description": "Represent unseen entities entirely via BERT embeddings of their textual descriptions, enabling link prediction for entities not present in training without any structural embedding.", "rationale": "Real-world KGs grow continuously; zero-shot generalisation to new entities is necessary for practical deployment.", "experiment": "FB15k-237 zero-shot split; compare to OWE and BLP on zero-shot MRR.", "contribution": "Zero-shot KG link prediction via text-only entity representations.", "novelty_score": 7.0},
        ],
    },
    # ── Adversarial Robustness ────────────────────────────────────────────────
    {
        "query": "certified adversarial robustness for deep neural networks",
        "domain": "Security/CV",
        "ideas": [
            {"title": "Randomized Smoothing with Input-Dependent Noise Calibration", "description": "Instead of fixed Gaussian noise in randomized smoothing, learn a noise scale function of the input that maximizes the certified radius while minimizing accuracy degradation on clean inputs.", "rationale": "Fixed noise hurts accuracy on all inputs equally; input-adaptive noise can preserve accuracy on easy inputs while providing better certification on harder ones.", "experiment": "ImageNet top-1 accuracy and certified radius at ε=0.5/1.0; compare to Cohen et al. and DSRS.", "contribution": "Input-adaptive noise calibration for randomized smoothing with improved accuracy-certification tradeoff.", "novelty_score": 8.0},
            {"title": "Interval Bound Propagation with Tighter Activation Bounds", "description": "Improve IBP verification by computing tighter activation bounds using the CROWN framework adaptively at each layer, reducing bound looseness without sacrificing verification speed.", "rationale": "Standard IBP produces loose bounds that force over-regularized networks; tighter bounds allow more expressive certified classifiers.", "experiment": "CIFAR-10 at ε=8/255; compare to IBP, CROWN-IBP, β-CROWN on verified accuracy.", "contribution": "Adaptive bound tightening for interval propagation with improved verified accuracy.", "novelty_score": 7.5},
            {"title": "Adversarial Training with Geometry-Aware Perturbation Budgets", "description": "Assign per-sample perturbation budgets during adversarial training based on the local geometric complexity of the decision boundary, giving larger budgets to samples near complex boundary regions.", "rationale": "Uniform perturbation budgets ignore boundary geometry; adaptive budgets focus training effort where the classifier is most vulnerable.", "experiment": "CIFAR-10 PGD-AT vs. geometry-aware AT; measure clean accuracy, PGD-20 robustness, and certified radius.", "contribution": "Geometry-adaptive adversarial training with improved clean-robust accuracy tradeoff.", "novelty_score": 7.5},
            {"title": "Data Augmentation via Certified Perturbation Interpolation", "description": "Generate augmented training samples by interpolating along certified perturbation directions between training examples, expanding the robust training manifold.", "rationale": "Standard data augmentation ignores adversarial structure; perturbation-aware augmentation exposes the model to the types of attacks it will face at test time.", "experiment": "STL-10 and CIFAR-100; compare to Cutout, AutoAugment, and adversarial training baselines.", "contribution": "Adversarially-structured data augmentation for improved generalisation under L-inf and L-2 attacks.", "novelty_score": 7.0},
            {"title": "Smooth Activation Functions Optimized for Lipschitz Bound Minimization", "description": "Design or select activation functions to minimize the Lipschitz constant of the network as a regularization objective, improving the tightness of Lipschitz-based certification.", "rationale": "Certification via Lipschitz bounds depends on activation smoothness; optimizing activations for small Lipschitz constants produces networks that are simultaneously accurate and easily certifiable.", "experiment": "Compare family of smooth activations on CIFAR-10 Lipschitz certification; trade clean accuracy vs. Lipschitz bound.", "contribution": "Lipschitz-optimized activation function selection for tighter verified robustness.", "novelty_score": 6.5},
            {"title": "Test-Time Denoising as a Pre-Processing Defense with Adaptive Adversary", "description": "Train a denoiser using adversarial examples as a pre-processing step, but evaluate it against an adaptive adversary that optimizes through the denoiser.", "rationale": "Most denoising defenses are evaluated with non-adaptive adversaries, leading to overestimation of robustness; adaptive evaluation is essential for honest comparison.", "experiment": "CIFAR-10 and ImageNet; compare denoiser defense with naive vs. adaptive PGD-100 adversary.", "contribution": "Honest adaptive-adversary evaluation framework for denoising-based robustness defenses.", "novelty_score": 6.0},
        ],
    },
    # ── Time Series ──────────────────────────────────────────────────────────
    {
        "query": "time series anomaly detection for industrial IoT sensor data",
        "domain": "Time Series",
        "ideas": [
            {"title": "Anomaly-Aware Transformer with Segment-Level Association Discrepancy", "description": "Train a transformer on normal IoT data and use the discrepancy between series-association and prior-association attention patterns as an anomaly score, exploiting the structural difference between normal and anomalous segments.", "rationale": "Anomalous segments break the normal association patterns in attention; the discrepancy is a robust unsupervised anomaly score.", "experiment": "MSL, SMAP, SMD benchmarks; compare to Anomaly Transformer, THOC on F1.", "contribution": "Attention-discrepancy anomaly score for transformer-based IoT time series detection.", "novelty_score": 7.5},
            {"title": "Multivariate Anomaly Detection via Graph-Structured Sensor Relations", "description": "Model sensor interdependencies as a dynamic graph and detect anomalies when sensor readings deviate from graph-predicted values, using GNN over the learned sensor topology.", "rationale": "Industrial sensors are physically coupled; ignoring sensor relationships causes false alarms from correlated fluctuations that are actually normal.", "experiment": "SWAT, WADI water treatment datasets; compare to MTAD-GAT, GDN on F1 and detection latency.", "contribution": "Graph-topology-aware multivariate anomaly detection for physically coupled sensor networks.", "novelty_score": 8.0},
            {"title": "Hierarchical Temporal Convolutional Autoencoder for Multi-Scale Anomalies", "description": "Build a TCN autoencoder with skip connections at multiple temporal scales, computing reconstruction error at each scale to detect both instantaneous spikes and gradual drifts.", "rationale": "Single-scale models miss gradual drift anomalies; multi-scale reconstruction captures anomalies across the full temporal hierarchy.", "experiment": "NASA SMAP + synthetic drift injection; compare to LSTM-AE, OmniAnomaly on both spike and drift detection F1.", "contribution": "Multi-scale reconstruction framework detecting both instantaneous and gradual anomalies in IoT streams.", "novelty_score": 7.5},
            {"title": "Self-Supervised Pre-Training on Unlabeled IoT Data with Contrastive Temporal Coding", "description": "Pre-train a time series encoder using contrastive loss between temporally close windows (positive pairs) and distant windows (negative pairs), then fine-tune with a small labeled anomaly set.", "rationale": "Labeled anomalies are rare in industrial settings; self-supervised pre-training on abundant unlabeled data should improve anomaly detection from few labels.", "experiment": "Few-shot fine-tuning on SWAT; measure F1 vs. number of labeled anomalies compared to supervised baselines.", "contribution": "Label-efficient anomaly detection via self-supervised temporal contrastive pre-training.", "novelty_score": 7.0},
            {"title": "Normalizing Flow for Density-Based Multivariate Anomaly Scoring", "description": "Model the joint distribution of multivariate sensor readings using a normalizing flow and score anomalies by the negative log-likelihood under the learned density.", "rationale": "Threshold-based and reconstruction-based detectors struggle with multivariate correlations; density-based scoring handles correlated normal modes naturally.", "experiment": "MSL, SMAP, SMD; compare NF-based scoring to THOC, DAGMM on both overall F1 and false alarm rate.", "contribution": "Exact density estimation for multivariate IoT anomaly scoring with calibrated anomaly probabilities.", "novelty_score": 7.0},
            {"title": "Concept Drift Detection as a Precursor to Anomaly Labelling", "description": "First detect concept drifts in the sensor stream using a statistical drift detector, then retrain the anomaly model on the post-drift normal distribution to prevent stale normal models.", "rationale": "Industrial processes evolve over time; static anomaly models become miscalibrated after process changes, causing increasing false alarms.", "experiment": "Synthetic concept-drift injection into SWAT; measure false alarm rate degradation over time with and without drift-adaptive retraining.", "contribution": "Drift-adaptive anomaly detection pipeline with automatic model refreshing on concept drift.", "novelty_score": 6.5},
        ],
    },
    # ── Table QA ─────────────────────────────────────────────────────────────
    {
        "query": "natural language question answering over structured tabular data",
        "domain": "NLP/DB",
        "ideas": [
            {"title": "Chain-of-Thought SQL Generation with Self-Consistency Voting", "description": "Generate multiple SQL queries via chain-of-thought prompting, execute each against the database, and use majority voting over the result sets to select the final answer.", "rationale": "SQL generation has multiple valid formulations; self-consistency over execution results reduces sensitivity to generation artifacts.", "experiment": "Spider and WikiTableQuestions; compare pass@1 with and without self-consistency on exact match accuracy.", "contribution": "Execution-grounded self-consistency for robust text-to-SQL with natural language questions.", "novelty_score": 7.0},
            {"title": "Hybrid Retrieval-Reasoning for Multi-Table Question Answering", "description": "First retrieve relevant tables and columns using a dense retrieval model, then perform reasoning over the retrieved subset using a fine-tuned table encoder.", "rationale": "Multi-table QA requires selecting among many candidate tables; retrieval reduces the search space before reasoning.", "experiment": "OTT-QA and HybridQA; compare to full-context approaches on EM and F1.", "contribution": "Retrieval-augmented multi-table QA with reduced reasoning context size.", "novelty_score": 7.5},
            {"title": "Schema-Linking via Contrastive Training on Column Names and Question Terms", "description": "Train a schema-linking module contrastively to align question mentions with column names and values, improving schema linking accuracy as a precursor to SQL generation.", "rationale": "Schema linking errors propagate to incorrect SQL; dedicated contrastive training on column-question pairs outperforms implicit linking in end-to-end models.", "experiment": "Spider schema-linking evaluation; ablate schema-linking accuracy on Text-to-SQL models.", "contribution": "Dedicated contrastive schema-linking module with improved downstream SQL accuracy.", "novelty_score": 7.0},
            {"title": "Decomposed Execution over Tabular Data for Multi-Step Reasoning", "description": "Decompose complex multi-step questions into a sequence of elementary table operations (filter, aggregate, sort) using an LLM planner and execute each step sequentially.", "rationale": "End-to-end SQL generation struggles with multi-hop reasoning; step decomposition makes each step tractable.", "experiment": "WikiTableQuestions complex-split and FeTaQA; compare to direct SQL, TAPAS, OmniTab.", "contribution": "Decomposed step-by-step tabular reasoning with intermediate result grounding.", "novelty_score": 7.5},
            {"title": "Table Serialization via Structure-Aware Pre-Training", "description": "Pre-train a table encoder on a structure-aware objective that predicts missing cell values and column types, then fine-tune for downstream NLU tasks over tables.", "rationale": "Generic pre-trained LMs are not trained on tabular structure; structure-aware pre-training improves table comprehension.", "experiment": "Fine-tune on WikiSQL, SQA, TabFact; compare to TAPAS and OmniTab on accuracy and efficiency.", "contribution": "Structure-aware tabular pre-training with improved cell-level understanding.", "novelty_score": 6.5},
            {"title": "Feedback-Driven SQL Repair via Error Message Grounding", "description": "When the generated SQL produces a database error, feed the error message back to the LLM with the original question and ask it to correct the specific error type.", "rationale": "LLMs produce SQL syntax and semantic errors that the database executor identifies; error-grounded feedback reduces the need for manual correction.", "experiment": "Spider; measure execution accuracy improvement from zero-shot to error-feedback loop over 3 rounds.", "contribution": "Error-grounded SQL repair loop reducing execution failures in text-to-SQL.", "novelty_score": 6.5},
        ],
    },
    # ── Low-Resource NMT ─────────────────────────────────────────────────────
    {
        "query": "neural machine translation for extremely low-resource language pairs",
        "domain": "NLP",
        "ideas": [
            {"title": "Pivot-Based Translation via Multilingual Encoder with Language Routing", "description": "Use a multilingual encoder shared across 50+ languages and train a language-routing gate that selects relevant encoder layers per language pair, enabling zero-shot transfer for unseen pairs.", "rationale": "Direct parallel data for many pairs is unavailable; multilingual sharing transfers knowledge from related high-resource pairs.", "experiment": "FLoRes-200 benchmark; evaluate on 10 unseen low-resource pairs from diverse families.", "contribution": "Language-routing multilingual NMT with zero-shot transfer to unseen low-resource pairs.", "novelty_score": 7.5},
            {"title": "Back-Translation with Quality Filtering via Dual Cross-Entropy", "description": "Generate back-translations from monolingual target data, filter by dual cross-entropy score (sentence-level fluency + translation adequacy), and use filtered pairs for augmentation.", "rationale": "Unfiltered back-translations introduce noise; quality filtering retains the most informative augmented pairs.", "experiment": "WMT Nepali-English and Sinhala-English (< 50K parallel sentences); compare filtered vs. unfiltered BT.", "contribution": "Quality-filtered back-translation augmentation for improved low-resource NMT.", "novelty_score": 7.0},
            {"title": "Cross-Lingual Transfer via Script Normalization and Phoneme Mapping", "description": "Normalize text across related scripts (Devanagari, Bengali, Gujarati) to a shared phoneme space and train a shared NMT model over the normalized representations.", "rationale": "Related South Asian languages share most phonemes but use different scripts; script normalization allows training data to be pooled across languages.", "experiment": "FLORES for 5 Indo-Aryan languages; compare to script-specific models and mBART on BLEU.", "contribution": "Script-normalized multilingual NMT for related low-resource South Asian language pairs.", "novelty_score": 8.0},
            {"title": "Adapter-Based Domain Adaptation for Low-Resource Specialized NMT", "description": "Fine-tune small language-pair-specific adapters on a domain-specific glossary and a handful of in-domain parallel sentences, without updating the shared multilingual backbone.", "rationale": "Domain-specific MT requires specialized vocabulary; adapters allow low-resource adaptation without losing multilingual generalization.", "experiment": "Medical and legal domain test sets for 5 low-resource pairs; compare to full fine-tuning and in-domain mBART.", "contribution": "Adapter-based domain specialization for low-resource NMT without catastrophic forgetting.", "novelty_score": 7.5},
            {"title": "Word Alignment-Guided Attention for Low-Resource NMT", "description": "Use fast unsupervised word alignment (GIZA++, awesome-align) to supervise cross-attention weights during training, reducing hallucination in low-resource settings.", "rationale": "Low-resource NMT frequently hallucinates; alignment supervision steers attention toward correct source tokens.", "experiment": "FLoRes low-resource pairs; measure BLEU and hallucination rate (via LaBSE similarity) before and after alignment supervision.", "contribution": "Alignment-supervised attention mechanism for reducing hallucination in low-resource NMT.", "novelty_score": 7.0},
            {"title": "Morphological Decomposition for Agglutinative Low-Resource Languages", "description": "Pre-process agglutinative source sentences (Turkish, Finnish, Swahili) using morphological segmentation and train the NMT model on morpheme-level tokens rather than BPE subwords.", "rationale": "BPE creates arbitrary subword splits for agglutinative languages; morpheme-level tokenization aligns with linguistic structure and reduces OOV rates.", "experiment": "Turkish-English, Finnish-English (low-resource splits); compare BPE to morphological tokenization on BLEU and OOV rate.", "contribution": "Morphology-aware tokenization for improved translation quality in agglutinative low-resource languages.", "novelty_score": 6.5},
        ],
    },
    # ── XAI / Medical ────────────────────────────────────────────────────────
    {
        "query": "explainable AI methods for clinical decision support systems",
        "domain": "XAI",
        "ideas": [
            {"title": "Concept-Based Explanations Aligned with Clinical Ontologies", "description": "Train a concept bottleneck model where concepts correspond to ICD-coded diagnoses or SNOMED-CT clinical findings, producing explanations in terms clinicians already use.", "rationale": "Pixel-level attribution maps are not clinically interpretable; concept-level explanations map directly to clinical knowledge.", "experiment": "MIMIC-III mortality prediction; compare concept explanation accuracy and clinician acceptance in user study.", "contribution": "Ontology-aligned concept bottleneck model for clinician-interpretable decision support.", "novelty_score": 8.0},
            {"title": "Counterfactual Explanations with Actionable Clinical Feature Changes", "description": "Generate counterfactual explanations (what would need to change for a different prediction) constrained to clinically actionable features (e.g., exclude age, include medication changes).", "rationale": "Unrestricted counterfactuals suggest impossible feature changes; clinical actionability constraints make explanations useful for treatment planning.", "experiment": "MIMIC-III sepsis prediction; measure clinical actionability and plausibility of counterfactuals vs. DICE and CEM.", "contribution": "Clinically actionable counterfactual explanations for treatment-guiding decision support.", "novelty_score": 8.5},
            {"title": "Faithful SHAP Approximations via Shapley Value Coalitional Game Sampling", "description": "Improve SHAP value estimation for clinical models by using coalition-aware sampling that oversamples underrepresented feature interactions, reducing variance on rare but clinically important features.", "rationale": "Standard SHAP sampling underestimates attribution of rare features; coalition-aware sampling improves faithfulness for low-prevalence clinical variables.", "experiment": "Compare SHAP faithfulness (ground truth from full computation on small models) vs. standard KernelSHAP on MIMIC tabular data.", "contribution": "Coalition-aware SHAP approximation with improved attribution faithfulness for rare clinical features.", "novelty_score": 7.0},
            {"title": "Temporal Saliency Maps for Sequential Clinical Event Explanations", "description": "Extend gradient-based saliency methods to temporal sequences of clinical events (EHR timelines), attributing predictions to specific past events and their timing.", "rationale": "Clinical decisions depend on sequences of events over time; static saliency maps miss temporal structure.", "experiment": "MIMIC-III ICU mortality; show temporal saliency against structured clinical event timeline; evaluate alignment with expert annotations.", "contribution": "Temporal extension of gradient saliency for event-sequence clinical prediction models.", "novelty_score": 7.5},
            {"title": "Prototype-Based Explanation via Case-Based Reasoning with EHR Embeddings", "description": "For each prediction, retrieve the k most similar historical patient cases and present them alongside the model prediction, enabling clinicians to reason by analogy.", "rationale": "Clinicians naturally reason from similar past cases; prototype-based explanations leverage this mental model directly.", "experiment": "MIMIC-III; user study measuring clinician trust, accuracy of override decisions, and time-on-task vs. attention/SHAP explanations.", "contribution": "Case-based retrieval explanation interface for clinical AI that aligns with analogical clinical reasoning.", "novelty_score": 7.5},
            {"title": "Uncertainty-Aware Explanations with Conformal Prediction Sets", "description": "Combine conformal prediction (valid coverage guarantees) with feature attribution to produce explanations that include guaranteed-coverage prediction sets and uncertainty-weighted attributions.", "rationale": "Point prediction explanations overstate certainty; conformal prediction provides coverage guarantees that reduce over-reliance on AI recommendations.", "experiment": "MIMIC-IV sepsis prediction; measure coverage, prediction set size, and clinician calibration in user study.", "contribution": "Conformal-prediction-grounded explanations with valid statistical coverage guarantees for clinical AI.", "novelty_score": 8.0},
        ],
    },
    # ── Robotics ─────────────────────────────────────────────────────────────
    {
        "query": "reinforcement learning for dexterous robotic manipulation",
        "domain": "Robotics",
        "ideas": [
            {"title": "Curriculum Learning for Contact-Rich Manipulation via Automatic Task Staging", "description": "Automatically stage manipulation tasks from easy (large-tolerance grasp) to hard (precise peg-in-hole) by measuring agent success rate and advancing the stage when it crosses a threshold.", "rationale": "Contact-rich tasks have extremely sparse rewards; automatic curriculum staging provides a smooth learning gradient without manual task design.", "experiment": "Shadow Hand dexterous manipulation benchmark; compare to flat RL, manual curriculum, and HER.", "contribution": "Automatic difficulty-staged curriculum for contact-rich dexterous manipulation.", "novelty_score": 7.5},
            {"title": "Tactile Feedback Integration for Slip Detection and Grasp Adjustment", "description": "Integrate tactile sensor readings as a separate observation stream and train the policy to detect incipient slip and adjust grip force reactively within the same episode.", "rationale": "Vision-only policies fail at grasp adjustment because slip is not visually observable until after the object has moved; tactile feedback enables reactive control.", "experiment": "Physical Allegro hand with BioTac sensors; measure grasp success rate on slippery objects vs. vision-only policy.", "contribution": "Tactile-visual policy with reactive slip detection for robust in-hand manipulation.", "novelty_score": 8.0},
            {"title": "Sim-to-Real Transfer via Domain Randomization with Adaptive Distribution", "description": "Start with a narrow simulation randomization distribution and adaptively widen it based on which parameters the policy finds most challenging, as measured by performance variance.", "rationale": "Uniform domain randomization wastes capacity on easy variations; adaptive randomization focuses training on the dimensions that matter most for transfer.", "experiment": "Franka Panda reaching and pick-and-place; measure sim-to-real transfer gap vs. uniform randomization.", "contribution": "Adaptive domain randomization distribution that concentrates on high-variance parameters for efficient sim-to-real transfer.", "novelty_score": 7.5},
            {"title": "Hierarchical RL with Reusable Primitive Skills for Dexterous Tasks", "description": "Learn a library of low-level manipulation primitives (pinch, roll, push) using RL and then train a high-level policy that sequences these primitives for complex assembly tasks.", "rationale": "Monolithic policies for dexterous assembly are hard to train from scratch; reusable primitives reduce the search space for the high-level planner.", "experiment": "IKEA furniture assembly task; compare primitive-based hierarchy to flat PPO and SAC.", "contribution": "Primitive skill library with compositional high-level RL for complex dexterous assembly.", "novelty_score": 7.0},
            {"title": "Demonstration-Augmented RL via Residual Policy Learning", "description": "Initialize from human demonstrations using behavior cloning, then learn a residual correction policy using RL that adds corrections to the BC baseline action.", "rationale": "Pure RL from scratch is sample-inefficient; BC provides a good warm-start but is limited to demonstrated behaviors; residual RL extends beyond demonstrations.", "experiment": "DAPG dexterous hand tasks; compare to BC, DAPG, and pure SAC on success rate and sample efficiency.", "contribution": "Residual correction policy for demonstration-guided RL with improved sample efficiency.", "novelty_score": 7.0},
            {"title": "Model-Based RL with Learned Differentiable Contact Models", "description": "Learn a differentiable contact model from experience and use it within a model-based RL framework to plan contact-rich trajectories via gradient-based trajectory optimization.", "rationale": "Model-free RL is sample-inefficient for contact-rich tasks; differentiable contact models enable gradient-based planning.", "experiment": "Peg-in-hole and valve-turning tasks; compare sample efficiency to SAC, TD-MPC, and Dreamer.", "contribution": "Differentiable contact model for gradient-based model-based RL in contact-rich manipulation.", "novelty_score": 8.5},
        ],
    },
    # ── Protein Structure ─────────────────────────────────────────────────────
    {
        "query": "protein structure prediction and design using deep learning",
        "domain": "Bioinformatics",
        "ideas": [
            {"title": "Diffusion Model for Protein Backbone Generation with Sequence Co-Design", "description": "Train a SE(3)-equivariant diffusion model that jointly denoises backbone coordinates and residue identities, producing novel proteins where structure and sequence are co-optimized.", "rationale": "Separate structure prediction and sequence design miss co-dependencies; joint diffusion captures the joint distribution.", "experiment": "Designability (self-consistency RMSD) and novelty vs. RFDiffusion and ProteinMPNN on de novo protein design.", "contribution": "Joint backbone-sequence diffusion model for designable and novel protein generation.", "novelty_score": 8.5},
            {"title": "Multi-State Protein Structure Prediction for Conformational Ensembles", "description": "Extend AlphaFold2-style prediction to produce an ensemble of low-energy conformational states rather than a single structure, by modifying the structure module to optimize for ensemble diversity.", "rationale": "Proteins are dynamic; single-structure prediction misses conformational changes important for function and drug design.", "experiment": "Benchmark against NMR ensembles and MD simulation on 100-protein test set; compare RMSD distribution.", "contribution": "Multi-conformation protein structure predictor capturing functionally relevant states.", "novelty_score": 8.0},
            {"title": "Language Model Pre-Training on Evolutionary Sequence Data for Functional Annotation", "description": "Pre-train a protein language model on evolutionary sequence alignments (MSAs) rather than individual sequences and fine-tune for enzyme function prediction and stability annotation.", "rationale": "MSA-aware pre-training captures evolutionary constraints; individual-sequence models ignore the evolutionary record.", "experiment": "FLIP benchmark for fitness prediction; compare to ESM-2 and MSA Transformer on held-out proteins.", "contribution": "MSA-aware protein language model with improved functional annotation accuracy.", "novelty_score": 7.5},
            {"title": "Zero-Shot Mutation Effect Prediction via Energy-Based Language Models", "description": "Use the masked language model probability change upon mutation as a zero-shot proxy for mutational effect on protein stability and function without task-specific fine-tuning.", "rationale": "Labeled mutation data is scarce; zero-shot LM scoring is surprisingly predictive and requires no training.", "experiment": "ProteinGym DMS benchmark; compare to EVE, ESM-1v, and ESM-2 log-likelihood across 50+ assays.", "contribution": "Zero-shot mutation effect predictor via energy-based protein LM scoring without fine-tuning.", "novelty_score": 7.0},
            {"title": "Structure-Conditioned Antibody CDR Design via Masked Diffusion", "description": "Design antibody CDR loops conditioned on the antigen structure using masked diffusion over discrete residue identities, enabling target-specific antibody optimization.", "rationale": "Antibody effectiveness depends on CDR complementarity to the antigen surface; structure-conditioned design should produce higher-affinity variants.", "experiment": "RAbD antibody design benchmark; compare designed CDR sequences to native and Rosetta-designed sequences on affinity and binding geometry.", "contribution": "Antigen-conditioned antibody CDR design via masked diffusion with improved binding complementarity.", "novelty_score": 8.0},
            {"title": "Protein Function Prediction via Hierarchical GO Term Classification", "description": "Train a multi-label classifier over the Gene Ontology hierarchy using graph neural propagation, enforcing is-a relationships between GO terms during training.", "rationale": "GO annotations form a DAG; ignoring the hierarchy causes inconsistent predictions (e.g., predicting a specific term without its parent).", "experiment": "CAFA-5 benchmark; compare to DeepFRI, ProtCNN on F-max and hierarchical consistency.", "contribution": "Hierarchy-consistent multi-label protein function predictor via GO-graph-aware propagation.", "novelty_score": 7.0},
        ],
    },
    # ── Anomaly Detection (NLP) ───────────────────────────────────────────────
    {
        "query": "out-of-distribution detection for deployed machine learning models",
        "domain": "ML Safety",
        "ideas": [
            {"title": "Energy-Based OOD Detection with Temperature-Scaled Logits", "description": "Use the free energy of the softmax distribution (negative log-sum-exp of logits) as an OOD score, with temperature scaling calibrated on a held-out validation set.", "rationale": "Maximum softmax probability is overconfident for OOD inputs; energy-based scoring exploits the full logit distribution and is less sensitive to overconfident predictions.", "experiment": "CIFAR-10/100 ID with SVHN/iSUN OOD; compare to MSP, ODIN, Mahalanobis on AUROC.", "contribution": "Calibrated energy-based OOD score with improved separation from in-distribution confidence.", "novelty_score": 7.0},
            {"title": "Feature-Space Density Estimation for OOD Detection via Normalizing Flows", "description": "Fit a normalizing flow on feature representations from the penultimate layer of a trained classifier and use the negative log-likelihood under the flow as an OOD score.", "rationale": "Softmax-based scores ignore feature-space structure; density estimation in feature space captures the full geometry of the in-distribution manifold.", "experiment": "OpenOOD benchmark; compare to KNN, Mahalanobis, and energy-based detectors on AUROC.", "contribution": "Feature-space density estimator for OOD detection with exact likelihood scoring.", "novelty_score": 7.5},
            {"title": "Semantic Shift Detection via CLIP Embedding Distribution Monitoring", "description": "Monitor the distribution of CLIP embeddings of incoming test batches and detect OOD by measuring Wasserstein distance to the training embedding distribution.", "rationale": "CLIP provides a semantically rich embedding space; distribution-level monitoring detects gradual semantic shift that single-sample scores miss.", "experiment": "Wilds benchmarks (iWILDCam, FMoW); measure batch-level OOD detection AUROC at varying shift severity.", "contribution": "Batch-level semantic distribution monitoring for gradual OOD shift detection in deployed models.", "novelty_score": 8.0},
            {"title": "Test-Time Augmentation Consistency as an OOD Signal", "description": "Measure prediction consistency across a set of augmented versions of the test input; in-distribution inputs are expected to produce stable predictions while OOD inputs show high variance.", "rationale": "In-distribution inputs lie near the learned decision boundary and are stable under augmentation; OOD inputs that fall off the manifold are sensitive to augmentation.", "experiment": "CIFAR-10 vs. Textures; measure prediction variance across 50 augmented copies vs. energy-based and Mahalanobis baselines.", "contribution": "Augmentation-consistency OOD score with no additional training or feature extraction.", "novelty_score": 7.0},
            {"title": "Contrastive Training Auxiliary to Improve Feature Separability for OOD", "description": "Add a contrastive auxiliary loss during training that pushes features of different classes further apart, making the resulting feature space more amenable to Mahalanobis and KNN OOD detection.", "rationale": "OOD detection methods that rely on feature-space distances work better when in-distribution features are well-separated; contrastive training directly optimizes for this.", "experiment": "OpenOOD; measure Mahalanobis AUROC with and without contrastive auxiliary on ResNet-50.", "contribution": "Contrastive feature shaping for improved downstream OOD detection without changing the detection method.", "novelty_score": 7.5},
            {"title": "Conformal Prediction Sets for Distribution-Free OOD Quantification", "description": "Use conformal prediction to produce prediction sets with valid coverage guarantees and flag inputs with large prediction sets as OOD, providing a distribution-free uncertainty quantification.", "rationale": "Conformal prediction is model-agnostic and provides provable coverage; prediction set size is a principled proxy for in-distribution confidence.", "experiment": "CIFAR-10 and CIFAR-100 with Tiny ImageNet OOD; measure AUROC and coverage guarantee validity.", "contribution": "Distribution-free OOD detection via conformal prediction set size with valid coverage guarantees.", "novelty_score": 7.0},
        ],
    },
]

# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """You are an independent scientific research evaluator.
You did NOT generate the ideas you are judging.
Compare two research ideas on four criteria and pick the better one."""

_JUDGE_PROMPT = """Research topic: {query}

=== IDEA A ===
Title: {title_a}
Description: {description_a}
Experiment: {experiment_a}

=== IDEA B ===
Title: {title_b}
Description: {description_b}
Experiment: {experiment_b}

Rate each idea 1-5 on: novelty, feasibility, impact, specificity.
Pick the winner overall.

Return ONLY JSON:
{{"winner": "A"|"B"|"tie",
  "scores": {{"A": {{"novelty":1-5,"feasibility":1-5,"impact":1-5,"specificity":1-5}},
              "B": {{"novelty":1-5,"feasibility":1-5,"impact":1-5,"specificity":1-5}}}},
  "rationale": "<2 sentences>"}}"""


def run_judge(query: str, idea_a: dict, idea_b: dict) -> dict:
    prompt = _JUDGE_PROMPT.format(
        query=query,
        title_a=idea_a.get("title", ""),
        description_a=idea_a.get("description", ""),
        experiment_a=idea_a.get("experiment", ""),
        title_b=idea_b.get("title", ""),
        description_b=idea_b.get("description", ""),
        experiment_b=idea_b.get("experiment", ""),
    )
    try:
        msg = ModelRegistry.completion(
            "critic",
            [Message(role="user", content=prompt)],
            system_prompt=_JUDGE_SYSTEM,
            agent_sender="eval",
        )
        return parse_json_from_llm_response(msg.content or "")
    except Exception as e:
        logger.warning("Judge call failed: {}", e)
        return {"winner": "tie", "rationale": f"Judge error: {e}", "scores": {}}


# ---------------------------------------------------------------------------
# Mean pairwise similarity (TF-IDF cosine, no sklearn)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def mean_pairwise_similarity(ideas: list[dict]) -> float:
    """Mean pairwise TF-IDF cosine similarity of idea descriptions. Lower = more diverse."""
    texts = [f"{i.get('title','')} {i.get('description','')}" for i in ideas]
    n = len(texts)
    if n < 2:
        return 0.0
    tokenized = [_tokenize(t) for t in texts]
    vocab = sorted({tok for doc in tokenized for tok in doc})
    if not vocab:
        return 0.0
    vi = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    tf = [[0.0] * V for _ in range(n)]
    for d, doc in enumerate(tokenized):
        cnt = Counter(doc)
        total = max(len(doc), 1)
        for w, c in cnt.items():
            if w in vi:
                tf[d][vi[w]] = c / total

    df = [0] * V
    for doc in tokenized:
        for w in set(doc):
            if w in vi:
                df[vi[w]] += 1
    idf = [math.log((n + 1) / (df[i] + 1)) + 1.0 for i in range(V)]
    tfidf = [[tf[d][i] * idf[i] for i in range(V)] for d in range(n)]

    def dot(a: list, b: list) -> float: return sum(x * y for x, y in zip(a, b))
    def norm(a: list) -> float: return math.sqrt(sum(x * x for x in a))

    norms = [norm(r) for r in tfidf]
    total = sum(
        dot(tfidf[i], tfidf[j]) / (norms[i] * norms[j])
        for i in range(n) for j in range(n)
        if i != j and norms[i] > 0 and norms[j] > 0
    )
    return total / (n * (n - 1))


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------

@dataclass
class IdeaInfo:
    title: str
    composite_score: float
    lift: float          # relative to initial_best_composite
    operator: str        # "seed" | "improve" | "combine"


@dataclass
class JudgePair:
    label_a: str
    label_b: str
    winner: str          # "A" (=label_a wins), "B" (=label_b wins), "tie"
    rationale: str
    scores: dict


@dataclass
class QueryResult:
    query: str
    domain: str
    n_seeds: int

    # Four configurations
    baseline_novelty: IdeaInfo       # top seed by novelty_score
    baseline_composite: IdeaInfo     # top seed by 4D composite (score-only, 0 iterations)
    improve_only: IdeaInfo           # search with improve_fraction=1.0
    improve_combine: IdeaInfo        # search with improve_fraction=0.75 (default)

    initial_best_composite: float    # best composite in seed population (n=6 context)
    # Best surviving seed composite in each search run's final n=8 scoring pass.
    # This is the level-playing-field baseline for lift: same population, same n.
    # Falls back to initial_best_composite when all seeds are replaced by operators.
    seed_baseline_improve: float
    seed_baseline_full: float

    # Three pairwise judge comparisons
    judge_novelty_vs_full: JudgePair       # baseline_novelty vs improve_combine
    judge_composite_vs_full: JudgePair     # baseline_composite vs improve_combine
    judge_improve_vs_combine: JudgePair    # improve_only vs improve_combine

    # Final population metrics (from improve_combine run)
    operator_distribution: dict[str, int]
    mean_pairwise_sim: float

    llm_calls_total: int


def _top_by_novelty(ideas: list[dict]) -> dict:
    scored = [(float(i.get("novelty_score") or 0.0), i) for i in ideas if isinstance(i, dict)]
    scored.sort(key=lambda p: p[0], reverse=True)
    return scored[0][1] if scored else ideas[0]


def evaluate_query(
    query_data: dict,
    score_workers: int = 4,
) -> QueryResult:
    query = query_data["query"]
    domain = query_data["domain"]
    seeds = query_data["ideas"]
    total_calls = 0

    logger.info("=" * 60)
    logger.info("Query: {} ({})", query, domain)

    # Score seeds once — shared across all ablations so initial_best is consistent
    # and we don't pay 4 calls × 3 runs for the same seed population.
    logger.info("Scoring {} seed ideas (shared initial pass) …", len(seeds))
    seed_nodes = [IdeaNode(idea=idea, operator="seed", generation=0) for idea in seeds]
    score_population(seed_nodes, query, max_workers=score_workers)
    total_calls += 4
    initial_best = max(n.composite_score or 0.0 for n in seed_nodes)

    # baseline_novelty: idea chosen by novelty_score, looked up in scored seeds
    bn_title = _top_by_novelty(seeds).get("title", "")
    bn_node = next((n for n in seed_nodes if n.idea.get("title") == bn_title), seed_nodes[0])
    bn_idea = {
        **bn_node.idea,
        "composite_score": bn_node.composite_score,
        "score_novelty": bn_node.score_novelty,
        "score_feasibility": bn_node.score_feasibility,
        "score_impact": bn_node.score_impact,
        "score_specificity": bn_node.score_specificity,
        "search_operator": "seed",
    }

    # baseline_composite: top seed by composite — no search iterations
    # Pass pre-scored nodes so run_idea_search skips re-scoring and the final
    # pass is also skipped (any_new_nodes stays False).
    logger.info("Running: score-only baseline …")
    r_composite = run_idea_search(
        seeds, query, max_iterations=0, score_workers=score_workers,
        _pre_scored_nodes=copy.deepcopy(seed_nodes),
    )
    total_calls += r_composite.llm_calls_used  # 0 calls now (no operators, no final pass)
    bc_idea = r_composite.best_ideas[0] if r_composite.best_ideas else bn_idea

    # improve_only
    logger.info("Running: improve-only search …")
    r_improve = run_idea_search(
        seeds, query, improve_fraction=1.0, score_workers=score_workers,
        _pre_scored_nodes=copy.deepcopy(seed_nodes),
    )
    total_calls += r_improve.llm_calls_used
    imp_idea = r_improve.best_ideas[0] if r_improve.best_ideas else bc_idea

    # improve_combine (default, full system)
    logger.info("Running: improve+combine search …")
    r_full = run_idea_search(
        seeds, query, score_workers=score_workers,
        _pre_scored_nodes=copy.deepcopy(seed_nodes),
    )
    total_calls += r_full.llm_calls_used
    full_idea = r_full.best_ideas[0] if r_full.best_ideas else bc_idea

    def _best_seed_final(best_ideas: list[dict]) -> float:
        """Best composite among seed-operator ideas in the final population.

        Falls back to initial_best when all seeds were replaced by operators
        (common after 3 iterations — seeds often score below the new ideas in n=8 context).
        """
        seed_composites = [
            i.get("composite_score") or 0.0
            for i in best_ideas
            if i.get("search_operator") == "seed"
        ]
        return max(seed_composites) if seed_composites else initial_best

    seed_final_improve = _best_seed_final(r_improve.best_ideas)
    seed_final_full    = _best_seed_final(r_full.best_ideas)

    def _lift(composite: float, baseline: float) -> float:
        return (composite - baseline) / baseline if baseline > 0 else 0.0

    def _info(idea: dict, op_fallback: str = "seed", baseline: float | None = None) -> IdeaInfo:
        b = initial_best if baseline is None else baseline
        return IdeaInfo(
            title=idea.get("title", ""),
            composite_score=idea.get("composite_score") or 0.0,
            lift=_lift(idea.get("composite_score") or 0.0, b),
            operator=idea.get("search_operator", op_fallback),
        )

    # Judge comparisons
    logger.info("Running 3 judge comparisons …")
    j1 = run_judge(query, bn_idea, full_idea)
    total_calls += 1
    j2 = run_judge(query, bc_idea, full_idea)
    total_calls += 1
    j3 = run_judge(query, imp_idea, full_idea)
    total_calls += 1

    def _pair(label_a: str, label_b: str, j: dict) -> JudgePair:
        return JudgePair(
            label_a=label_a, label_b=label_b,
            winner=j.get("winner", "tie"),
            rationale=j.get("rationale", ""),
            scores=j.get("scores", {}),
        )

    # Operator distribution from full run
    ops: dict[str, int] = {}
    for idea in r_full.best_ideas:
        op = idea.get("search_operator", "seed")
        ops[op] = ops.get(op, 0) + 1

    sim = mean_pairwise_similarity(r_full.best_ideas)
    logger.info(
        "Done. Calls: {}  |  composite: {:.3f} → {:.3f}  |  sim: {:.3f}",
        total_calls, initial_best, full_idea.get("composite_score") or 0.0, sim,
    )

    return QueryResult(
        query=query, domain=domain, n_seeds=len(seeds),
        # baseline runs use initial_best (n=6 context — comparing seed selection strategies)
        baseline_novelty=_info(bn_idea),
        baseline_composite=_info(bc_idea),
        # search runs use each run's own final-pass seed baseline (n=8 context — level playing field)
        improve_only=_info(imp_idea, baseline=seed_final_improve),
        improve_combine=_info(full_idea, baseline=seed_final_full),
        initial_best_composite=initial_best,
        seed_baseline_improve=seed_final_improve,
        seed_baseline_full=seed_final_full,
        judge_novelty_vs_full=_pair("baseline_novelty", "improve_combine", j1),
        judge_composite_vs_full=_pair("baseline_composite", "improve_combine", j2),
        judge_improve_vs_combine=_pair("improve_only", "improve_combine", j3),
        operator_distribution=ops,
        mean_pairwise_sim=sim,
        llm_calls_total=total_calls,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _win_rate_stats(results: list[QueryResult], pair_attr: str, b_label: str) -> tuple[int, int, int, float, float]:
    """Return (b_wins, a_wins, ties, win_rate, p_one_tailed) for a judge pair."""
    b_wins = sum(1 for r in results if getattr(r, pair_attr).winner == "B")
    a_wins = sum(1 for r in results if getattr(r, pair_attr).winner == "A")
    ties   = sum(1 for r in results if getattr(r, pair_attr).winner == "tie")
    n = len(results)
    win_rate = b_wins / n if n else 0.0
    z = (b_wins - 0.5 * n) / math.sqrt(0.25 * n) if n >= 4 else float("nan")
    p = 0.5 * math.erfc(z / math.sqrt(2)) if not math.isnan(z) else float("nan")
    return b_wins, a_wins, ties, win_rate, p


def print_report(results: list[QueryResult]) -> None:
    SEP = "=" * 76

    print(f"\n{SEP}")
    print("IDEA SEARCH EVALUATION REPORT")
    print(f"{SEP}\n")

    # Per-query table
    hdr = f"{'Query':<38} {'Dom':<10} {'Lift_full':>9} {'Sim':>6} {'J1':>4} {'J2':>4} {'J3':>4}"
    print(hdr)
    print("-" * 76)
    for r in results:
        q = r.query[:36]
        lift = f"{r.improve_combine.lift:+.1%}"
        sim  = f"{r.mean_pairwise_sim:.3f}"
        j1   = r.judge_novelty_vs_full.winner
        j2   = r.judge_composite_vs_full.winner
        j3   = r.judge_improve_vs_combine.winner
        print(f"{q:<38} {r.domain:<10} {lift:>9} {sim:>6} {j1:>4} {j2:>4} {j3:>4}")
    print()
    print("J1 = baseline_novelty vs improve_combine   (A wins = baseline better)")
    print("J2 = baseline_composite vs improve_combine (A wins = scoring alone is enough)")
    print("J3 = improve_only vs improve_combine       (A wins = combine adds no value)")
    print()

    n = len(results)
    # Aggregate metrics
    avg_lift_bn  = sum(r.baseline_composite.lift for r in results) / n
    avg_lift_imp = sum(r.improve_only.lift for r in results) / n
    avg_lift_ful = sum(r.improve_combine.lift for r in results) / n
    avg_sim = sum(r.mean_pairwise_sim for r in results) / n
    avg_calls = sum(r.llm_calls_total for r in results) / n

    all_ops: dict[str, int] = {}
    for r in results:
        for op, cnt in r.operator_distribution.items():
            all_ops[op] = all_ops.get(op, 0) + cnt

    print(f"{SEP}")
    print("AGGREGATE METRICS")
    print(f"{SEP}")
    print(f"  Queries evaluated         : {n}")
    print()
    print("  Average composite score lift over initial best seed:")
    print(f"    score-only (baseline)   : {avg_lift_bn:+.1%}")
    print(f"    improve-only            : {avg_lift_imp:+.1%}")
    print(f"    improve+combine (full)  : {avg_lift_ful:+.1%}")
    print()
    print(f"  Mean pairwise similarity  : {avg_sim:.3f}  (>0.7 = population collapse)")
    print(f"  Avg LLM calls / query     : {avg_calls:.0f}")
    print()
    print("  Final population operator distribution (improve+combine runs):")
    op_total = sum(all_ops.values())
    for op in ("seed", "improve", "combine"):
        cnt = all_ops.get(op, 0)
        print(f"    {op:<10}: {cnt:>4}  ({cnt/op_total:.0%})" if op_total else f"    {op}: 0")
    print()

    # Judge win rates
    print("  Judge win-rates (B = right-side configuration wins):")
    for attr, label_a, label_b in [
        ("judge_novelty_vs_full",    "baseline_novelty",   "improve_combine"),
        ("judge_composite_vs_full",  "baseline_composite", "improve_combine"),
        ("judge_improve_vs_combine", "improve_only",       "improve_combine"),
    ]:
        bw, aw, ties, wr, p = _win_rate_stats(results, attr, label_b)
        sig = "* p<0.05" if p < 0.05 else ("~ p<0.10" if p < 0.10 else "")
        print(f"    {label_b:>18} vs {label_a:<22}: {bw}/{n} ({wr:.0%})  p≈{p:.3f}  {sig}")
    print()

    print(f"{SEP}")
    print("PER-QUERY DETAILS")
    print(f"{SEP}")
    for r in results:
        print(f"\n[{r.domain}] {r.query}")
        seed_bl_note = "" if r.seed_baseline_improve == r.initial_best_composite else f"  (final-pass seed: improve={r.seed_baseline_improve:.3f}, full={r.seed_baseline_full:.3f})"
        print(f"  initial best (seed, n=6): composite={r.initial_best_composite:.3f}{seed_bl_note}")
        configs = [
            ("baseline_novelty",   r.baseline_novelty),
            ("baseline_composite", r.baseline_composite),
            ("improve_only",       r.improve_only),
            ("improve_combine",    r.improve_combine),
        ]
        for name, info in configs:
            print(f"  {name:<22}: composite={info.composite_score:.3f}  lift={info.lift:+.1%}  [{info.operator}]  {info.title[:50]}")
        for attr, la, lb in [
            ("judge_novelty_vs_full",    "novelty_baseline", "improve_combine"),
            ("judge_composite_vs_full",  "composite_base",   "improve_combine"),
            ("judge_improve_vs_combine", "improve_only",     "improve_combine"),
        ]:
            jp = getattr(r, attr)
            print(f"  judge {la} vs {lb}: {jp.winner} — {jp.rationale[:100]}")

    print(f"\n{SEP}")
    print("NOTES")
    print("  Lift (baseline_novelty / baseline_composite):")
    print("    = (composite - initial_best) / initial_best")
    print("    Both measured in n=6 seed context (comparing seed selection strategies).")
    print("  Lift (improve_only / improve_combine):")
    print("    = (composite - seed_baseline) / seed_baseline")
    print("    seed_baseline = best surviving seed's composite in the final n=8 scoring pass.")
    print("    Falls back to initial_best when all seeds are replaced by operators (common after")
    print("    3 iterations). In that case the n=6 vs n=8 scale difference adds ~5-10% noise.")
    print("  Judge = 'critic' model (independent from 'ideation' used in search).")
    print("  Mean pairwise similarity: TF-IDF cosine on titles+descriptions — lower = more diverse.")
    print(f"{SEP}\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate idea search ablations on fixed seed idea pools",
        prog="python -m evals.eval_idea_search",
    )
    parser.add_argument("--queries", nargs="*", type=int,
                        help="Indices of queries to run (0-19). Default: all 20.")
    parser.add_argument("--sequential", action="store_true",
                        help="Score sequentially (1 thread). Use on rate-limited free-tier APIs.")
    parser.add_argument("--call-delay", type=float, default=0.0, metavar="SECONDS",
                        help="Seconds between sequential scoring calls. Recommended: 4.0 for Gemini free tier.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional path to write JSON results.")
    args = parser.parse_args()

    try:
        registered = register_defaults_from_yaml()
        logger.info("Registered models: {}", sorted(registered))
    except Exception as e:
        logger.error("Failed to register models: {}", e)
        return 1

    for role in ("ideation", "critic"):
        try:
            ModelRegistry.instance().get_model_params(role)
        except ValueError:
            logger.error("Role '{}' not registered. Check role_defaults.yaml and API keys.", role)
            return 1

    score_workers = 1 if args.sequential else 4
    if score_workers == 1:
        logger.info("Sequential scoring mode.")
    if args.call_delay > 0:
        import scider.agents.ideation_agent.idea_search as _ism
        _ism._SEQUENTIAL_CALL_DELAY_S = args.call_delay
        logger.info("Inter-call delay: {}s", args.call_delay)

    indices = args.queries if args.queries else list(range(len(QUERIES)))
    to_run = [QUERIES[i] for i in indices if 0 <= i < len(QUERIES)]
    logger.info("Running {} queries.", len(to_run))

    results: list[QueryResult] = []
    for q in to_run:
        try:
            results.append(evaluate_query(q, score_workers=score_workers))
        except Exception as e:
            logger.exception("Query '{}' failed: {}", q["query"], e)

    if not results:
        logger.error("No results collected.")
        return 1

    print_report(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump([
                {
                    "query": r.query, "domain": r.domain,
                    "initial_best_composite": r.initial_best_composite,
                    "seed_baseline_improve": r.seed_baseline_improve,
                    "seed_baseline_full": r.seed_baseline_full,
                    "baseline_novelty":   {"title": r.baseline_novelty.title,   "composite": r.baseline_novelty.composite_score,   "lift": r.baseline_novelty.lift},
                    "baseline_composite": {"title": r.baseline_composite.title, "composite": r.baseline_composite.composite_score, "lift": r.baseline_composite.lift},
                    "improve_only":       {"title": r.improve_only.title,       "composite": r.improve_only.composite_score,       "lift": r.improve_only.lift},
                    "improve_combine":    {"title": r.improve_combine.title,    "composite": r.improve_combine.composite_score,    "lift": r.improve_combine.lift},
                    "judge_novelty_vs_full":    {"winner": r.judge_novelty_vs_full.winner,    "rationale": r.judge_novelty_vs_full.rationale},
                    "judge_composite_vs_full":  {"winner": r.judge_composite_vs_full.winner,  "rationale": r.judge_composite_vs_full.rationale},
                    "judge_improve_vs_combine": {"winner": r.judge_improve_vs_combine.winner, "rationale": r.judge_improve_vs_combine.rationale},
                    "operator_distribution": r.operator_distribution,
                    "mean_pairwise_similarity": r.mean_pairwise_sim,
                    "llm_calls_total": r.llm_calls_total,
                }
                for r in results
            ], f, indent=2)
        logger.info("Results written to {}", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
