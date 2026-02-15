# Benchmark for AI Idea Bench

We assume all commands are run from the same dir as this README file.

First, create a new uv env:

```bash
uv init --python 3.12
uv venv --seed
source .venv/bin/activate
```


Then, install the dependencies:

```bash
pip install -r ../../requirements.txt
pip install -r requirements.txt
```

Download the dataset from [Huggingface](https://huggingface.co/datasets/yanshengqiu/AI_Idea_Bench_2025) and place `papers_data/` and `target_paper_data.json` under `AI_Idea_Bench/` (this directory).

Make sure Java 11 is installed.
Then install [SciPDF Parser](https://github.com/titipata/scipdf_parser) and start grobid:

```bash
git clone https://github.com/titipata/scipdf_parser.git
pip install git+https://github.com/titipata/scipdf_parser
python -m spacy download en_core_web_sm
cd scipdf_parser; bash serve_grobid.sh; cd ..
```

My project code framework for reference
```
aiideabench/
├── AI_Idea_Bench (submodule repo)
├── papers_data
├── target_paper_data.json

```

Run idea generation:

```bash
export SCIEVO_DIR=/path/to/SciEvo
python AI_Idea_Bench/AI-Scientist/generate_ideas_fron_papers_scievo.py
```

Grading:

```bash
# Multiple-choice evaluation
pushd AI_Idea_Bench; python Make_MCQ.py; python MCQ.py; popd

# Idea-to-idea matching
pushd AI_Idea_Bench; python idea_gt_idea.py; popd

# Idea-to-topic matching
pushd AI_Idea_Bench; python idea_gt_topic.py; popd

# Competition among baselines
pushd AI_Idea_Bench; python competition.py; popd

# Novelty assessment
pushd AI_Idea_Bench; python find_paper_by_kewords.py; python extract_hd_cd_paper.py; python Novelty.py; popd

# Feasibility
pushd AI_Idea_Bench; python split_experimental_plan.py; python feasibility.py; popd
```
