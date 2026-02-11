# Benchmark for MLEBench

We assume all commands are run from the same dir as this README file.

First, create a new uv env:

```bash
uv init --python 3.12
uv venv --seed
source .venv/bin/activate
```

Then, install the dependencies:

```bash
pip install -e ./SciCode
pip install -r ../../requirements.txt
```

Download the [numeric test results](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR?usp=drive_link) and save them as `SciCode/eval/data/test_data.h5`.

Run the evaluation script:

```bash
export SCIEVO_DIR=/path/to/SciEvo
# w/o background
pushd SciCode/; python eval/scripts/gencode_scievo.py --model scievo; popd
# with background
pushd SciCode/; python eval/scripts/gencode_scievo.py --model scievo --with-background; popd
```

Grading script:

```bash
# w/o background
python eval/scripts/test_generated_code.py --model scievo
# with background
python eval/scripts/test_generated_code.py --model scievo --with-background
```
