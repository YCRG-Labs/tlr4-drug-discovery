Code Associated With:

## Machine Learning-Driven Prediction of TLR4 Binding Affinity: A Comprehensive Molecular Feature Analysis for Drug Discovery

#### Brandon Yee, Maximilian Rutowski, Wilson Collins, Daniel Huang, Caden Wang, Mihir Tekal, Lev Kung, Oliver Pierborne

Correspondence: b.yee@ycrg-labs.org

Data from: [TLR4 Binding Data](https://github.com/YCRG-Labs/binding-data)

### Quick Start

``` bash
# 0) From repo root
git clone https://github.com/YCRG-Labs/drug_discovery_tlr4

# 1) From dataset repo
cd drug_discovery_tlr4
git clone https://github.com/YCRG-Labs/binding-data.git

# 2) Create/activate env
python3 -m venv venv && source venv/bin/activate

# 3) Install deps
pip install -r requirements.txt

# 4) Prepare inputs (adjust paths if yours differ)
#    - PDBQT files in: binding-data/raw/pdbqt/
#    - Binding CSV at: binding-data/processed/processed_logs.csv

# 5) Run full pipeline (features + preprocess + train + evaluate)
python main.py --pipeline complete --pdbqt-dir binding-data/raw/pdbqt --binding-csv binding-data/processed/processed_logs.csv

# 6) Or run stage-by-stage

# 6a) Extract features
python main.py --pipeline features
  --pdbqt-dir binding-data/raw/pdbqt

# 6b) Train models (uses generated features + binding data)
python main.py --pipeline train --features-csv data/processed/features.csv --binding-csv binding-data/processed/processed_logs.csv

# 6c) Predict for a single compound
python main.py --pipeline predict --model-path models/trained/best_model.joblib --pdbqt-file binding-data/raw/pdbqt/example.pdbqt

# 7) (Recommended) Run tests in verbose mode
python -m pytest tests -v
```

