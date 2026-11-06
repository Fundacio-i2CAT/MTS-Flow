# Flow Matching for Multivariate Time-Series Generation with Mamba

## Description
This repository contains the official implementation of "MTS-Flow: Flow Matching for Multivariate Time-Series Generation with Mamba", a Flow matching and Mamba (State Space Models) model designed for efficient long-sequence multivariate time-series generation.  

## Project Architecture
The codebase strictly separates concerns into independent modules:
- `configs/`: YAML configuration files containing all hyperparameters.
- `src/architecture/`: Model implementations and factory patterns.
  - `mamba_flow_matching/`: Specifics of the Mamba flow-matching components (time embeddings, blocks, ODE flow).
- `src/data/`: Universal multi-format (CSV, Numpy, Parquet) Dataloaders.
- `src/utils/`: MLflow logging, configuration and other utils.
- `src/evaluation/`: statistical metrics, dicriminative and predictive evaluation, performance anaysis.
- `train.py`: Unified training loop with MLflow integration.
- `inference.py`: Evaluation and sampling loop with Euler ODE solver.

## Installation

This project utilizes `uv` for lightning-fast and deterministic package management. No `requirements.txt` is used; dependencies are maintained inside `pyproject.toml`.

```bash
# Sync dependencies and create a virtual environment
uv sync

# Activate environment (or simply use `uv run`)
source .venv/bin/activate
```

## Usage
  1. Download and process datasets

      Place your datasets in the `data/` directory.
      The expected data format is: [num_sequences, sequence_length, num_features]

      These scripts will convert and save the datasets into the correct format for training your network.

      - ETT dataset:
        1. Download the raw dataset from the GitHub repository: [ETTh1](https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTh1.csv)
        2. Place it under:
            data/ETT/ETT_raw/
        3. Then run the processing script:
            ```bash
            python3 data/ett_processing.py
            ```
 
      - hihAigua dataset:
        1. Download the raw dataset and `processing_script.py` from Zenodo:
            [hihAigua](https://zenodo.org/records/21704780).
        2. Place both of them under:
            data/hihAigua/
        3. Run the datset `processing_script.py`:
            ```bash
            python3 data/hihAigua/processing_script.py
            ```
        4. Run the `hihAigua_processing.py` script located in `data/hihAigua_processing.py`:
            ```bash
            python3 data/hihAigua_processing.py
            ```

  2. Start MLflow 

      This repository leverages MLflow for experiment tracking. Start the MLflow UI to train models, view runs, compare metrics, and access checkpoints:
      ```bash
      uv run mlflow ui --backend-store-uri sqlite:///mlruns.db
      ```
  3. Training

      All hyperparameters are controlled via configuration YAMLs.
      The `config.yaml` manages:
      - **`dataset`**: Specifies path, format, lengths, and dimensionalities.
      - **`model`**: Architectural choices (d_model, layers, d_state).
      - **`flow_matching`**: ODE/SDE parameters like `sigma_min`.
      - **`training`**: Learning rate, precision, epochs, etc.
      - **`mlflow`**: Experiment tracking configurations.
    
      ```bash
      uv run python train.py --config configs/config.yaml
      ```
      During training, parameters, losses, metrics, and models are tracked automatically using MLflow.

      To perform hyperparameter optimization, set the --use_optuna flag to true and define the search space in the `objective` function in `train.py`.

  4. Inference

      Load a trained checkpoint and generate samples.
      ```bash
      uv run python inference.py \
          --config configs/config.yaml \
          --checkpoint checkpoints/model_epoch_100.pt \
          --samples 100 \
          --ode-steps 50 \
          --output generated.npy
      ```

## Authors
Developed by Santiago Escuder Folch (I2CAT Foundation) and Oriol Graupera-Serra (I2CAT Foundation).  

## Copyright
This code has been developed by Fundació Privada Internet i Innovació Digital a Catalunya (i2CAT).
i2CAT is a *non-profit research and innovation centre* that  promotes mission-driven knowledge to solve business challenges, co-create solutions with a transformative impact, empower citizens through open and participative digital social innovation with territorial capillarity, and promote pioneering and strategic initiatives.
i2CAT *aims to transfer* research project results to private companies in order to create social and economic impact via the out-licensing of intellectual property and the creation of spin-offs.
Find more information of i2CAT projects and IP rights at https://i2cat.net/tech-transfer/

## License
This code is licensed under the terms GNU Affero General Public License v3.0 (*AGPLv3*). Information about the license can be located at [link](https://www.gnu.org/licenses/agpl-3.0.html).
If you find that this license doesn't fit with your requirements regarding the use, distribution or redistribution of our code for your specific work, please, don’t hesitate to contact the intellectual property managers in i2CAT at the following address: techtransfer@i2cat.net

## Citation
If you find this repo useful, please cite our paper via
```bibtex

```
## Support
For questions, issues, or collaboration inquiries, please contact:  
santiago.escuder@i2cat.net
oriol.graupera@i2cat.net
albert.calvo@i2cat.net