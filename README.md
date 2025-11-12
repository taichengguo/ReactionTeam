# ReactionTeam
ReactionTeam: Teaming Experts for Divergent Thinking Beyond Typical Reaction Patterns 

## 📑 Table of Contents

- [1. Environmental setup](#-1-environmental-setup)
  - [System requirements](#system-requirements)
  - [Using conda](#using-conda)
- [2. Data preparation](#-2-data-preparation)
- [3. Model training and validation](#-3-model-training-and-validation)
  - [3.1 Base Model Training](#31-base-model-training)
  - [3.2 LoRA Experts Training](#32-lora-experts-training)
- [4. Testing](#-4-testing)
  - [4.1 Base Model Testing](#41-base-model-testing)
  - [4.2 Teacher Forcing Based Ranking with Multiple Experts](#42-teacher-forcing-based-ranking-with-multiple-experts)
- [5. Acknowledgements](#-5-acknowledgements)
- [6. Contact](#-6-contact)

## 🔧 1. Environmental setup
### System requirements
**Ubuntu**: >= 16.04 <br>
**conda**: >= 4.0 <br>
**GPU**: at least 8GB Memory with CUDA >= 10.1

Note: there is some known compatibility issue with RTX 3090,
for which the PyTorch would need to be upgraded to >= 1.8.0.
The code has not been heavily tested under 1.8.0, so our best advice is to use some other GPU.

### Using conda
Please ensure that conda has been properly initialized, i.e. **conda activate** is runnable. Then
```
bash -i scripts/setup.sh
conda activate graph2smiles
```

## 📊 2. Data preparation
Download the raw (cleaned and tokenized) data from Google Drive by
```
python scripts/download_raw_data.py --data_name=USPTO_50k
python scripts/download_raw_data.py --data_name=USPTO_full
python scripts/download_raw_data.py --data_name=USPTO_480k
python scripts/download_raw_data.py --data_name=USPTO_STEREO
```
It is okay to only download the dataset(s) you want.
For each dataset, modify the following environmental variables in **scripts/preprocess.sh**:

DATASET: one of [**USPTO_50k**, **USPTO_full**, **USPTO_480k**, **USPTO_STEREO**] <br>
TASK: **retrosynthesis** for 50k and full, or **reaction_prediction** for 480k and STEREO <br>
N_WORKERS: number of CPU cores (for parallel preprocessing)

Then run the preprocessing script by
```
sh scripts/preprocess.sh
```

## 🚀 3. Model training and validation

### 3.1 Base Model Training
Modify the following environmental variables in **scripts/train_g2s.sh**:

EXP_NO: your own identifier (any string) for logging and tracking <br>
DATASET: one of [**USPTO_50k**, **USPTO_full**, **USPTO_480k**, **USPTO_STEREO**] <br>
TASK: **retrosynthesis** for 50k and full, or **reaction_prediction** for 480k and STEREO <br>
MPN_TYPE: one of [**dgcn**, **dgat**]

Then run the training script by
```
sh scripts/train_g2s.sh
```

The training process regularly evaluates on the validation sets, both with and without teacher forcing.
While this evaluation is done mostly with top-1 accuracy,
it is also possible to do holistic evaluation *after* training finishes to get all the top-n accuracies on the val set.
To do that, first modify the following environmental variables in **scripts/validate.sh**:

EXP_NO: your own identifier (any string) for logging and tracking <br>
DATASET: one of [**USPTO_50k**, **USPTO_full**, **USPTO_480k**, **USPTO_STEREO**] <br>
CHECKPOINT: the *folder* containing the checkpoints <br>
FIRST_STEP: the step of the first checkpoints to be evaluated <br>
LAST_STEP: the step of the last checkpoints to be evaluated

Then run the evaluation script by
```
sh scripts/validate.sh
```
Note: the evaluation process performs beam search over the whole val sets for all checkpoints.
It can take tens of hours.

### 3.2 LoRA Experts Training
Based on the pretrained base model, we support training multiple LoRA (Low-Rank Adaptation) experts for divergent thinking.
The LoRA experts training process includes:

1. **LoRA Application**: LoRA adapters are automatically applied to the decoder's self-attention layers with configurable rank and alpha parameters (default: rank=128, alpha=256).

2. **Parameter Freezing**: During LoRA expert training, all base model parameters are frozen, and only LoRA parameters are trainable, making the training efficient.

3. **Boosting Mechanism**: The training uses a boosting strategy that:
   - Tracks correctly predicted samples for each epoch (saved to `correct_index_epoch_{epoch}.txt`)
   - Periodically (controlled by `--boosting_interval`, default: 2) removes already-learned samples from the training set
   - Uses the intersection of correctly predicted samples across multiple epochs to identify hard examples
   - Updates the training dataset to focus on remaining challenging samples

4. **Expert Checkpointing**: Each LoRA expert is saved as a separate checkpoint (e.g., `save_stere/model.{epoch}.pt`) after the boosting interval.

To train LoRA experts, use the same **scripts/train_g2s.sh** script with a pretrained checkpoint specified in `LOAD_FROM`.
The training will automatically:
- Load the pretrained model
- Apply LoRA adapters to the decoder
- Freeze base model weights
- Train only LoRA parameters with the boosting mechanism

Key parameters in **scripts/train_g2s.sh**:
- `LOAD_FROM`: path to the pretrained base model checkpoint
- `boosting_interval`: number of epochs between expert checkpoints (default: 2)
- The script automatically adjusts `boosting_interval` to 1 after epoch 12

We provide pretrained model checkpoints for all four datasets with both dgcn and dgat,
which can be downloaded from Google Drive with
```
python scripts/download_checkpoints.py --data_name=$DATASET --mpn_type=$MPN_TYPE
```
using any combinations of DATASET and MPN_TYPE.

## 🧪 4. Testing

### 4.1 Base Model Testing
Modify the following environmental variables in **scripts/predict.sh**:

EXP_NO: your own identifier (any string) for logging and tracking <br>
DATASET: one of [**USPTO_50k**, **USPTO_full**, **USPTO_480k**, **USPTO_STEREO**] <br>
CHECKPOINT: the *path* to the checkpoint (which is a .pt file) <br>

Then run the testing script by
```
sh scripts/predict.sh
```
which will first run beam search to generate the results for all the test inputs,
and then computes the average top-n accuracies.

### 4.2 Teacher Forcing Based Ranking with Multiple Experts
For improved prediction accuracy using multiple LoRA experts and dropout variants, we provide a teacher forcing-based ranking system.

The ranking system works as follows:

1. **Candidate Collection**: Collects prediction candidates from:
   - The base model (top-N predictions)
   - Multiple LoRA expert models (specified by `--num_experts`)
   - Multiple dropout variants of each expert (specified by `--num_dropout`)

2. **Teacher Forcing Scoring**: Uses teacher forcing to compute scores for each candidate prediction by evaluating the input-output likelihood using `model.input_output_score()`.

3. **Ranking and Selection**: 
   - Sorts all candidates by their teacher forcing scores
   - Removes duplicate predictions while preserving order
   - Outputs the top-ranked predictions

To use the teacher forcing-based ranking system, modify the following environmental variables in **scripts/predict_ours_sorting.sh**:

- `DATASET`: one of [**USPTO_50k**, **USPTO_full**, **USPTO_480k**, **USPTO_STEREO**]
- `CHECKPOINT`: path to the base model checkpoint
- `MPN_TYPE`: one of [**dgcn**, **dgat**]
- `BS`: beam size for candidate generation (default: 30)
- `NBEST`: number of best candidates to keep per model (default: 50)
- `--num_experts`: number of LoRA expert models to use (e.g., 6)
- `--num_dropout`: number of dropout variants per expert (e.g., 10)

Then run the ranking script by
```
sh scripts/predict_ours_sorting.sh
```

**Prerequisites**: Before running the ranking script, you need to:
1. Generate prediction results from the base model and save them in `./results/` directory
2. Generate prediction results from each LoRA expert model (with and without dropout variants)
3. Ensure prediction result files follow the naming convention: `{DATASET}_g2s_series_rel_smiles_smiles_{expert_id}_drop{seed}_result_idx_token.txt`

The script will:
- Load all candidate predictions from the specified experts and dropout variants
- Score each candidate using teacher forcing
- Rank and deduplicate predictions
- Compute top-N accuracies and save results to `results_summary{topn}_final.txt`

## 🙏 5. Acknowledgements

This work is built upon the [Graph2SMILES](https://github.com/coleygroup/Graph2SMILES) project. We gratefully acknowledge the original authors for their excellent work on graph-to-sequence models for retrosynthesis and reaction outcome prediction.

Graph2SMILES provides the foundation for:
- Graph-to-sequence model architecture
- Data preprocessing pipelines
- Base model training infrastructure
- Evaluation frameworks

We extend their work by adding:
- LoRA (Low-Rank Adaptation) experts training for divergent thinking
- Teacher forcing-based ranking system for multi-expert ensemble predictions
- Boosting mechanism for focusing on challenging samples

## 📧 6. Contact

For questions, issues, or collaborations, please contact tguo2@nd.edu, thanks
