# RetroDCVAE
A novel template-free retrosynthesizer that can generate diverse sets of reactants for a desired product via discrete conditional variational autoencoders.

## 1. Environmental setup
### System requirements
**Ubuntu**: >= 16.04 <br>
**conda**: >= 4.0 <br>
**GPU**: at least 8GB Memory with CUDA >= 10.1


### Using conda
Please ensure that conda has been properly initialized, i.e. **conda activate** is runnable. Then
```
bash -i scripts/setup.sh
conda activate retro
```

## 2. Data preparation
Download the raw (cleaned and tokenized) data from Google Drive by
```
python scripts/download_raw_data.py --data_name=USPTO_50k
python scripts/download_raw_data.py --data_name=USPTO_DIVERSE
```
It is okay to only download the dataset(s) you want.
Run **homeMade.ipynb** in **data/** to derive new dataset USPTO-DIVERSE.
For each dataset, modify the following environmental variables in **scripts/preprocess.sh** or run as follows:

DATASET: one of [**USPTO_50k**, **USPTO_DIVERSE**] <br>
N_WORKERS: number of CPU cores (for parallel preprocessing)

Then run the preprocessing script by
```
sh scripts/preprocess.sh DATASET
sh scripts/preprocess.sh "USPTO_DIVERSE"
```

## 3. Model training and validation
Modify the following environmental variables in **scripts/train_g2s.sh** or run as follows:

EXP_NO: your own identifier (any string) for logging and tracking <br>
DATASET: one of [**USPTO_50k**, **USPTO_DIVERSE**] <br>
K_SIZE: user-defined latent size

Then run the training script by
```
sh scripts/train_g2s.sh EXP_NO DATASET K_SIZE
sh scripts/train_g2s.sh "demo" "USPTO_DIVERSE" "20"
sh scripts/train_g2s.sh "demo" "USPTO_50k" "10"
```

The training process regularly evaluates on the validation sets, both with and without teacher forcing.
While this evaluation is done mostly with top-1 accuracy,
it is also possible to do holistic evaluation *after* training finishes to get all the top-n accuracies on the val set.
To do that, first modify the following environmental variables in **scripts/validate.sh**:

EXP_NO: your own identifier (any string) for logging and tracking <br>
DATASET: one of [**USPTO_50k**, **USPTO_DIVERSE**] <br>
CHECKPOINT: the *folder* containing the checkpoints <br>
FIRST_STEP: the step of the first checkpoints to be evaluated <br>
LAST_STEP: the step of the last checkpoints to be evaluated

Then run the evaluation script by
```
sh scripts/validate.sh EXP_NO DATASET
sh scripts/validate.sh "demo" "USPTO_50k"
```
Note: the evaluation process performs beam search over the whole val sets for all checkpoints.


## 4. Testing
Modify the following environmental variables in **scripts/predict.sh**:

EXP_NO: your own identifier (any string) for logging and tracking <br>
DATASET: one of [**USPTO_50k**, **USPTO_DIVERSE**] <br>
CHECKPOINT: the *path* to the checkpoint (which is a .pt file) <br>

Then run the testing script by
```
sh scripts/predict.sh EXP_NO DATASET
## change CKPT in predict.sh according to the top-1 accuracy during the validation phase
sh scripts/predict.sh "demo" "USPTO_50k"
```
which will first run beam search to generate the results for all the test inputs,
and then computes the average top-n accuracies.
