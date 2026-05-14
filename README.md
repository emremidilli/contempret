# contempret

## Run Container

In order to enter inside of the container, run following command:

```bash
docker exec -it contempret-app_contempret-1 bash
```

## Experiments

### ETTh1-96

#### Pre-processing for pre-training
```bash
python pre_process_pre_training.py \
  --input_dir="./bin/interim/ETTh1_OT_96_96" \
  --output_dir="./bin/preprocessed/ETTh1_OT_96" \
  --pool_size_trend="24" \
  --sigma="1.96" \
  --use_timestamp="True"
```

#### Hyperparameter tuning
```bash
python tune_hyperparameters.py \
  --input_dir="./bin/preprocessed/ETTh1_OT_96" \
  --output_dir="./bin/hp-tuning/ETTh1_OT_96" \
  --mini_batch_size=64 \
  --max_epochs=10 \
  --executions_per_trial=2 \
  --factor=3
```

#### Pre-train foundational model
```bash
python task/pre_train.py \
  --input_dir="./bin/preprocessed/ETTh1_OT_96" \
  --output_dir="./bin/models/pt04/" \
  --mask_rate=0.40 \
  --mask_scalar=0.00 \
  --mini_batch_size=64 \
  --clip_norm=0.1 \
  --l1_trend=0.000020347 \
  --l2_trend=0.00041159 \
  --l1_seasonality=0.000085324 \
  --l2_seasonality=0.000090101 \
  --l1_residual=0.000065279 \
  --l2_residual=0.0050711 \
  --nr_of_encoder_blocks=2 \
  --nr_of_heads=2 \
  --dropout_rate=0.3 \
  --encoder_ffn_units=64 \
  --embedding_dims=32 \
  --projection_head=16 \
  --warmup_steps=4000 \
  --scale_factor=1.0 \
  --mae_threshold_comp=0.25 \
  --mae_threshold_tre=0.25 \
  --mae_threshold_sea=0.10 \
  --cl_margin=0.25 \
  --patch_size=12 \
  --prompt_pool_size=50 \
  --nr_of_most_similar_prompts=3 \
  --patience=20 \
  --warmup_epochs_early_stopping=100\
  --nr_of_seeds=1\
  --nr_of_epochs=3

  # --nr_of_most_similar_prompts=11 \
  # --encoder_ffn_units=384 \
  # --embedding_dims=256 \
  # --nr_of_seeds=10\
  # --nr_of_epochs=1000
```

#### Fine-tuning for full-shot

```bash
python task/fine_tune.py \
  --input_dir="./bin/input_representation/ETTh1_ALL_96_96/" \
  --output_dir="./bin/models/ft04/" \
  --pre_trained_model_dir="./bin/models/pt04/" \
  --patience="50" --clip_norm="0.1" \
  --learning_rate="0.0001" --warmup_epochs="1" --tune_time2vec="True" \
  --nr_of_seeds="1" \
  --nr_of_epochs="3" \
  --mini_batch_size=128

  #--mini_batch_size=128
  # --nr_of_seeds="10" \
  # --nr_of_epochs="10000"
```

