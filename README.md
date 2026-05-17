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
python task/tune_hyperparameters.py \
  --input_dir="./bin/preprocessed/ETTh1_OT_96" \
  --output_dir="./bin/hp-tuning/pt04" \
  --mini_batch_size=64 \
  --max_epochs=10 \
  --executions_per_trial=2 \
  --factor=3
```

#### Pre-train foundational model
```bash
python task/pre_train.py \
  --input_dir="./bin/preprocessed/ETTh1_OT_96" \
  --output_dir="./bin/models/pt10/" \
  --mask_rate=0.40 \
  --mask_scalar=0.00 \
  --mini_batch_size=128 \
  --clip_norm=0.1 \
  --l1_trend=0.000631 \
  --l2_trend=1.5435e-06 \
  --l1_seasonality=0.00081708 \
  --l2_seasonality=0.000090101 \
  --l1_residual=0.0068574 \
  --l2_residual=0.0017232 \
  --nr_of_encoder_blocks=8 \
  --nr_of_heads=2 \
  --dropout_rate=0.3 \
  --encoder_ffn_units=256 \
  --embedding_dims=64 \
  --projection_head=32 \
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
  --warmup_epochs_early_stopping=100 \
  --nr_of_seeds=1 \
  --nr_of_epochs=2
```
`
  # the most recent one
4                 |8                 |nr_of_encoder_blocks
0.3               |0.3               |dropout_rate
384               |256               |encoder_ffn_units
64                |64                |embedding_dims
2                 |2                 |nr_of_heads
32                |32                |projection_head_units
40                |50                |prompt_pool_size
8                 |12                |patch_size
4                 |3                 |nr_of_most_similar_prompts
0.00012635        |0.000631          |l1_trend
4.2692e-05        |1.5435e-06        |l2_trend
0.00014249        |0.00081708        |l1_seasonality
0.00019867        |1.0746e-05        |l2_seasonality
0.00011195        |0.0068574         |l1_residual
0.0090439         |0.0017232         |l2_residual
`

#### Fine-tuning for full-shot

```bash
python task/fine_tune.py \
  --input_dir="./bin/input_representation/ETTh1_ALL_96_96/" \
  --output_dir="./bin/models/ft10/" \
  --pre_trained_model_dir="./bin/models/pt10/" \
  --patience=50 \
  --clip_norm=0.1 \
  --learning_rate=0.0001 \
  --warmup_epochs=1 \
  --tune_time2vec="True" \
  --nr_of_seeds=1 \
  --nr_of_epochs=10 \
  --mini_batch_size=128
```
