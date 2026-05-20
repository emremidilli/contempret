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
  --output_dir="./bin/models/ft_etth1_96_96_wo_prompt/" \
  --pre_trained_model_dir="./bin/models/pt_etth1_96_wo_prompt/" \
  --patience=50 \
  --clip_norm=0.1 \
  --learning_rate=0.001 \
  --warmup_epochs=1 \
  --tune_time2vec="True" \
  --nr_of_seeds=10 \
  --nr_of_epochs=1000 \
  --mini_batch_size=128
```
