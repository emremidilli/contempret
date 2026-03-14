# contempret

## Fine-Tuning

```bash
python fine_tune.py \
  --input_dir="./bin/input_representation/ETTh1_OT_96_96/" \
  --output_dir="./bin/models/ETTh1_OT_96_96/" \
  --pre_trained_model_dir="./bin/models/96_utsd1gv4_16/" \
  --mini_batch_size=128 --patience="50" --clip_norm="0.1" --nr_of_epochs="10000" \
  --learning_rate="0.0001" --warmup_epochs="1" --tune_time2vec="True" --nr_of_seeds="10"
```