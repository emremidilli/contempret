# contempret

## Run Container

In order to enter inside of the container, run following command:

```bash
docker exec -it contempret-app_contempret-1 bash
```

## Experiments

## Pre-Training

### Stopping Criteria

| model_name | composed | tred | seasonality |
|:----:|:-----:|:-----:|:-----:|
| pt_etth_96 | 0.16 | 0.09 | 0.07 |
| pt_etth_192 | 0.19 | 0.14 | 0.08 |
| pt_etth_384 | --- | --- | --- |
| pt_etth_768 | --- | --- | --- |

### Ablation Studies

#### W/O Sequential Training

Models whose hyperparameter configuration is same with `pt_etth_96` are pre-trained with following weighted losses based on a grid search:

| trial | w_comp | w_tre | w_sea | w_cl | test_mae_composed |
|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|1|0.500|0.167|0.167|0.167|0.774|
|2|0.167|0.500|0.167|0.167|0.679|
|3|0.167|0.167|0.500|0.167|0.931|
|4|0.167|0.167|0.167|0.500|0.853|
|5|0.333|0.333|0.167|0.167|0.701|
|6|0.333|0.167|0.333|0.167|0.827|
|7|0.333|0.167|0.167|0.333|0.791|
|8|0.167|0.333|0.333|0.167|0.784|
|9|0.167|0.333|0.167|0.333|0.724|
|10|0.167|0.167|0.333|0.333|0.895|

Each trial is run for `epochs=5` and `seed=1`. The configuration whose `val_mae_composed` is the smallest is identified as the competing model. The competing model is pre-trained with the same `epochs` and `seed` with the baseline.
