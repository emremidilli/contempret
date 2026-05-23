# contempret

## Run Container

In order to enter inside of the container, run following command:

```bash
docker exec -it contempret-app_contempret-1 bash
```

## Experiments


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