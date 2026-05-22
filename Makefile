CONTAINER = contempret-app_contempret-1
PYTHON = docker exec -i $(CONTAINER) python

.PHONY: install run preprocess preprocess_ft \
	preprocess_pt_ETTh1_96 preprocess_pt_ETTh1_192 preprocess_pt_ETTh1_384 preprocess_pt_ETTh1_768 \
	preprocess_pt_ETTh2_96 preprocess_pt_ETTh2_192 preprocess_pt_ETTh2_384 preprocess_pt_ETTh2_768 \
	preprocess_pt_ETTm1_96 preprocess_pt_ETTm1_192 preprocess_pt_ETTm1_384 preprocess_pt_ETTm1_768 \
	preprocess_pt_ETTm2_96 preprocess_pt_ETTm2_192 preprocess_pt_ETTm2_384 preprocess_pt_ETTm2_768 \
	preprocess_ft_ETTh1_96_96 preprocess_ft_ETTh1_192_192 preprocess_ft_ETTh1_384_336 preprocess_ft_ETTh1_768_720 \
	preprocess_ft_ETTh2_96_96 preprocess_ft_ETTh2_192_192 preprocess_ft_ETTh2_384_336 preprocess_ft_ETTh2_768_720 \
	preprocess_ft_ETTm1_96_96 preprocess_ft_ETTm1_192_192 preprocess_ft_ETTm1_384_336 preprocess_ft_ETTm1_768_720 \
	preprocess_ft_ETTm2_96_96 preprocess_ft_ETTm2_192_192 preprocess_ft_ETTm2_384_336 preprocess_ft_ETTm2_768_720 \
	train_foundation_etth1_96

ID ?= pt_ETTh1_96

install:
	docker-compose down
	docker-compose build

run:
	docker-compose up -d

# ── Pre-training preprocessing ─────────────────────────────────────────────────

define PIPELINE_PREPROCESS_PRE_TRAINING
	$(MAKE) run_preprocess_pre_training ID=$(ID)
endef

run_preprocess_pre_training: run
	$(PYTHON) task/pre_process_pre_training.py \
		--input_dir="./bin/interim/$(ID)" \
		--output_dir="./bin/preprocessed/$(ID)" \
		--pool_size_trend="24" \
		--sigma="1.96" \
		--use_timestamp="True"

# ── Fine-tuning preprocessing ─────────────────────────────────────────────────
define PIPELINE_PREPROCESS_FINE_TUNING
	$(MAKE) run_preprocess_fine_tuning ID=$(ID)
endef

run_preprocess_fine_tuning: run
	$(PYTHON) task/pre_process_fine_tuning.py \
		--input_dir="./bin/interim/$(ID)" \
		--output_dir="./bin/preprocessed/$(ID)" \
		--pool_size_trend="24" \
		--sigma="1.96"

# ── ETTh1 ──────────────────────────────────────────────────────────────────────

preprocess_pt_ETTh1_96: ID = pt_ETTh1_96
preprocess_pt_ETTh1_96:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_ft_ETTh1_96_96: ID = ft_ETTh1_96_96
preprocess_ft_ETTh1_96_96:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_pt_ETTh1_192: ID = pt_ETTh1_192
preprocess_pt_ETTh1_192:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_ft_ETTh1_192_192: ID = ft_ETTh1_192_192
preprocess_ft_ETTh1_192_192:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_pt_ETTh1_384: ID = pt_ETTh1_384
preprocess_pt_ETTh1_384:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_ft_ETTh1_384_336: ID = ft_ETTh1_384_336
preprocess_ft_ETTh1_384_336:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_pt_ETTh1_768: ID = pt_ETTh1_768
preprocess_pt_ETTh1_768:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_ft_ETTh1_768_720: ID = ft_ETTh1_768_720
preprocess_ft_ETTh1_768_720:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

# ── ETTh2 ──────────────────────────────────────────────────────────────────────

preprocess_pt_ETTh2_96: ID = pt_ETTh2_96
preprocess_pt_ETTh2_96:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_ft_ETTh2_96_96: ID = ft_ETTh2_96_96
preprocess_ft_ETTh2_96_96:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_pt_ETTh2_192: ID = pt_ETTh2_192
preprocess_pt_ETTh2_192:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_ft_ETTh2_192_192: ID = ft_ETTh2_192_192
preprocess_ft_ETTh2_192_192:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_pt_ETTh2_384: ID = pt_ETTh2_384
preprocess_pt_ETTh2_384:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_ft_ETTh2_384_336: ID = ft_ETTh2_384_336
preprocess_ft_ETTh2_384_336:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_pt_ETTh2_768: ID = pt_ETTh2_768
preprocess_pt_ETTh2_768:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_ft_ETTh2_768_720: ID = ft_ETTh2_768_720
preprocess_ft_ETTh2_768_720:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

# ── ETTm1 ──────────────────────────────────────────────────────────────────────

preprocess_pt_ETTm1_96: ID = pt_ETTm1_96
preprocess_pt_ETTm1_96:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_ft_ETTm1_96_96: ID = ft_ETTm1_96_96
preprocess_ft_ETTm1_96_96:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_pt_ETTm1_192: ID = pt_ETTm1_192
preprocess_pt_ETTm1_192:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_ft_ETTm1_192_192: ID = ft_ETTm1_192_192
preprocess_ft_ETTm1_192_192:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_pt_ETTm1_384: ID = pt_ETTm1_384
preprocess_pt_ETTm1_384:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_ft_ETTm1_384_336: ID = ft_ETTm1_384_336
preprocess_ft_ETTm1_384_336:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_pt_ETTm1_768: ID = pt_ETTm1_768
preprocess_pt_ETTm1_768:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_ft_ETTm1_768_720: ID = ft_ETTm1_768_720
preprocess_ft_ETTm1_768_720:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

# ── ETTm2 ──────────────────────────────────────────────────────────────────────

preprocess_pt_ETTm2_96: ID = pt_ETTm2_96
preprocess_pt_ETTm2_96:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_ft_ETTm2_96_96: ID = ft_ETTm2_96_96
preprocess_ft_ETTm2_96_96:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_pt_ETTm2_192: ID = pt_ETTm2_192
preprocess_pt_ETTm2_192:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_ft_ETTm2_192_192: ID = ft_ETTm2_192_192
preprocess_ft_ETTm2_192_192:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_pt_ETTm2_384: ID = pt_ETTm2_384
preprocess_pt_ETTm2_384:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_ft_ETTm2_384_336: ID = ft_ETTm2_384_336
preprocess_ft_ETTm2_384_336:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_pt_ETTm2_768: ID = pt_ETTm2_768
preprocess_pt_ETTm2_768:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_ft_ETTm2_768_720: ID = ft_ETTm2_768_720
preprocess_ft_ETTm2_768_720:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_pt: \
	preprocess_pt_ETTh1_96 preprocess_pt_ETTh1_192 preprocess_pt_ETTh1_384 preprocess_pt_ETTh1_768 \
	preprocess_pt_ETTh2_96 preprocess_pt_ETTh2_192 preprocess_pt_ETTh2_384 preprocess_pt_ETTh2_768 \
	preprocess_pt_ETTm1_96 preprocess_pt_ETTm1_192 preprocess_pt_ETTm1_384 preprocess_pt_ETTm1_768 \
	preprocess_pt_ETTm2_96 preprocess_pt_ETTm2_192 preprocess_pt_ETTm2_384 preprocess_pt_ETTm2_768

preprocess_ft: \
	preprocess_ft_ETTh1_96_96 preprocess_ft_ETTh1_192_192 preprocess_ft_ETTh1_384_336 preprocess_ft_ETTh1_768_720 \
	preprocess_ft_ETTh2_96_96 preprocess_ft_ETTh2_192_192 preprocess_ft_ETTh2_384_336 preprocess_ft_ETTh2_768_720 \
	preprocess_ft_ETTm1_96_96 preprocess_ft_ETTm1_192_192 preprocess_ft_ETTm1_384_336 preprocess_ft_ETTm1_768_720 \
	preprocess_ft_ETTm2_96_96 preprocess_ft_ETTm2_192_192 preprocess_ft_ETTm2_384_336 preprocess_ft_ETTm2_768_720

# ── Tiny ───────────────────────────────────────────────────────────────────────
preprocess_pt_tiny: ID = pt_tiny
preprocess_pt_tiny:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

# ── Training ───────────────────────────────────────────────────────────────────

train_foundation_etth1_96:
	$(PYTHON) task/pre_train.py \
		--input_dir="./bin/preprocessed/pt_ETTh1_96" \
		--output_dir="./bin/models/pt_ETTh1_96/" \
		--mask_rate=0.40 \
		--mask_scalar=0.00 \
		--mini_batch_size=128 \
		--clip_norm=0.1 \
		--l1_trend=0.000631 \
		--l2_trend=1.5435e-06 \
		--l1_seasonality=0.00081708 \
		--l2_seasonality=1.0746e-05 \
		--l1_residual=0.0068574 \
		--l2_residual=0.0017232 \
		--nr_of_encoder_blocks=8 \
		--nr_of_heads=2 \
		--dropout_rate=0.3 \
		--encoder_ffn_units=256 \
		--embedding_dims=64 \
		--projection_head=32 \
		--warmup_steps=4000 \
		--scale_factor=0.1 \
		--mae_threshold_comp=0.25 \
		--mae_threshold_tre=0.25 \
		--mae_threshold_sea=0.10 \
		--cl_margin=0.25 \
		--patch_size=12 \
		--prompt_pool_size=0 \
		--nr_of_most_similar_prompts=0 \
		--patience=20 \
		--warmup_epochs_early_stopping=100 \
		--nr_of_seeds=10 \
		--nr_of_epochs=1000


train_foundation_tiny:
	$(PYTHON) task/pre_train.py \
		--input_dir="./bin/preprocessed/pt_tiny" \
		--output_dir="./bin/models/pt_tiny/" \
		--mask_rate=0.40 \
		--mask_scalar=0.00 \
		--mini_batch_size=32 \
		--clip_norm=0.1 \
		--l1_trend=0.000631 \
		--l2_trend=1.5435e-06 \
		--l1_seasonality=0.00081708 \
		--l2_seasonality=1.0746e-05 \
		--l1_residual=0.0068574 \
		--l2_residual=0.0017232 \
		--nr_of_encoder_blocks=2 \
		--nr_of_heads=2 \
		--dropout_rate=0.3 \
		--encoder_ffn_units=32 \
		--embedding_dims=32 \
		--projection_head=8 \
		--warmup_steps=4000 \
		--scale_factor=0.1 \
		--mae_threshold_comp=0.25 \
		--mae_threshold_tre=0.25 \
		--mae_threshold_sea=0.10 \
		--cl_margin=0.25 \
		--patch_size=12 \
		--prompt_pool_size=0 \
		--nr_of_most_similar_prompts=0 \
		--patience=20 \
		--warmup_epochs_early_stopping=100 \
		--nr_of_seeds=1 \
		--nr_of_epochs=10