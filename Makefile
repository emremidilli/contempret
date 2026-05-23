CONTAINER = contempret-app_contempret-1
PYTHON = docker exec -i $(CONTAINER) python

.PHONY: install run preprocess_pt preprocess_ft \
	preprocess_pt_etth_96 preprocess_pt_etth_192 preprocess_pt_etth_384 preprocess_pt_etth_768 \
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

preprocess_ft_ETTh1_96_96: ID = ft_ETTh1_96_96
preprocess_ft_ETTh1_96_96:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_ft_ETTh1_192_192: ID = ft_ETTh1_192_192
preprocess_ft_ETTh1_192_192:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_ft_ETTh1_384_336: ID = ft_ETTh1_384_336
preprocess_ft_ETTh1_384_336:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_ft_ETTh1_768_720: ID = ft_ETTh1_768_720
preprocess_ft_ETTh1_768_720:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

# ── ETTh2 ──────────────────────────────────────────────────────────────────────

preprocess_ft_ETTh2_96_96: ID = ft_ETTh2_96_96
preprocess_ft_ETTh2_96_96:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_ft_ETTh2_192_192: ID = ft_ETTh2_192_192
preprocess_ft_ETTh2_192_192:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_ft_ETTh2_384_336: ID = ft_ETTh2_384_336
preprocess_ft_ETTh2_384_336:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_ft_ETTh2_768_720: ID = ft_ETTh2_768_720
preprocess_ft_ETTh2_768_720:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

# ── ETTm1 ──────────────────────────────────────────────────────────────────────

preprocess_ft_ETTm1_96_96: ID = ft_ETTm1_96_96
preprocess_ft_ETTm1_96_96:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_ft_ETTm1_192_192: ID = ft_ETTm1_192_192
preprocess_ft_ETTm1_192_192:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_ft_ETTm1_384_336: ID = ft_ETTm1_384_336
preprocess_ft_ETTm1_384_336:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_ft_ETTm1_768_720: ID = ft_ETTm1_768_720
preprocess_ft_ETTm1_768_720:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

# ── ETTm2 ──────────────────────────────────────────────────────────────────────

preprocess_ft_ETTm2_96_96: ID = ft_ETTm2_96_96
preprocess_ft_ETTm2_96_96:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_ft_ETTm2_192_192: ID = ft_ETTm2_192_192
preprocess_ft_ETTm2_192_192:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_ft_ETTm2_384_336: ID = ft_ETTm2_384_336
preprocess_ft_ETTm2_384_336:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_ft_ETTm2_768_720: ID = ft_ETTm2_768_720
preprocess_ft_ETTm2_768_720:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

preprocess_ft: \
	preprocess_ft_ETTh1_96_96 preprocess_ft_ETTh1_192_192 preprocess_ft_ETTh1_384_336 preprocess_ft_ETTh1_768_720 \
	preprocess_ft_ETTh2_96_96 preprocess_ft_ETTh2_192_192 preprocess_ft_ETTh2_384_336 preprocess_ft_ETTh2_768_720 \
	preprocess_ft_ETTm1_96_96 preprocess_ft_ETTm1_192_192 preprocess_ft_ETTm1_384_336 preprocess_ft_ETTm1_768_720 \
	preprocess_ft_ETTm2_96_96 preprocess_ft_ETTm2_192_192 preprocess_ft_ETTm2_384_336 preprocess_ft_ETTm2_768_720

# ── Tiny ───────────────────────────────────────────────────────────────────────
preprocess_pt_tiny: ID = pt_tiny
preprocess_pt_tiny:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_ft_tiny: ID = ft_tiny
preprocess_ft_tiny:
	$(PIPELINE_PREPROCESS_FINE_TUNING)

hp_tune_tiny:
	$(PYTHON) task/tune_hyperparameters.py \
		--input_dir="./bin/preprocessed/pt_tiny" \
		--output_dir="./bin/hp-tuning/pt_tiny" \
		--mini_batch_size=32 \
		--max_epochs=3 \
		--executions_per_trial=1 \
		--factor=2 \
		--training_mode="weighted"

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
		--nr_of_epochs=10 \
		--w_comp=0.25 \
		--w_tre=0.75 \
		--w_sea=1.00 \
		--w_cl=1.00

train_fine_tuning_tiny:
	$(PYTHON) task/fine_tune.py \
		--input_dir="./bin/preprocessed/ft_tiny/" \
		--output_dir="./bin/models/ft_tiny/" \
		--pre_trained_model_dir="./bin/models/pt_tiny/" \
		--patience=50 \
		--clip_norm=0.1 \
		--learning_rate=0.001 \
		--warmup_epochs=1 \
		--tune_time2vec="True" \
		--nr_of_seeds=1 \
		--nr_of_epochs=10 \
		--mini_batch_size=32

# ── Compile heteregenous data ───────────────────────────────────────────────────────────────────
compile_pt_etth_96_data:
	$(PYTHON) task/compile_heteregenous_data.py \
		--input_dir_parent="./bin/interim/" \
		--children_list="['pt_ETTh1_96', 'pt_ETTh2_96']" \
		--output_dir="./bin/interim/pt_etth_96/"

compile_pt_etth_192_data:
	$(PYTHON) task/compile_heteregenous_data.py \
		--input_dir_parent="./bin/interim/" \
		--children_list="['pt_ETTh1_192', 'pt_ETTh2_192']" \
		--output_dir="./bin/interim/pt_etth_192/"

compile_pt_etth_384_data:
	$(PYTHON) task/compile_heteregenous_data.py \
		--input_dir_parent="./bin/interim/" \
		--children_list="['pt_ETTh1_384', 'pt_ETTh2_384']" \
		--output_dir="./bin/interim/pt_etth_384/"

compile_pt_etth_768_data:
	$(PYTHON) task/compile_heteregenous_data.py \
		--input_dir_parent="./bin/interim/" \
		--children_list="['pt_ETTh1_768', 'pt_ETTh2_768']" \
		--output_dir="./bin/interim/pt_etth_768/"

preprocess_pt_etth_96: ID = pt_etth_96
preprocess_pt_etth_96:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_pt_etth_192: ID = pt_etth_192
preprocess_pt_etth_192:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_pt_etth_384: ID = pt_etth_384
preprocess_pt_etth_384:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_pt_etth_768: ID = pt_etth_768
preprocess_pt_etth_768:
	$(PIPELINE_PREPROCESS_PRE_TRAINING)

preprocess_pt: \
	preprocess_pt_etth_96 preprocess_pt_etth_192 preprocess_pt_etth_384 preprocess_pt_etth_768