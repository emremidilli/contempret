CONTAINER = contempret-app_contempret-1
PYTHON = docker exec -i $(CONTAINER) python

.PHONY: install run preprocess_pt preprocess_ft \
	preprocess_pt_etth_96 preprocess_pt_etth_192 preprocess_pt_etth_384 preprocess_pt_etth_768 \
	preprocess_ft_ETTh1_96_96 preprocess_ft_ETTh1_192_192 preprocess_ft_ETTh1_384_336 preprocess_ft_ETTh1_768_720 \
	preprocess_ft_ETTh2_96_96 preprocess_ft_ETTh2_192_192 preprocess_ft_ETTh2_384_336 preprocess_ft_ETTh2_768_720 \
	preprocess_ft_ETTm1_96_96 preprocess_ft_ETTm1_192_192 preprocess_ft_ETTm1_384_336 preprocess_ft_ETTm1_768_720 \
	preprocess_ft_ETTm2_96_96 preprocess_ft_ETTm2_192_192 preprocess_ft_ETTm2_384_336 preprocess_ft_ETTm2_768_720 \
	train_foundation_etth1_96 \
	train_pt_etth_96_wo_sequential_01 train_pt_etth_96_wo_sequential_02 train_pt_etth_96_wo_sequential_03 \
	train_pt_etth_96_wo_sequential_04 train_pt_etth_96_wo_sequential_05 train_pt_etth_96_wo_sequential_06 \
	train_pt_etth_96_wo_sequential_07 train_pt_etth_96_wo_sequential_08 train_pt_etth_96_wo_sequential_09 \
	train_pt_etth_96_wo_sequential_10

ID ?= pt_ETTh1_96

install:
	docker-compose down
	docker-compose build

run:
	docker-compose up -d

# ── Definitions ─────────────────────────────────────────────────

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

define PIPELINE_PREPROCESS_FINE_TUNING
	$(MAKE) run_preprocess_fine_tuning ID=$(ID)
endef

run_preprocess_fine_tuning: run
	$(PYTHON) task/pre_process_fine_tuning.py \
		--input_dir="./bin/interim/$(ID)" \
		--output_dir="./bin/preprocessed/$(ID)" \
		--pool_size_trend="24" \
		--sigma="1.96"

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

train_pt_tiny:
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
		--projection_head_units=8 \
		--warmup_steps=4000 \
		--scale_factor=0.1 \
		--mae_threshold_comp=0.75 \
		--mae_threshold_tre=0.75 \
		--mae_threshold_sea=0.75 \
		--cl_margin=0.25 \
		--patch_size=12 \
		--prompt_pool_size=0 \
		--nr_of_most_similar_prompts=0 \
		--patience=20 \
		--nr_of_seeds=1 \
		--nr_of_epochs=10000 \
		--w_comp=0.25 \
		--w_tre=0.75 \
		--w_sea=1.00 \
		--w_cl=1.00

train_ft_tiny:
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
		--mini_batch_size=128

# ── Foundation Models ───────────────────────────────────────────────────────────────────
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

hp_tune_pt_etth_96:
	$(PYTHON) task/tune_hyperparameters.py \
		--input_dir="./bin/preprocessed/pt_etth_96" \
		--output_dir="./bin/hp-tuning/pt_etth_96" \
		--mini_batch_size=128 \
		--max_epochs=27 \
		--executions_per_trial=1 \
		--factor=3 \
		--training_mode="sequential"

train_pt_etth_96:
	$(PYTHON) task/pre_train.py \
		--input_dir="./bin/preprocessed/pt_etth_96" \
		--output_dir="./bin/models/pt_etth_96" \
		--mask_rate=0.40 \
		--mask_scalar=0.00 \
		--mini_batch_size=128 \
		--clip_norm=0.10 \
		--l1_trend=0.0009069200356077791 \
		--l2_trend=0.00011516902230932779 \
		--l1_seasonality=6.986327036159758e-06 \
		--l2_seasonality=1.844456834877414e-05 \
		--l1_residual=0.0033743574546682815 \
		--l2_residual=0.00048162281165686684 \
		--nr_of_encoder_blocks=8 \
		--nr_of_heads=2 \
		--dropout_rate=0.1 \
		--encoder_ffn_units=192 \
		--embedding_dims=256 \
		--projection_head_units=16 \
		--warmup_steps=4000 \
		--scale_factor=0.1 \
		--mae_threshold_comp=0.16 \
		--mae_threshold_tre=0.09 \
		--mae_threshold_sea=0.07 \
		--cl_margin=0.25 \
		--patch_size=4 \
		--prompt_pool_size=40 \
		--nr_of_most_similar_prompts=7 \
		--patience=20 \
		--nr_of_seeds=10 \
		--nr_of_epochs=10000

train_pt_etth1_96:
	$(PYTHON) task/pre_train.py \
		--input_dir="./bin/preprocessed/pt_ETTh1_96" \
		--output_dir="./bin/models/pt_etth1_96" \
		--mask_rate=0.40 \
		--mask_scalar=0.00 \
		--mini_batch_size=128 \
		--clip_norm=0.10 \
		--l1_trend=0.0009069200356077791 \
		--l2_trend=0.00011516902230932779 \
		--l1_seasonality=6.986327036159758e-06 \
		--l2_seasonality=1.844456834877414e-05 \
		--l1_residual=0.0033743574546682815 \
		--l2_residual=0.00048162281165686684 \
		--nr_of_encoder_blocks=8 \
		--nr_of_heads=2 \
		--dropout_rate=0.1 \
		--encoder_ffn_units=192 \
		--embedding_dims=256 \
		--projection_head_units=16 \
		--warmup_steps=4000 \
		--scale_factor=0.1 \
		--mae_threshold_comp=0.16 \
		--mae_threshold_tre=0.09 \
		--mae_threshold_sea=0.07 \
		--cl_margin=0.25 \
		--patch_size=4 \
		--prompt_pool_size=40 \
		--nr_of_most_similar_prompts=7 \
		--patience=20 \
		--nr_of_seeds=10 \
		--nr_of_epochs=10000

# ── Fine-Tuning Models ───────────────────────────────────────────────────────────────────
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

train_ft_etth1_96_96:
	$(PYTHON) task/fine_tune.py \
		--input_dir="./bin/preprocessed/ft_ETTh1_96_96/" \
		--output_dir="./bin/models/ft_etth1_96_96/" \
		--pre_trained_model_dir="./bin/models/pt_etth_96/" \
		--patience=50 \
		--clip_norm=1.0 \
		--learning_rate=0.0001 \
		--warmup_epochs=100 \
		--tune_time2vec="True" \
		--nr_of_seeds=10 \
		--nr_of_epochs=10000 \
		--mini_batch_size=128



# ── Ablation Studies ───────────────────────────────────────────────────────────────────

PT_ETTH_96_WO_SEQ_ARGS = \
	--input_dir="./bin/preprocessed/pt_etth_96" \
	--mask_rate=0.40 \
	--mask_scalar=0.00 \
	--mini_batch_size=128 \
	--clip_norm=0.1 \
	--l1_trend=0.0009069200356077791 \
	--l2_trend=0.00011516902230932779 \
	--l1_seasonality=6.986327036159758e-06 \
	--l2_seasonality=1.844456834877414e-05 \
	--l1_residual=0.0033743574546682815 \
	--l2_residual=0.00048162281165686684 \
	--nr_of_encoder_blocks=8 \
	--nr_of_heads=2 \
	--dropout_rate=0.1 \
	--encoder_ffn_units=192 \
	--embedding_dims=256 \
	--projection_head_units=16 \
	--warmup_steps=4000 \
	--scale_factor=0.1 \
	--mae_threshold_comp=0.16 \
	--mae_threshold_tre=0.09 \
	--mae_threshold_sea=0.07 \
	--cl_margin=0.25 \
	--patch_size=4 \
	--prompt_pool_size=40 \
	--nr_of_most_similar_prompts=7 \
	--patience=20 \
	--nr_of_seeds=1 \
	--nr_of_epochs=5

train_pt_etth_96_wo_sequential_01:
	$(PYTHON) task/pre_train.py $(PT_ETTH_96_WO_SEQ_ARGS) \
		--output_dir="./bin/models/pt_etth_96_wo_sequential_01" \
		--w_comp=0.500 --w_tre=0.167 --w_sea=0.167 --w_cl=0.167

train_pt_etth_96_wo_sequential_02:
	$(PYTHON) task/pre_train.py $(PT_ETTH_96_WO_SEQ_ARGS) \
		--output_dir="./bin/models/pt_etth_96_wo_sequential_02" \
		--w_comp=0.167 --w_tre=0.500 --w_sea=0.167 --w_cl=0.167

train_pt_etth_96_wo_sequential_03:
	$(PYTHON) task/pre_train.py $(PT_ETTH_96_WO_SEQ_ARGS) \
		--output_dir="./bin/models/pt_etth_96_wo_sequential_03" \
		--w_comp=0.167 --w_tre=0.167 --w_sea=0.500 --w_cl=0.167

train_pt_etth_96_wo_sequential_04:
	$(PYTHON) task/pre_train.py $(PT_ETTH_96_WO_SEQ_ARGS) \
		--output_dir="./bin/models/pt_etth_96_wo_sequential_04" \
		--w_comp=0.167 --w_tre=0.167 --w_sea=0.167 --w_cl=0.500

train_pt_etth_96_wo_sequential_05:
	$(PYTHON) task/pre_train.py $(PT_ETTH_96_WO_SEQ_ARGS) \
		--output_dir="./bin/models/pt_etth_96_wo_sequential_05" \
		--w_comp=0.333 --w_tre=0.333 --w_sea=0.167 --w_cl=0.167

train_pt_etth_96_wo_sequential_06:
	$(PYTHON) task/pre_train.py $(PT_ETTH_96_WO_SEQ_ARGS) \
		--output_dir="./bin/models/pt_etth_96_wo_sequential_06" \
		--w_comp=0.333 --w_tre=0.167 --w_sea=0.333 --w_cl=0.167

train_pt_etth_96_wo_sequential_07:
	$(PYTHON) task/pre_train.py $(PT_ETTH_96_WO_SEQ_ARGS) \
		--output_dir="./bin/models/pt_etth_96_wo_sequential_07" \
		--w_comp=0.333 --w_tre=0.167 --w_sea=0.167 --w_cl=0.333

train_pt_etth_96_wo_sequential_08:
	$(PYTHON) task/pre_train.py $(PT_ETTH_96_WO_SEQ_ARGS) \
		--output_dir="./bin/models/pt_etth_96_wo_sequential_08" \
		--w_comp=0.167 --w_tre=0.333 --w_sea=0.333 --w_cl=0.167

train_pt_etth_96_wo_sequential_09:
	$(PYTHON) task/pre_train.py $(PT_ETTH_96_WO_SEQ_ARGS) \
		--output_dir="./bin/models/pt_etth_96_wo_sequential_09" \
		--w_comp=0.167 --w_tre=0.333 --w_sea=0.167 --w_cl=0.333

train_pt_etth_96_wo_sequential_10:
	$(PYTHON) task/pre_train.py $(PT_ETTH_96_WO_SEQ_ARGS) \
		--output_dir="./bin/models/pt_etth_96_wo_sequential_10" \
		--w_comp=0.167 --w_tre=0.167 --w_sea=0.333 --w_cl=0.333

train_pt_etth_96_wo_sequential:
	$(PYTHON) task/pre_train.py\
	--input_dir="./bin/preprocessed/pt_etth_96" \
	--output_dir="./bin/models/pt_etth_96_wo_sequential" \
	--w_comp=0.167 --w_tre=0.500 --w_sea=0.167 --w_cl=0.167 \
	--mask_rate=0.40 \
	--mask_scalar=0.00 \
	--mini_batch_size=128 \
	--clip_norm=0.1 \
	--l1_trend=0.0009069200356077791 \
	--l2_trend=0.00011516902230932779 \
	--l1_seasonality=6.986327036159758e-06 \
	--l2_seasonality=1.844456834877414e-05 \
	--l1_residual=0.0033743574546682815 \
	--l2_residual=0.00048162281165686684 \
	--nr_of_encoder_blocks=8 \
	--nr_of_heads=2 \
	--dropout_rate=0.1 \
	--encoder_ffn_units=192 \
	--embedding_dims=256 \
	--projection_head_units=16 \
	--warmup_steps=4000 \
	--scale_factor=0.1 \
	--mae_threshold_comp=0.16 \
	--mae_threshold_tre=0.09 \
	--mae_threshold_sea=0.07 \
	--cl_margin=0.25 \
	--patch_size=4 \
	--prompt_pool_size=40 \
	--nr_of_most_similar_prompts=7 \
	--patience=20 \
	--nr_of_seeds=10 \
	--nr_of_epochs=10000

train_ft_etth1_96_96_wo_sequential:
	$(PYTHON) task/fine_tune.py \
		--input_dir="./bin/preprocessed/ft_ETTh1_96_96/" \
		--output_dir="./bin/models/ft_etth1_96_96_wo_sequential/" \
		--pre_trained_model_dir="./bin/models/pt_etth_96_wo_sequential/" \
		--patience=50 \
		--clip_norm=1.0 \
		--learning_rate=0.0001 \
		--warmup_epochs=100 \
		--tune_time2vec="True" \
		--nr_of_seeds=10 \
		--nr_of_epochs=10000 \
		--mini_batch_size=128

# python task/fine_tune.py \
# 	--input_dir="./bin/preprocessed/ft_ETTh1_96_96/" \
# 	--output_dir="./bin/models/ft_etth1_96_96_02/" \
# 	--pre_trained_model_dir="./bin/models/pt_etth1_96/" \
# 	--patience=50 \
# 	--clip_norm=1.0 \
# 	--learning_rate=0.0001 \
# 	--warmup_epochs=100 \
# 	--tune_time2vec="True" \
# 	--nr_of_seeds=10 \
# 	--nr_of_epochs=10000 \
# 	--mini_batch_size=128

# python task/pre_train.py \
# 	--input_dir="./bin/preprocessed/pt_ETTh1_96" \
# 	--output_dir="./bin/models/pt_etth1_96_02" \
# 	--mask_rate=0.40 \
# 	--mask_scalar=0.00 \
# 	--mini_batch_size=128 \
# 	--clip_norm=0.10 \
# 	--l1_trend=0.0009069200356077791 \
# 	--l2_trend=0.00011516902230932779 \
# 	--l1_seasonality=6.986327036159758e-06 \
# 	--l2_seasonality=1.844456834877414e-05 \
# 	--l1_residual=0.0033743574546682815 \
# 	--l2_residual=0.00048162281165686684 \
# 	--nr_of_encoder_blocks=8 \
# 	--nr_of_heads=2 \
# 	--dropout_rate=0.1 \
# 	--encoder_ffn_units=192 \
# 	--embedding_dims=256 \
# 	--projection_head_units=16 \
# 	--warmup_steps=4000 \
# 	--scale_factor=0.1 \
# 	--mae_threshold_comp=0.16 \
# 	--mae_threshold_tre=0.09 \
# 	--mae_threshold_sea=0.07 \
# 	--cl_margin=0.25 \
# 	--patch_size=8 \
# 	--prompt_pool_size=40 \
# 	--nr_of_most_similar_prompts=7 \
# 	--patience=20 \
# 	--nr_of_seeds=10 \
# 	--nr_of_epochs=10000
