import argparse
import sys


def get_args():
    """
    Parses arguments
    """
    parser = argparse.ArgumentParser(
        description="Time Series Forecasting Arguments"
    )

    # Shared args
    parser.add_argument("--input_dir", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="test")
    parser.add_argument("--mini_batch_size", type=int, default=128)
    parser.add_argument("--nr_of_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--warmup_epochs_early_stopping",
                        type=int, default=10)
    parser.add_argument("--nr_of_seeds", type=int, default=1)

    # Pre-train specific
    parser.add_argument("--prompt_dir", type=str, default=None)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--l1_trend", type=float, default=0.01)
    parser.add_argument("--l2_trend", type=float, default=0.01)
    parser.add_argument("--l1_seasonality", type=float, default=0.01)
    parser.add_argument("--l2_seasonality", type=float, default=0.01)
    parser.add_argument("--l1_residual", type=float, default=0.01)
    parser.add_argument("--l2_residual", type=float, default=0.01)
    parser.add_argument("--nr_of_encoder_blocks", type=int, default=1)
    parser.add_argument("--nr_of_heads", type=int, default=1)
    parser.add_argument("--encoder_ffn_units", type=int, default=8)
    parser.add_argument("--embedding_dims", type=int, default=8)
    parser.add_argument("--projection_head", type=int, default=8)
    parser.add_argument("--dropout_rate", type=float, default=0.10)
    parser.add_argument("--mask_scalar", type=float, default=0.00)
    parser.add_argument("--mask_rate", type=float, default=0.40)
    parser.add_argument("--mae_threshold_comp", type=float, default=0.50)
    parser.add_argument("--mae_threshold_tre", type=float, default=0.50)
    parser.add_argument("--mae_threshold_sea", type=float, default=0.05)
    parser.add_argument("--cl_margin", type=float, default=0.25)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--prompt_pool_size", type=int, default=2)
    parser.add_argument("--nr_of_most_similar_prompts", type=int, default=1)
    parser.add_argument("--force_mae_comp", type=int,
                        choices=[-1, 0, 1], default=0)
    parser.add_argument("--force_mae_tre", type=int,
                        choices=[-1, 0, 1], default=0)
    parser.add_argument("--force_mae_sea", type=int,
                        choices=[-1, 0, 1], default=0)
    parser.add_argument("--force_cl", type=int,
                        choices=[-1, 0, 1], default=0)
    parser.add_argument("--save_only_light_artifacts",
                        type=eval, default="False")
    parser.add_argument("--prefer_dense_to_time2vec",
                        type=eval, default="False")

    # Fine-tune specific
    parser.add_argument("--pre_trained_model_dir", type=str,
                        default="s3://time-series-forecasting-"
                                "media/training/test_fine_tuning/")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--task_type", type=str,
                        choices=["sequence_prediction",
                                 "time_series_classification"],
                        default="sequence_prediction")
    parser.add_argument("--tune_time2vec", type=eval, default="False")

    # Hyperparameter tuning
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--executions_per_trial", type=int, default=1)

    # Pre-process
    parser.add_argument("--pool_size_trend", type=int, default=24)
    parser.add_argument("--sigma", type=float, default=1.96)
    parser.add_argument("--use_timestamp", type=eval, default="True")

    # Sync data
    parser.add_argument("--uri", type=str, default="test")
    parser.add_argument("--local_dir", type=str, default="test")
    parser.add_argument("--in_out", type=str,
                        choices=["in", "out"], default="in")

    # Evaluate
    parser.add_argument("--model_dir", type=str, default="test")

    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        sys.exit(0)

    return args
