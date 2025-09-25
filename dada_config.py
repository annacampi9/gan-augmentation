import os


class DADAConfig:
    # Data
    dataset: str = os.environ.get("DADA_DATASET", "cifar10")  # or 'svhn'
    data_dir: str = os.environ.get("DADA_DATA_DIR", "data")
    results_dir: str = os.environ.get("DADA_RESULTS_DIR", "results")
    logs_dir: str = os.environ.get("DADA_LOGS_DIR", "logs")

    # Training
    seed: int = int(os.environ.get("DADA_SEED", 1))
    batch_size: int = int(os.environ.get("DADA_BATCH_SIZE", 100))
    learning_rate: float = float(os.environ.get("DADA_LR", 3e-4))
    # Separate learning rates (override above if set)
    g_learning_rate: float = float(os.environ.get("DADA_G_LR", 1.5e-4))
    d_learning_rate: float = float(os.environ.get("DADA_D_LR", 3e-4))
    beta1: float = 0.5
    beta2: float = 0.999
    gan_epochs: int = int(os.environ.get("DADA_GAN_EPOCHS", 200))  # warmup
    total_epochs: int = int(os.environ.get("DADA_TOTAL_EPOCHS", 700))
    samples_per_class: int = int(os.environ.get("DADA_COUNT", 200))  # low-data default
    num_classes: int = 10  # optionally infer from loader

    # Learning-rate schedule (epoch based)
    # Example to mimic original: base at 3e-4 until 300, then 3e-5, 3e-6, 3e-7, 3e-8
    lr_schedule_type: str = os.environ.get("DADA_LR_SCHEDULE", "step")  # {"constant","step"}
    lr_step_epochs: list[int] = [300, 400, 500, 600]
    lr_values_d: list[float] = [3e-4, 3e-5, 3e-6, 3e-7, 3e-8]
    lr_values_g: list[float] = [1.5e-4, 1.5e-5, 1.5e-6, 1.5e-7, 1.5e-8]

    # Model
    noise_dim: int = 100
    weight_norm_all: bool = False
    # Transfer learning for discriminator
    use_transfer_disc: bool = False
    transfer_backbone: str = os.environ.get("DADA_TRANSFER_BACKBONE", "ResNet50")
    transfer_freeze_epochs: int = int(os.environ.get("DADA_TRANSFER_FREEZE_EPOCHS", 10))

    # Loss schedule (DADA)
    warmup_w: float = 0.0
    post_w: float = 1.0
    feat_match_weight: float = 0.5
    g_updates_per_step_warmup: int = 2
    g_label_mode: str = "random"  # {"random","batch"}

    # Augmentation (enabled after gan_epochs)
    enable_aug_after_gan: bool = True
    # Original DADA uses pad+random-crop+flip; keep rotation/shifts off by default
    aug_rotation: float = 0.0
    aug_width_shift: float = 0.0
    aug_height_shift: float = 0.0
    aug_horizontal_flip: bool = True
    use_pad_and_crop: bool = True
    aug_pad: int = 4  # pixels

    # Hybrid augmentations (Mixup / CutMix)
    use_dada: bool = True
    use_mixup: bool = False
    use_cutmix: bool = False
    alpha: float = 1.0  # Beta distribution parameter for mixup/cutmix
    # Mixup/CutMix control (applies to supervised path; ramp suggested for low-data)
    hybrid_prob: float = float(
        os.environ.get("DADA_HYBRID_PROB", 0.5)
    )  # application probability per batch
    hybrid_start_epoch: int = int(os.environ.get("DADA_HYBRID_START", 220))
    hybrid_end_epoch: int = int(os.environ.get("DADA_HYBRID_END", 700))
    hybrid_on_supervised_only: bool = True

    # Sampling & checkpoints
    sample_grid_rows: int = 10
    sample_grid_cols: int = 10
    sample_every_epochs: int = 1
    nearest_neighbor_panel_every: int = 20
    checkpoint_every: int = 100

    # TF options
    mixed_precision: bool = False
    enable_xla: bool = False

    # EMA and label smoothing
    use_ema: bool = False
    ema_decay: float = 0.999
    label_smoothing: float = 0.0

    # Test mode (tiny run on CPU-friendly settings)
    test_mode: bool = os.environ.get("DADA_TEST", "0") == "1"
    test_samples_per_class: int = int(os.environ.get("DADA_TEST_SPC", 5))
    test_epochs: int = int(os.environ.get("DADA_TEST_EPOCHS", 2))
    test_gan_epochs: int = int(os.environ.get("DADA_TEST_GAN_EPOCHS", 1))
    test_batch_size: int = int(os.environ.get("DADA_TEST_BS", 8))
    test_max_batches_per_epoch: int = int(os.environ.get("DADA_TEST_MAXB", 3))
    test_sample_grid_rows: int = 2
    test_sample_grid_cols: int = 2


cfg = DADAConfig()
