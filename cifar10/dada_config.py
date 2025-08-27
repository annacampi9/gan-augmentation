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
    beta1: float = 0.5
    beta2: float = 0.999
    gan_epochs: int = int(os.environ.get("DADA_GAN_EPOCHS", 200))   # warmup
    total_epochs: int = int(os.environ.get("DADA_TOTAL_EPOCHS", 700))
    samples_per_class: int = int(os.environ.get("DADA_COUNT", 200)) # low-data default
    num_classes: int = 10  # optionally infer from loader

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
    g_label_mode: str = "random"   # {"random","batch"}

    # Augmentation (enabled after gan_epochs)
    enable_aug_after_gan: bool = True
    aug_rotation: float = 20.0
    aug_width_shift: float = 0.1
    aug_height_shift: float = 0.1
    aug_horizontal_flip: bool = True

    # Sampling & checkpoints
    sample_grid_rows: int = 10
    sample_grid_cols: int = 10
    sample_every_epochs: int = 1
    nearest_neighbor_panel_every: int = 20
    checkpoint_every: int = 100

    # TF options
    mixed_precision: bool = False
    enable_xla: bool = False

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