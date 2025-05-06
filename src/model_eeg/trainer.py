import argparse
import importlib
import os
import tensorflow as tf
import logging 
from train import train


def run(db, gpu, from_fold, to_fold, suffix='', random_seed=42):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logging.info(f"Using GPU: {physical_devices[0]}")
    else:
        logging.warning("No GPU found, running on CPU")

    config_file = os.path.join('config', f'{db}.py')
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found")
    
    spec = importlib.util.spec_from_file_location("*", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    output_dir = f'experiments/{db}{suffix}'
    os.makedirs(output_dir, exist_ok=True)

    if from_fold > to_fold:
        raise ValueError("from_fold must be less than or equal to to_fold")
    if to_fold >= config.train["n_folds"]:
        raise ValueError(f"to_fold ({to_fold}) exceeds n_folds ({config.train['n_folds']})")

    for fold_idx in range(from_fold, to_fold + 1):
        print(f"\n{'='*50}")
        print(f"Training fold {fold_idx} on GPU {gpu}")
        print(f"{'='*50}\n")

        train(
            config_file=config_file,
            fold_idx=fold_idx,
            output_dir=os.path.join(output_dir, 'models'),
            log_file=os.path.join(output_dir, f'fold_{fold_idx}.log'),
            restart=True,  
            random_seed=random_seed + fold_idx,
        )

        tf.keras.backend.clear_session()
        # if tf.config.experimental.get_memory_info('GPU:0')['current'] > 0:
        #     tf.config.experimental.reset_memory_stats('GPU:0')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-validation Trainer')
    parser.add_argument("--db", type=str, required=True, help="Dataset name")
    parser.add_argument("--gpu", type=int, required=True, help="GPU device ID")
    parser.add_argument("--from_fold", type=int, default=0, help="Start fold index")
    parser.add_argument("--to_fold", type=int, default=4, help="End fold index")
    parser.add_argument("--suffix", type=str, default='', help="Experiment suffix")
    parser.add_argument("--random_seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run(
        db=args.db,
        gpu=args.gpu,
        from_fold=args.from_fold,
        to_fold=args.to_fold,
        suffix=args.suffix,
        random_seed=args.random_seed,
    )