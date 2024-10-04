import os
import shutil
import argparse
import time
import logging
from tools.dataset.manager import DatasetManager


def add_dataset(args):
    if args.dataset and os.path.isfile(args.dataset):
        path = DatasetManager().dataset_path
        shutil.move(args.dataset, path)
        return
    logging.error(f'Invalid dataset: {args.dataset}')


def train_model(args):
    start_time = time.time()

    logging.info(f'Starting training for model: {args.model}')

    if args.model == 'probability':
        from models.probability import train
        logging.info(f'Using "probability" model')
        train.main()

    elif args.model == 'arguments':
        from models.arguments import train
        logging.info('Using "arguments" model.')
        train.main()

    else:
        logging.error(f'Model {args.model} not recognized.')
        return

    logging.info(f'Training completed. Total time: {time.time() - start_time:.2f} seconds')


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(prog='ai-tool', description='CLI tool for AI training')
    subparsers = parser.add_subparsers(title='subcommands', description='valid subcommands', help='Additional help')

    parser_dataset = subparsers.add_parser('add', help='Add dataset')
    parser_dataset.add_argument('--dataset', required=True, help='Path to the training dataset')
    parser_dataset.set_defaults(func=add_dataset)

    parser_train = subparsers.add_parser('train', help='Train model')
    parser_train.add_argument('--model', choices=['probability', 'arguments'], required=True, help='Model to be trained')
    parser_train.set_defaults(func=train_model)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
