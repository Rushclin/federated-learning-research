import os
import sys
import time
import torch
import argparse
import traceback

from importlib import import_module
from torch.utils.tensorboard import SummaryWriter

from src import Range, set_logger, check_args, set_seed, load_dataset, load_model, tensorboard_runner


def main(args, writer):

    set_seed(args.seed)

    server_dataset, client_datasets = load_dataset(args)

    args = check_args(args)

    model, args = load_model(args)

    server_class = import_module(f'src.server.{args.algorithm}server').__dict__[
        f'{args.algorithm.title()}Server']
    server = server_class(args=args, writer=writer, server_dataset=server_dataset,
                          client_datasets=client_datasets, model=model)

    for curr_round in range(1, args.R + 1):
        server.round = curr_round

        selected_ids = server.update()

        if (curr_round % args.eval_every == 0) or (curr_round == args.R):
            server.evaluate(excluded_ids=selected_ids)
    else:
        server.finalize()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '--exp_name', help='Nom de l\'experience', type=str, required=True)
    parser.add_argument('--seed', help='Seed global', type=int, default=5959)
    parser.add_argument(
        '--device', help='Peripherique à utiliser, `cuda`, `cuda:GPU_NUMBER`', type=str, default='cpu')
    parser.add_argument('--log_path', help='Chemin des logs',
                        type=str, default='./log')
    parser.add_argument(
        '--result_path', help='Chemin pour sauvegarder les resultats', type=str, default='./result')
    parser.add_argument('--use_tb', help='TensorBoard', action='store_true')
    parser.add_argument('--tb_port', help='TensorBoard',
                        type=int, default=6006)
    parser.add_argument('--tb_host', help='TensorBoard',
                        type=str, default='0.0.0.0')

    parser.add_argument(
        '--dataset', help='''Nom du dataset, pas très important''', type=str, required=True)
    parser.add_argument('--test_size', help='Fraction à assigner pour les tests',
                        type=float, choices=[Range(-1, 1.)], default=0.2)

    parser.add_argument('--resize', help='resize des images',
                        type=int, default=None)
    parser.add_argument('--crop', help='crop des images',
                        type=int, default=None)
    parser.add_argument(
        '--imnorm', help='normalized es images', action='store_true')
    parser.add_argument(
        '--randrot', help='randomly rotate sur les images', type=int, default=None)
    parser.add_argument('--randhf', help='randomly flip sur les images',
                        type=float, choices=[Range(0., 1.)], default=None)
    parser.add_argument('--randvf', help='randomly flip sur les images',
                        type=float, choices=[Range(0., 1.)], default=None)
    parser.add_argument('--randjit', help='randomly change tsur les images',
                        type=float, choices=[Range(0., 1.)], default=None)

    # statistical heterogeneity simulation arguments
    parser.add_argument('--split_type', help='''Type de scenario de decoupage
    - `iid`
    - `non-iid`
    ''', type=str, choices=['iid', 'non-iid'], required=True)

    parser.add_argument('--model_name', help='', type=str,
                        choices=[
                            'TwoNN', 'TwoCNN',
                            'VGG9', 'VGG9BN', 'VGG11', 'VGG11BN', 'VGG13', 'VGG13BN',
                            'ResNet10', 'ResNet18', 'ResNet34',
                        ],
                        required=True
                        )
    parser.add_argument(
        '--hidden_size', help='Nombre de couche cachées', type=int, default=64)
    parser.add_argument('--dropout', help='dropout',
                        type=float, choices=[Range(0., 1.)], default=0.1)
    parser.add_argument(
        '--num_layers', help='number of layers in recurrent cells', type=int, default=2)
    parser.add_argument(
        '--num_embeddings', help='size of an embedding layer', type=int, default=1000)
    parser.add_argument(
        '--embedding_size', help='output dimension of an embedding layer', type=int, default=512)
    parser.add_argument('--init_type', help='', type=str, default='xavier', choices=[
                        'normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'truncnorm', 'none'])
    parser.add_argument('--init_gain', type=float, default=1.0, help='')

    parser.add_argument('--algorithm', help='Algorithme à utiliser', type=str,
                        choices=['fedavg', 'fedsecure', 'fedprox',],
                        required=True
                        )
    parser.add_argument('--eval_type', help='''- `local` - `global` - 'both' ''', type=str,
                        choices=['local', 'global', 'both'],
                        required=True
                        )
    parser.add_argument('--eval_fraction', help='', type=float,
                        choices=[Range(1e-8, 1.)], default=1.)
    parser.add_argument('--eval_every', help='', type=int, default=1)
    parser.add_argument('--eval_metrics', help='Metriques', type=str,
                        choices=[
                            'acc1',  'f1', 'precision', 'recall'
                        ], nargs='+', required=True
                        )
    parser.add_argument('--K', help='Nombre de client', type=int, default=100)
    parser.add_argument(
        '--R', help='Nombre de tour de communication', type=int, default=1000)
    parser.add_argument('--C', help='Fration de client',
                        type=float, choices=[Range(0., 1.)], default=0.1)
    parser.add_argument(
        '--E', help='Nombre d\'epoque locale', type=int, default=5)
    parser.add_argument('--B', help='Taille des lots', type=int, default=10)
    parser.add_argument('--beta1', help='Momentum, plus pour FedProx',
                        type=float, choices=[Range(0., 1.)], default=0.)

    parser.add_argument('--no_shuffle', help='', action='store_true')
    parser.add_argument('--optimizer', help='Optimisateur',
                        type=str, default='SGD', required=True)
    parser.add_argument('--max_grad_norm', help='', type=float,
                        choices=[Range(0., float('inf'))], default=0.)
    parser.add_argument('--weight_decay', help='Pour corriger le L2',
                        type=float, choices=[Range(0., 1.)], default=0)
    parser.add_argument('--momentum', help='momentum factor',
                        type=float, choices=[Range(0., 1.)], default=0.)
    parser.add_argument('--lr', help='Learning ratet', type=float,
                        choices=[Range(0., 100.)], default=0.01, required=True)
    parser.add_argument('--lr_decay', help='Decay Learning rate',
                        type=float, choices=[Range(0., 1.)], default=1.0)
    parser.add_argument('--lr_decay_step',
                        help='Interval of Learning rate', type=int, default=20)
    parser.add_argument(
        '--criterion', help='Fonction objectif ou d\'optimisation', type=str, required=True)
    parser.add_argument('--mu', help='Terme de regulatisation pour FedProx',
                        type=float, choices=[Range(0., 1e6)], default=0.01)

    parser.add_argument('--input_folder', help='', type=str, default="data")
    parser.add_argument('--output_folder', help='',
                        type=str, default="dataset")
    parser.add_argument('--train_ratio', help='', type=float, default=0.8)
    parser.add_argument('--num_classes', help='', type=int, default=4)
    parser.add_argument('--in_channels', help='', type=int, default=3)
    parser.add_argument('--train', type=bool, default=True)

    # Methode de securisation secure_mechanism  
    parser.add_argument('--dp_mechanism', type=str, default="Laplace") # Ou Gaussian
    parser.add_argument('--dp_epsilon', type=float, default=10) 
    parser.add_argument('--dp_clip', type=float, default=10)
    parser.add_argument('--dp_sample', type=float, default=0.01) 
    parser.add_argument('--max_physical_batch_size', type=float, default=11) 

    args = parser.parse_args()

    curr_time = time.strftime("%y%m%d_%H%M%S", time.localtime())
    args.result_path = os.path.join(
        args.result_path, f'{args.exp_name}_{curr_time}')
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    set_logger(f'{args.log_path}/{args.exp_name}_{curr_time}.log', args)

    writer = SummaryWriter(log_dir=os.path.join(
        args.log_path, f'{args.exp_name}_{curr_time}'), filename_suffix=f'_{curr_time}')
    # tensorboard_runner(args)

    try:
        main(args, writer)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
