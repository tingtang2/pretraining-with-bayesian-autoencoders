import argparse
import logging
import sys
from datetime import date

import torch
from torch.nn import MSELoss
from torch.optim import AdamW, Adam

from trainers.vae_trainer import (
    VAENotMNIST2MNISTTrainer, VAENoPretrainingMNIST,
    VAENoPretrainingFashionMNIST, VAENotMNIST2FashionMNISTTrainer,
    VAEFashionMNIST2notMNISTTrainer, VAENoPretrainingNotMNIST,
    VAEFashionMNIST2MNISTTrainer)

from trainers.vae_color_trainer import VAENoPretrainingCIFAR10Trainer, VAETinyImageNetPreTrainer

from trainers.sa_vae_trainer import SA_VAENotMNIST2MNISTTrainer

arg_trainer_map = {
    'vae': VAENotMNIST2MNISTTrainer,
    'vae_fashion': VAENotMNIST2FashionMNISTTrainer,
    'not_pretrained_vae': VAENoPretrainingMNIST,
    'not_pretrained_vae_fashion': VAENoPretrainingFashionMNIST,
    'vae_fashion_2_not': VAEFashionMNIST2notMNISTTrainer,
    'not_pretrained_vae_not': VAENoPretrainingNotMNIST,
    'vae_fashion_2_mnist': VAEFashionMNIST2MNISTTrainer,
    'not_pretrained_vae_cifar10': VAENoPretrainingCIFAR10Trainer,
    'pretrain_vae_tiny_imagenet': VAETinyImageNetPreTrainer,
    'sa_vae_not_2_mnist': SA_VAENotMNIST2MNISTTrainer
}
arg_optimizer_map = {'adamw': AdamW, 'adam': Adam}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=
        'Create and run pretraining and finetuning experiments with Bayesian autoencoders'
    )

    parser.add_argument('--pretrain_epochs',
                        default=100,
                        type=int,
                        help='number of epochs to pretrain model')
    parser.add_argument('--finetune_epochs',
                        default=50,
                        type=int,
                        help='number of epochs to finetune model')
    parser.add_argument('--device',
                        '-d',
                        default='cuda',
                        type=str,
                        help='cpu or gpu ID to use')
    parser.add_argument('--batch_size',
                        default=128,
                        type=int,
                        help='mini-batch size used to train model')
    parser.add_argument('--dropout_prob',
                        default=0.15,
                        type=float,
                        help='probability for dropout layers')
    parser.add_argument('--save_dir',
                        default='/home/tingchen/bayes-ae-save/',
                        help='path to saved model files')
    parser.add_argument('--data_dir',
                        default='/home/tingchen/data/',
                        help='path to data files')
    parser.add_argument('--optimizer',
                        default='adam',
                        help='type of optimizer to use')
    parser.add_argument('--learning_rate',
                        default=1e-3,
                        type=float,
                        help='learning rate for optimizer')
    parser.add_argument('--svi_learning_rate_mu',
                        default=1,
                        type=float,
                        help='learning rate for SVI optimizer mean')
    parser.add_argument('--svi_learning_rate_sigma',
                        default=1,
                        type=float,
                        help='learning rate for SVI optimizer std deviation')
    parser.add_argument('--max_grad_norm',
                        default=5,
                        type=float,
                        help='norm threshold to get clipped at for SA VAE')
    parser.add_argument('--model_type',
                        default='vae',
                        help='type of model to use')
    parser.add_argument('--pretraining_inference_type',
                        default='vae',
                        help='inference type for pretraining')
    parser.add_argument('--fine_tuning_inference_type',
                        default='map',
                        help='inference type for fine tuning')
    parser.add_argument('--num_repeats',
                        default=3,
                        type=int,
                        help='number of times to repeat experiment')
    parser.add_argument('--seed',
                        default=11202022,
                        type=int,
                        help='random seed to be used in numpy and torch')
    parser.add_argument('--perform_pretrain',
                        action='store_true',
                        help='pretrain model')
    parser.add_argument('--perform_finetune',
                        action='store_true',
                        help='finetune model on dataset')
    parser.add_argument('--bayesian_encoder',
                        action='store_true',
                        help='whether to use a bayesian NN encoder')
    parser.add_argument('--bayesian_decoder',
                        action='store_true',
                        help='whether to use a bayesian NN decoder')
    parser.add_argument('--experiment_name',
                        default="experiment",
                        type=str,
                        help='experiment name for the purposes of saving files')


    args = parser.parse_args()
    configs = args.__dict__

    if 'not_pretrained' in configs['model_type']:
        configs['pretraining_inference_type'] = 'none'
        configs['perform_pretrain'] = False

    # for repeatability
    torch.manual_seed(configs['seed'])

    # set up logging
    filename = f'{configs["experiment_name"]}-{date.today()}'
    # filename = f'{configs["model_type"]}-pretraining-{configs["pretraining_inference_type"]}-fine_tuning-{configs["fine_tuning_inference_type"]}-{date.today()}'
    FORMAT = '%(asctime)s;%(levelname)s;%(message)s'
    logging.basicConfig(level=logging.DEBUG,
                        filename=f'{configs["save_dir"]}logs/{filename}.log',
                        filemode='a',
                        format=FORMAT)
    logging.info(configs)

    # get trainer
    trainer_type = arg_trainer_map[configs['model_type']]
    trainer = trainer_type(
        optimizer_type=arg_optimizer_map[configs['optimizer']],
        criterion=MSELoss(reduction='sum'),
        **configs)

    # perform experiment n times
    #for iter in range(configs['num_repeats']):
    if trainer.perform_pretrain:
        trainer.pretrain()
    if trainer.perform_finetune:
        trainer.finetune()

    return 0


if __name__ == '__main__':
    sys.exit(main())