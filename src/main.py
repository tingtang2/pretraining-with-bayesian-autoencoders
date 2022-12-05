import argparse
import torch
import logging
import sys
from datetime import date


def main() -> int:
    parser = argparse.ArgumentParser(
        description=
        'Create and run active learning experiments on 5 prime splicing data')
    parser.add_argument('--epochs',
                        default=300,
                        type=int,
                        help='number of epochs to train model')
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
                        help='probability for dropout before dense layers')
    parser.add_argument('--save_dir',
                        default='active-learning-save/saved_metrics/',
                        help='path to saved metric files')
    parser.add_argument(
        '--log_save_dir',
        default='/home/tingchen/active-learning-save/active-learning-logs/',
        help='path to saved log files')
    parser.add_argument('--optimizer',
                        default='adamw',
                        help='type of optimizer to use')
    parser.add_argument('--num_repeats',
                        default=3,
                        type=int,
                        help='number of times to repeat experiment')
    parser.add_argument('--seed',
                        default=11202022,
                        type=int,
                        help='random seed to be used in numpy and torch')

    args = parser.parse_args()
    configs = args.__dict__

    # for repeatability
    torch.manual_seed(configs['seed'])

    # set up logging
    filename = f'al-{configs["model_type"]}-{configs["pretraining_inference"]}-{configs["fine_tuning_inference"]}-{date.today()}'
    FORMAT = '%(asctime)s;%(levelname)s;%(message)s'
    logging.basicConfig(level=logging.DEBUG,
                        filename=f'{configs["log_save_dir"]}{filename}.log',
                        filemode='a',
                        format=FORMAT)
    logging.info(configs)

    # get trainer
    # trainer_type, model_type = arg_model_trainer_map[
    #     configs['acquisition_fn_type']]
    # trainer = trainer_type(
    #     model_type=model_type,
    #     optimizer_type=arg_optimizer_map[configs['optimizer']],
    #     criterion=MSELoss(),
    #     **configs)

    # trainer.load_data()

    # # perform experiment n times
    # for iter in range(configs['num_repeats']):
    #     trainer.active_train_loop(iter)

    return 0


if __name__ == '__main__':
    sys.exit(main())