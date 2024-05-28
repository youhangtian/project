import time
import argparse
from tqdm import tqdm 
from src.utils import get_config_from_yaml, get_logger, cosine_scheduler, AverageMeter

def get_config():
    parser = argparse.ArgumentParser(description='training argument parser')
    parser.add_argument('-c', '--config_file', default='cfg.yaml', type=str, help='config file')
    parser.add_argument('opts', help='modify config options from the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = get_config_from_yaml(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze
    return cfg

def main(cfg):
    logger = get_logger(cfg.output_dir)
    logger.info(f'config:\n{cfg}')

    n = 10
    lr_scheduler = cosine_scheduler(0.1, 0, cfg.train.epochs, n)
    loss_am = AverageMeter()

    steps = 0
    for epoch in range(cfg.train.epochs):
        for i in tqdm(range(n), f'[epoch{epoch}]'):
            lr = lr_scheduler[steps]
            loss = steps * 0.1

            loss_am.update(loss)
            logger.info(f'epoch[{epoch}]steps[{steps}]: lr: {lr:.6f}, loss_avg: {loss_am.avg:.6f} ------')
            time.sleep(1)

            steps += 1

    logger.info(f'project done')

if __name__ == '__main__':
    cfg = get_config()
    main(cfg)
