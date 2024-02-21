from tqdm import tqdm 
from utils import get_config_from_yaml, get_logger
import time 

cfg = get_config_from_yaml('cfg.yaml')
logger = get_logger(cfg.output_dir)
logger.info(cfg)

for epoch in range(cfg.train.epochs):
    print(f'do something in epoch{epoch}...')
    for _ in tqdm(range(5)):
        time.sleep(1)

logger.info(f'project done')
