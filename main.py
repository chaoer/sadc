import argparse
import yaml

from trainers.base_trainer import BaseTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='path to config file')

try:
    args = parser.parse_args()
except IOError as msg:
    parser.error(str(msg))

with open("configs/" + args.config, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

trainer = BaseTrainer(params)
trainer.train()
