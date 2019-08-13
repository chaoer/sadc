import argparse
import yaml

from trainers.base_trainer import BaseTrainer
from trainers.rgb_only_trainer import RGBOnlyTrainer
from trainers.simple_trainer import SimpleTrainer
from trainers.local_global_trainer import LocalGlobalTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='path to config file')

try:
    args = parser.parse_args()
except IOError as msg:
    parser.error(str(msg))

with open("configs/" + args.config, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

if params["type"] == "simple":
    trainer = SimpleTrainer(params)
elif params["type"] == "local_global":
    trainer = LocalGlobalTrainer(params)
elif params["type"] == "rgb":
    trainer = RGBOnlyTrainer(params)
trainer.train()
