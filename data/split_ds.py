from pathlib import Path
import argparse
import random

parser = argparse.ArgumentParser(description='split_ds')
parser.add_argument('--path', type=str, help='dataset main dir', required=True)
parser.add_argument('--split_pct', default=0.8, type=float, help='n_training/n_total ratio')
args = parser.parse_args()

main_dir = Path(args.path)
anno_dir = main_dir/'Annotations'
out_dir = main_dir/'ImageSets/Main'
if not out_dir.is_absolute(): out_dir = Path.cwd() / out_dir
out_dir.mkdir(parents=True, exist_ok=True)

test = open(out_dir/'test.txt', mode='w')
trainval = open(out_dir/'trainval.txt', mode='w')

# glob provides random walk
for label_file in anno_dir.glob('**/*.xml'):
    # True or False output based on probability
    relative_path = label_file.relative_to(anno_dir).with_suffix('').as_posix()
    if random.random() < args.split_pct:
        trainval.write(relative_path + '\n')
    else:
        test.write(relative_path + '\n')

test.close()
trainval.close()
