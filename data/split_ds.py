from collections import defaultdict
from pathlib import Path
import argparse
import random

parser = argparse.ArgumentParser(description='split_ds')
parser.add_argument('--path', type=str, help='dataset main dir', required=True)
parser.add_argument('--split_pct', default=0.8, type=float, help='n_training/n_total ratio')
parser.add_argument('--folder_max', default=300, type=int, help='maximum number of files taken from each folder')
args = parser.parse_args()

main_dir = Path(args.path)
anno_dir = main_dir/'Annotations'
out_dir = main_dir/'ImageSets/Main'
if not out_dir.is_absolute(): out_dir = Path.cwd() / out_dir
out_dir.mkdir(parents=True, exist_ok=True)

test = open(out_dir/'test.txt', mode='w')
trainval = open(out_dir/'trainval.txt', mode='w')

# glob provides random walk
catalogs = defaultdict(lambda : defaultdict(list))


for label_file in anno_dir.glob('**/*.xml'):
    # True or False output based on probability
    relative_path = label_file.relative_to(anno_dir)
    parent = relative_path.parent or 'root'
    file = (parent/relative_path.stem).as_posix()
    if random.random() < args.split_pct:
        # trainval.write(relative_path + '\n')
        catalogs[parent]['trainval'].append(file)
    else:
        # test.write(relative_path + '\n')
        catalogs[parent]['test'].append(file)

for catalog, files in catalogs.items():
    trainval_files = files['trainval'][:int(args.folder_max*args.split_pct)]
    test_files = files['test'][:int(args.folder_max*(1-args.split_pct))]
    trainval.write('\n'.join(trainval_files)+'\n')
    test.write('\n'.join(test_files)+'\n')
test.close()
trainval.close()
