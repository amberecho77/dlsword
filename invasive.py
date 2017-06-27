import argparse
import yaml
import os
import csv
import random
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
import time

from utils.trainer import Trainer
from utils.utils import get_loader, finetune
import models

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from sklearn.cross_validation import KFold
"""
Configuration
"""

parser = argparse.ArgumentParser(description='PyTorch Image Training')
parser.add_argument(
    '--config',
    default='./config/invasive.yaml',
    type=str,
    help='training configuration file')
parser.add_argument('--test', default=None, type=str, help='To test a model')
parser.add_argument(
    '--fold', default=4, type=int, help='K fold cross validation')
parser.add_argument(
    '--vis', dest='vis', action='store_true', help='To repeat test')
args = parser.parse_args()

if args.test is None:
    with open(args.config, 'r') as f:
        config = yaml.load(f)
else:
    logs_list = glob.glob(args.test + '*')
    with open(os.path.join(logs_list[-1], 'config.yaml'), 'r') as f:
        config = yaml.load(f)
"""
Data Loading
"""
root_dir = config['data_folder']
label_file = os.path.join(root_dir, 'train_labels.csv')
test_dir = os.path.join(root_dir, 'test')

imgs = []
with open(label_file, 'rb') as f:
    reader = csv.DictReader(f)
    for row in reader:
        img = os.path.join(root_dir, 'train', row['name'] + '.jpg')
        label = int(row['invasive'])
        imgs.append((img, label))

test_imgs = sorted(glob.glob(test_dir + '/*.jpg'))
test_list = [(item, 0) for item in test_imgs]
random.shuffle(imgs)

if args.test is None:
    kf = KFold(len(imgs), n_folds=args.fold)

    def train_session(idx, train_index, val_index):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(idx)
        train_list = [imgs[x] for x in train_index]
        val_list = [imgs[x] for x in val_index]
        train_loader = get_loader(train_list, config, 'train')
        val_loader = get_loader(val_list, config, 'val')
        model = models.__dict__[config['model']](pretrained=config['pretrain'])
        model, last_layer, feature_layer = finetune(model, config)
        model = torch.nn.DataParallel(model).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam([{
            'params': last_layer.parameters(),
            'lr': 1e-3
        }, {
            'params': feature_layer.parameters(),
            'lr': 1e-4
        }])
        trainer = Trainer(
            model,
            optimizer,
            criterion,
            config,
            train_loader,
            val_loader,
            regime=None)

        trainer.run()

    plist = [
        Process(target=train_session, args=(idx, train_index, val_index))
        for idx, (train_index, val_index) in enumerate(kf)
    ]

    for p in plist:
        p.start()
        time.sleep(1.5)

    for p in plist:
        p.join()

else:

    def test_session(log_folder, idx):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(idx)
        model = models.__dict__[config['model']]()
        model, last_layer, feature_layer = finetune(model, config)
        model = torch.nn.DataParallel(model).cuda()
        best_model = sorted(glob.glob(log_folder + '/model_best*'))[-1]
        checkpoint = torch.load(best_model)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        test_loader = get_loader(test_list, config, 'val')
        softmax = nn.Softmax()
        probs_list = []
        names_list = []
        with tqdm(total=len(test_loader)) as pbar:
            for i, (input, target, names) in enumerate(test_loader):
                input_var = torch.autograd.Variable(input, volatile=True)
                prob = softmax(model(input_var)).data.cpu().numpy()
                img_id = [item.split('/')[-1].split('.')[0] for item in names]
                names_list += img_id
                probs_list += [item for item in prob]
                pbar.update(1)
        result = np.array(probs_list)
        output = zip(names_list, result)

        with open(
                os.path.join(log_folder,
                             'result_%s.csv' % checkpoint['epoch']), 'w') as f:
            f.write('name,invasive\n')
            for name, prob in output:
                f.write('%s,%f\n' % (name, prob[1]))

    logs_list = glob.glob(args.test + '*')

    plist = [
        Process(target=test_session, args=(item, idx))
        for idx, item in enumerate(logs_list)
    ]

    for p in plist:
        p.start()
        time.sleep(2)

    for p in plist:
        p.join()

print "hoho"
