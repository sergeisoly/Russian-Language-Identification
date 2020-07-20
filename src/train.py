import pandas as pd
import numpy as np
import numpy
import io
import os
import re
import shutil
from typing import Tuple

import tqdm
from collections import Counter

import argparse
import torch
from torchtext.data import Iterator
from torch.optim.lr_scheduler import LambdaLR
import yaml
import math
import datetime

from torch.optim import adam
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torchtext.data import Field, NestedField
import torchtext.data as data
from torchtext.data.dataset import Dataset
from torchtext.data.example import Example
from typing import Iterable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.data import load_data
from src.model import GRUIdentifier
from src.data import PAD_TOKEN

def save_model(output_dir, state, filename='best_model.pth.tar'):
    path = os.path.join(output_dir, filename)
    torch.save(state, path)

def test(model, testing_data : Iterator, output_matrix : bool=False, level: str='char') -> float:

    model.eval()
    classes = testing_data.dataset.fields['language'].vocab.itos
    n_classes = len(classes)
    confusion_matrix = numpy.zeros((n_classes, n_classes))
    sparse_matrix = Counter()

    for j, batch in enumerate(iter(testing_data)):

        sequence = batch.characters[0]
        lengths = batch.characters[1]
        target = batch.language
       
        predictions = model.forward(sequence, batch.characters[1])

        predicted_languages = torch.tensor([[torch.where(torch.tensor([pred[0] > 0.5]), torch.ones(1, dtype=torch.int16), torch.zeros(1, dtype=torch.int16))] for pred in predictions])

        # print(predicted_languages, '\n', target)
        for p, t in zip(predicted_languages, target):
            # print(p, t)
            confusion_matrix[p][t] += 1
            sparse_matrix[(classes[p],classes[t])] += 1
    # print(confusion_matrix)
    tn = confusion_matrix[0][0]
    fn = confusion_matrix[0][1]
    tp = confusion_matrix[1][1]
    fp = confusion_matrix[1][0]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    acc = (tp+tn)/(tp+fp+fn+tn)
    f1_score = 2*(precision*recall)/(precision + recall)


    if output_matrix:
        with open("confusion_matrix.txt", 'w') as f:
            f.write("\t")
            f.write("\t".join(classes))
            f.write("\n")
            for i, line in enumerate(confusion_matrix):
                f.write("{}\t".format(classes[i]))
                f.write("\t".join(map(str, line)))
                f.write("\n")

        with open("sparse_matrix.txt", 'w') as f:
            for lan, score in sparse_matrix.most_common():
                f.write("{} - {} : {}\n".format(lan[0], lan[1], score))

    return (precision, recall, acc, f1_score)


def train(par_optimizer, model,
          training_data: Iterator=None, validation_data: Iterator=None, testing_data: Iterator=None,
          learning_rate: float=1e-3, epochs: int=0, resume_state: dict=None, resume: str="",
          log_frequency: int=0, eval_frequency: int=0, model_type="", output_dir="", scheduler=None,
          level: str='char',
          **kwargs):

    # get command line arguments
    cfg = locals().copy()

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    best_metric, best_test_f1_score= 0, 0
    start_epoch = 0

    print(datetime.datetime.now(), " Training starts.")
    for i in range(start_epoch, epochs):

        if scheduler:
            scheduler.step()
     
        epoch_losses = []
        bar = tqdm.notebook.tqdm(total=len(training_data), desc=f'Epoch {i}', position=0)
        for j, batch in enumerate(iter(training_data)):
            model.train()
            par_optimizer.zero_grad()

            # We take the characters as input to the network, and the languages
            # as target
            sequence = batch.characters[0]
            lengths = batch.characters[1]
            target = batch.language
            # NLLLoss for using the log_softmax in the recurrent model
            pad_idx = training_data.dataset.fields['characters'].vocab.stoi[PAD_TOKEN]
            w = torch.tensor([5], device=device, dtype=torch.int16)
            # loss_function = torch.nn.NLLLoss(weight=w, ignore_index=pad_idx)
            loss_function = torch.nn.BCELoss(weight=w)

            # characters = batch.characters[0]
            # languages = batch.language
            batch_size = sequence.shape[0]

            predictions = model.forward(sequence, batch.characters[1])
            
            loss = loss_function(predictions.squeeze(1).float(), target.squeeze(1).float())
            epoch_losses.append(loss.item())


            # Update the weights
            loss.backward()
            par_optimizer.step()

            if (j + 1) % cfg["log_frequency"] == 0:
                if scheduler:
                    lr = scheduler.get_lr()[0]
                else:
                    lr = cfg["learning_rate"]
                t = datetime.datetime.now().time()
                print("{}:{:02d}:{:02d} Logging: Epoch: {} | Iter: {} | Loss: {} "
                                               "| LR: {}".format(t.hour, t.minute, t.second,
                    i, j, round(loss.item(), 4), round(lr, 5))
                      )

            if (j + 1) % cfg["eval_frequency"] == 0:

                precision, recall, acc, f1_score = test(model, validation_data, level=level)
                t = datetime.datetime.now().time()
                print("{}:{:02d}:{:02d} Evaluation: Epoch: {} | Iter: {} | Loss: {} | "
                      "precision: {} | recall: {} | f1: {} | accuracy: {}".format(t.hour, t.minute, t.second,
                    i, j, round(loss.item(), 4), round(precision, 3), round(recall, 3), round(f1_score, 3), round(acc, 3)))
                if f1_score > best_metric:
                    test_precision, test_recall, test_acc, test_f1_score = test(model, testing_data, True, level=level)
                    best_metric = f1_score
                    best_test_f1_score = test_f1_score
                    output_dir = cfg["output_dir"]
                    save_model(output_dir,
                               {
                                   'epoch': i,
                                   'state_dict': model.state_dict(),
                                   'val_f1_score': f1_score,
                                   'test_f1_score': test_f1_score,
                                    'test_precision': test_precision,
                                  'test_recall': test_recall,
                                  'test_acc': test_acc,
                                   'optimizer': par_optimizer.state_dict(),
                               },
                               filename=cfg["model_type"] + "_best_model.pth.tar")
            bar.update(1)


        precision, recall, acc, f1_score = test(model, validation_data, level=level)
        t = datetime.datetime.now().time()
        print("{}:{:02d}:{:02d} Epoch: {} finished | Average loss: {} | "
              "precision: {} | recall: {} | f1: {} | accuracy: {}".format(t.hour, t.minute, t.second,
              i + 1, round(np.mean(np.array(epoch_losses)), 2),round(precision, 3), round(recall, 3), round(f1_score, 3), round(acc, 3)))
    print(datetime.datetime.now(), " Done training.")
    print("Best model: Validation f1_score: {} | Test f1_score: {}".format(
          round(best_metric, 3), round(best_test_f1_score, 3)))


def main():

    ap = argparse.ArgumentParser(description="a Language Identification model")

    ap.add_argument('--output_dir', type=str, default='output')
    ap.add_argument('--config', type=str, default=None)

    cfg = vars(ap.parse_args())

    if cfg["config"] is not None:
        with open(cfg["config"]) as f: yaml_config = yaml.load(f)
        cfg.update(yaml_config)
    
    for k, v in cfg.items():
        print("  %12s : %s" % (k, v))

    print()

    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')

    # Load datasets and create iterators to use while training / testing
    training_data, validation_data, testing_data = load_data(**cfg)
    print("Data loaded.")

    training_iterator = Iterator(training_data, cfg['batch_size'], train=True, shuffle=True,
                                        sort_within_batch=True, device=device, repeat=False)

    validation_iterator = Iterator(validation_data, cfg['batch_size'], train=False, sort_within_batch=True,
                                        device=device, repeat=False)
    testing_iterator = Iterator(testing_data, cfg['batch_size'], train=False,
                                    sort_within_batch=True, device=device, repeat=False)

    print("Loaded %d training samples" % len(training_data))
    print("Loaded %d validation samples" % len(validation_data))
    print("Loaded %d test samples" % len(testing_data))
    print()

    char_vocab_size = len(training_data.fields['characters'].vocab)
    # n_classes = len(training_data.fields['language'].vocab)
    n_classes = 1

    model = GRUIdentifier(char_vocab_size, n_classes, **cfg)

    print("Vocab. size word: ", len(training_data.fields['paragraph'].vocab))
    print("First 10 words: ", " ".join(training_data.fields['paragraph'].vocab.itos[:10]))
    print("Vocab. size chars: ", len(training_data.fields['characters'].vocab))
    print("First 10 chars: ", " ".join(training_data.fields['characters'].vocab.itos[:10]))
    print("Number of languages: ", n_classes)
    print()

    par_optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = None

    print("Trainable Parameters")
    n_params = sum([np.prod(par.size()) for par in model.parameters() if par.requires_grad])

    print("Number of parameters: {}".format(n_params))
    for name, par in model.named_parameters():
        if par.requires_grad:
            print("{} : {}".format(name, list(par.size())))
    print()

    if use_cuda: model.cuda()

    train(model=model,
            training_data=training_iterator, validation_data=validation_iterator, testing_data=testing_iterator,
            par_optimizer=par_optimizer, scheduler=scheduler, resume_state=False,
        **cfg)

if __name__ == "__main__":
    main()
