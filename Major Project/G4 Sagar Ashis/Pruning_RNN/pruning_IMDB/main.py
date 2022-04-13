import argparse
import pickle
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from tqdm import tqdm

import model
import utils
from utils import PadCollate, RunningAverage, TextDataset, TextSampler
from pruner import ModelPruner
import math
import warnings
import tensorflow as tf




parser = argparse.ArgumentParser(description='PyTorch IMDB classifier with pruning')

parser.add_argument('--load_ds', action='store_true',help='Load dataset from disk')
parser.add_argument('--emsize', type=int, default=300, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=128, help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--l2alpha', type=float, default=0, help='weights l2 regularization term coefficient')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=5,  help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('--seed', type=int, default=42, help='random seed for reprodusability')
parser.add_argument('--bptt', type=int, default=70, help='sequence length')
parser.add_argument('--save', type=str, default='data/models/model.pt', help='path to save the final model')
parser.add_argument('--collectq', action='store_true', help='output weights 90 percentile for pruning')
parser.add_argument('--prune', action='store_true',  help='use pruning while training')
parser.add_argument('--config', type=str, default='configs/base.yaml',  help='pruning configuration file')
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--graph', type=str,default='Graph',help='Type the name of the graph')



###############################################################################
# Training code
###############################################################################


def evaluate():
    """Calculates loss and prediction accuracy given torch dataloader"""
    # Turn on evaluation mode which disables dropout.
    md.eval()
    avg_loss = RunningAverage()
    avg_acc = RunningAverage()
    T_P=0
    F_P=0
    T_N=0
    F_N=0
    F__P=0
    with torch.no_grad():
        pbar = tqdm(test_dl, ascii=True, leave=False)
        for batch in pbar:
            # run model
            inp, target = batch
            inp, target = inp.to(device), target.to(device)
            out = md(inp.t())

            # calculate loss
            loss = criterion(out.view(-1), target.float())
            avg_loss.update(loss.item())

            # calculate accuracy
            pred = out.view(-1) > 0.5
            correct = pred == target.byte()
            t_p,f_p,t_n,f_n,f__p = confusion(pred,target.byte())
            T_P+= t_p
            F_P+= f_p
            T_N+= t_n
            F_N+= f_n
            F__P+= f__p
            avg_acc.update(torch.sum(correct).item() / len(correct))

            pbar.set_postfix(loss=f'{avg_loss():05.3f}', acc=f'{avg_acc():05.2f}')
    
#     print('False_Positive',F_P)
#     print('True_Positive',T_P)
#     print('False_Neg',F_N)
#     print('True_Neg',T_N)
    # print('Check',F__P)
    if(T_P == 0):
        avg_prec=0.0
    else:
        avg_prec = T_P/(T_P + F_P)
    if(T_P == 0):
        avg_recall=0.0
    else:
        avg_recall = T_P/(T_P + F_N)
    if(avg_prec+avg_recall == 0.0):
        f1_score=0.0
    else:
        f1_score = 2*(avg_prec*avg_recall)/(avg_prec+avg_recall)
    return avg_loss(), avg_acc(), avg_prec, avg_recall, f1_score

def confusion(prediction, truth):
    confusion_vector = prediction / truth

    true_positives = torch.sum(confusion_vector == 1).item()
    false_negatives = torch.sum(confusion_vector == 0).item()
    true_negatives = torch.sum(prediction == 0).item() - false_negatives
    false_positives = prediction.size(dim=0) - true_positives-true_negatives-false_negatives
    f__p = torch.sum(prediction == 1).item() - true_positives

    return true_positives, false_positives, true_negatives, false_negatives, f__p

def train():
    # Turn on training mode which enables dropout.
    md.train()
    avg_loss = RunningAverage()
    avg_acc = RunningAverage()
    avg_prec = RunningAverage()
    avg_recall = RunningAverage()
    sparsity = 0.0
    info = {
        'loss': None,
        'acc': None,
    }

    pbar = tqdm(train_dl, ascii=True, leave=False)
    
    for batch in pbar:
        inp, target = batch
        inp, target = inp.to(device), target.to(device)
        # run model
        md.zero_grad()
        out = md(inp.t())
        loss = criterion(out.view(-1), target.float())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(md.parameters(), args.clip)
        optimizer.step()
        if args.prune:
            pruner.step()

        # upgrade stats
        avg_loss.update(loss.item())
        pred = out.view(-1) > 0.5
        
        correct = pred == target.byte()
        avg_acc.update(torch.sum(correct).item() / len(correct))
        
#         avg_prec.update(t_p/(t_p+f_p))
#         avg_recall.update(t_p/(t_p+f_n))
        info['loss'] = f'{avg_loss():05.3f}'
        info['acc'] = f'{avg_acc():05.2f}'
#         info['prec'] = f'{avg_prec():05.2f}'
#         info['recall'] = f'{avg_recall():05.2f}'
        if args.prune:
            sparsity = pruner.log()
            info['spar'] = f'{sparsity:.2f}'

        pbar.set_postfix(**info)
    
 
    
    return avg_loss(), avg_acc(), sparsity 

if __name__ == '__main__':   
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    warnings.filterwarnings('ignore')
    print(device)
    ###############################################################################
    # Load data
    ###############################################################################


    train_ds = TextDataset(load=args.load_ds, train=True)
    test_ds = TextDataset(load=True, train=False)
    
  
    
    train_sp = TextSampler(train_ds, key=lambda i: len(train_ds.texts[i]), batch_size=args.batch_size)
    train_dl = data.DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sp, collate_fn=PadCollate())
    test_dl = data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, collate_fn=PadCollate())
    
    
    
    with open('data/dataset/itos.pkl', 'rb') as f:
        itos = pickle.load(f)

    print('Loaded train and test data.')


    ###############################################################################
    # Create model
    ###############################################################################


    ntokens = len(itos)
    md = nn.Sequential(OrderedDict([
        ('encoder', model.ClassifierRNN(args.bptt,args.model, ntokens, args.emsize, args.nhid)),
        ('decoder', model.LinearDecoder(args.nhid, 1))
    ])).to(device)
    
    

    if args.prune:
        config = utils.parse_config(args.config)
        pruner = ModelPruner(md, len(train_dl), config)
    md.encoder.rnn.flatten_parameters()

    print(f'Created model with {utils.count_parameters(md)} parameters:')
    print(md)

    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.Adam(
        md.parameters(), 
        lr=args.lr, 
        weight_decay=args.l2alpha,
        betas=(0.8, 0.99),
    )

    ###############################################################################
    # Actual training
    ###############################################################################

    # Loop over epochs.
    lr = args.lr
    best_val_loss = None

    sparsity_arr = []
    trn_acc_arr = []
    trn_loss_arr = []
    val_loss_arr = []
    val_acc_arr = []
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        trn_loss, trn_acc, sparsity = train()
        val_loss, val_acc, prec, recall, f1_score = evaluate()
        trn_acc_arr.append(trn_acc)
        val_acc_arr.append(val_acc)
        sparsity_arr.append(sparsity)
        
        print('-' * 111)
        print(f'| end of epoch {epoch:3d} | time: {time.time()-epoch_start_time:5.2f}s '
            f'| train loss/val loss {trn_loss:05.3f}/{val_loss:05.3f} | train acc/val acc {trn_acc:04.2f}/{val_acc:04.2f} ' 
            f'| Precision {prec:05.3f} | Recall {recall:04.2f} | F1-Score {f1_score:04.2f} '
            f'| sparsity {sparsity:.2f}' )
        print('-' * 111)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(md, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
#     plt.plot(trn_acc_arr,'-o')
#     plt.plot(val_acc_arr,'-o')
#     plt.xlabel('epoch')
#     plt.ylabel('accuracy')
#     plt.legend(['Train','Test'])
#     plt.title('Accuracy')
#     plt.save(args.graph + '-training.png')
    
#     val_loss, val_acc, prec, recall, f1_score = evaluate()
    
#     print(f'| end of epoch {epoch:3d} | time: {time.time()-epoch_start_time:5.2f}s '
#             f'| test loss {val_loss:05.3f} | test acc {trn_acc:04.2f}/{val_acc:04.2f} '
#             f'| Precision {prec:05.3f} | Recall {recall:04.2f} | F1-Score {f1_score:04.2f} '
#             f'| sparsity {sparsity:.2f}' )
    
    if args.prune:
        pruner.save_plot_data(Path('data/pruning'))

    if args.collectq:
        with open('q_value', 'w') as fd:
            for name, param in md.named_parameters():
                weights = param.data.view(-1).cpu().numpy()
                q = np.percentile(weights, 90)
                fd.write(f'{name:20}|  q={q}\n')
