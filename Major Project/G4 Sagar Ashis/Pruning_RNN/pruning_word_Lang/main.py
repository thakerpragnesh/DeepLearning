import torch
import torch.nn as nn
import torch.onnx
import argparse
import time
import math
import os

from collections import OrderedDict

import pandas as pd

import model
import data
import utils
import config
from tabulate import tabulate


parser = argparse.ArgumentParser(description='PyTorch PTB RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/ptb',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='checkpoint.pth.tar',
                    help='path to save the final model')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

SUMMARY_CHOICES = ['sparsity', 'model', 'modules', 'png', 'percentile']
parser.add_argument('--summary', type=str, choices=SUMMARY_CHOICES,
                    help='print a summary of the model, and exit - options: ' +
                    ' | '.join(SUMMARY_CHOICES))
parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                    help='configuration file for pruning the model (default is to use hard-coded schedule)')
parser.add_argument('--momentum', default=0., type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0., type=float,
                    metavar='W', help='weight decay (default: 0)')

args = parser.parse_args()

def weights_sparsity_summary(model, return_total_sparsity=False, param_dims=[2, 4]):
    df = pd.DataFrame(columns=['Name', 'Shape', 'NNZ (dense)', 'NNZ (sparse)',
                               'Cols (%)', 'Rows (%)', 'Ch (%)', '2D (%)', '3D (%)',
                               'Fine (%)', 'Std', 'Mean', 'Abs-Mean'])
    pd.set_option('display.precision', 2)
    params_size = 0
    sparse_params_size = 0
    for name, param in model.state_dict().items():
        # Extract just the actual parameter's name, which in this context we treat as its "type"
        if param.dim() in param_dims and any(type in name for type in ['weight', 'bias']):
            _density = utils.density(param)
            params_size += torch.numel(param)
            sparse_params_size += param.numel() * _density
            df.loc[len(df.index)] = ([
                name,
                utils.size_to_str(param.size()),
                torch.numel(param),
                int(_density * param.numel()),
                utils.sparsity_cols(param)*100,
                utils.sparsity_rows(param)*100,
                utils.sparsity_ch(param)*100,
                utils.sparsity_2D(param)*100,
                utils.sparsity_3D(param)*100,
                (1-_density)*100,
                param.std().item(),
                param.mean().item(),
                param.abs().mean().item()
            ])

    total_sparsity = (1 - sparse_params_size/params_size)*100

    df.loc[len(df.index)] = ([
        'Total sparsity:',
        '-',
        params_size,
        int(sparse_params_size),
        0, 0, 0, 0, 0,
        total_sparsity,
        0, 0, 0])

    if return_total_sparsity:
        return df, total_sparsity
    return df

def weights_sparsity_tbl_summary(model, return_total_sparsity=False, param_dims=[2, 4]):
    df, total_sparsity = weights_sparsity_summary(model, return_total_sparsity=True, param_dims=param_dims)
    t = tabulate(df, headers='keys', tablefmt='psql', floatfmt=".5f")
    if return_total_sparsity:
        return t, total_sparsity
    return t

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_config(path):
    if isinstance(path, str):
        path = Path(path)
    assert path.exists()

    yaml = YAML()
    return yaml.load(path)

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")


###############################################################################
# Load data
###############################################################################

print("Preparing data")
corpus = data.Corpus(args.data)
print("Done")

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)

if args.resume:
    with open(args.resume, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        model.rnn.flatten_parameters()
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)


criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)



def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0), args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


def train(epoch, optimizer, compression_scheduler=None):
    # Turn on training mode which enables dropout.
    model.train()

    total_samples = train_data.size(0)
    steps_per_epoch = math.ceil(total_samples / args.bptt)

    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0), args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)

        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, minibatch_id=batch, minibatches_per_epoch=steps_per_epoch)
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)

        if compression_scheduler:
            # Before running the backward phase, we allow the scheduler to modify the loss
            # (e.g. add regularization loss)
            loss = compression_scheduler.before_backward_pass(epoch, minibatch_id=batch,
                                                              minibatches_per_epoch=steps_per_epoch, loss=loss,
                                                              return_loss_components=False)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()

        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch, minibatch_id=batch, minibatches_per_epoch=steps_per_epoch)

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            print(
                    '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} '
                    '| loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
            stats = ('Performance/Training/',
                OrderedDict([
                    ('Loss', cur_loss),
                    ('Perplexity', math.exp(cur_loss)),
                    ('LR', lr),
                    ('Batch Time', elapsed * 1000)])
                )
            steps_completed = batch + 1



def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
                   format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


compression_scheduler = None

if args.compress:
    # Create a CompressionScheduler and configure it from a YAML schedule file
    source = args.compress
    compression_scheduler = config.file_config(model, None, args.compress)

optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          patience=0, verbose=True, factor=0.5)

best_val_loss = float("inf")
try:
    for epoch in range(0, args.epochs):
        epoch_start_time = time.time()
        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch)

        train(epoch, optimizer, compression_scheduler)

        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.3f} | '
                       'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)

        t, total = weights_sparsity_tbl_summary(model, return_total_sparsity=True)
        print("\nParameters:\n" + str(t))
        print('Total sparsity: {:0.2f}\n'.format(total))

        stats = ('Performance/Validation/',
            OrderedDict([
                ('Loss', val_loss),
                ('Perplexity', math.exp(val_loss))]))

        with open(args.save, 'wb') as f:
            torch.save(model, f)

        if val_loss < best_val_loss:
            with open(args.save+".best", 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        lr_scheduler.step(val_loss)

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the last saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

