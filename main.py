from datetime import time

import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader

from Data import MyData
from model import Encoder, Decoder
from utils import read_langs, max_length, pad_sequences, sort_batch, loss_function


def main():
    input_lang, output_lang, pairs, data1, data2 = read_langs("eng", "fra", True)
    input_tensor = [[input_lang.word2index[s] for s in es.split(' ')] for es in data1]
    target_tensor = [[output_lang.word2index[s] for s in es.split(' ')] for es in data2]
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)

    input_tensor = [pad_sequences(x, max_length_inp) for x in input_tensor]
    target_tensor = [pad_sequences(x, max_length_tar) for x in target_tensor]
    print(len(target_tensor))

    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                    target_tensor,
                                                                                                    test_size=0.2)

    # Show length
    print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 64
    N_BATCH = BUFFER_SIZE // BATCH_SIZE
    embedding_dim = 256
    units = 1024
    vocab_inp_size = len(input_lang.word2index)
    vocab_tar_size = len(output_lang.word2index)

    train_dataset = MyData(input_tensor_train, target_tensor_train)
    val_dataset = MyData(input_tensor_val, target_tensor_val)

    dataset = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                         drop_last=True,
                         shuffle=True)

    device = torch.device("cpu")

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, units, BATCH_SIZE)

    encoder.to(device)
    decoder.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=0.001)

    EPOCHS = 10

    for epoch in range(EPOCHS):
        start = time()

        encoder.train()
        decoder.train()

        total_loss = 0

        for (batch, (inp, targ, inp_len)) in enumerate(dataset):
            loss = 0

            xs, ys, lens = sort_batch(inp, targ, inp_len)
            enc_output, enc_hidden = encoder(xs.to(device), lens, device)
            dec_hidden = enc_hidden
            dec_input = torch.tensor([[output_lang.word2index['<sos>']]] * BATCH_SIZE)

            for t in range(1, ys.size(1)):
                predictions, dec_hidden, _ = decoder(dec_input.to(device),
                                                     dec_hidden.to(device),
                                                     enc_output.to(device))
                loss += loss_function(criterion, ys[:, t].to(device), predictions.to(device))
                # loss += loss_
                dec_input = ys[:, t].unsqueeze(1)

            batch_loss = (loss / int(ys.size(1)))
            total_loss += batch_loss

            optimizer.zero_grad()

            loss.backward()

            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.detach().item()))

        ### TODO: Save checkpoint for model
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / N_BATCH))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
