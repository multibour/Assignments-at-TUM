from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        
        for epoch in range(num_epochs):

            # train
            for it, (inputs, target) in enumerate(train_loader, 1):
                inputs, target = Variable(inputs), Variable(target)#.view(-1).long()
                #if model.is_cuda:
                #    inputs, target = inputs.cuda(), target.cuda()

                optim.zero_grad()
                out = model(inputs)

                loss = self.loss_func(out, target)
                loss.backward()
                optim.step()

                self.train_loss_history.append(loss.data.cpu().numpy())

                if log_nth != 0 and it % log_nth == 0:
                    train_loss = np.mean(self.train_loss_history[-log_nth:])
                    print('[Iteration {}/{}] TRAIN loss: {}'.format(epoch*iter_per_epoch + it,
                                                                    num_epochs * iter_per_epoch,
                                                                    train_loss)
                          )

            pred = F.softmax(out, -1)
            _, pred = torch.max(out, 1)

            train_acc = np.mean((pred == target)[target >= 0].data.cpu().numpy())
            self.train_acc_history.append(train_acc)
            if log_nth != 0:
                print('[Epoch {}/{}] TRAIN acc/loss: {}/{}'.format(epoch+1,
                                                                   num_epochs,
                                                                   train_acc,
                                                                   train_loss)
                      )

            # validate
            val_losses, val_scores = [], []
            model.eval()
            for inputs, target in val_loader:
                inputs, target = Variable(inputs), Variable(target)#.view(-1).long()
                #if model.is_cuda:
                #    inputs, target = inputs.cuda(), target.cuda()

                out = model.forward(inputs)
                loss = self.loss_func(out, target)
                val_losses.append(loss.data.cpu().numpy())

                _, pred = torch.max(out, 1)

                scores = np.mean((pred == target)[target >= 0].data.cpu().numpy())
                val_scores.append(scores)

            model.train()
            self.val_acc_history.append(np.mean(val_scores))
            self.val_loss_history.append(np.mean(val_losses))

            if log_nth != 0:
                print('[Epoch {}/{} VAL acc/loss: {}/{}'.format(epoch+1,
                                                                num_epochs,
                                                                self.val_acc_history[-1],
                                                                self.val_loss_history[-1])
                      )

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
