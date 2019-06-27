from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 2e-4,
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
            print('(Epoch %d / %d)' % (epoch+1,num_epochs))
            for i, (images, labels) in enumerate(train_loader):
                

                #ForwardPass
                prediction = model(images)
                #_ , prediction = torch.max(prediction,1)
                #pred = prediction.view(10,-1)
                #lab = labels.view(10,-1)
                #print(pred.size())
                #print(lab.size())
                pred = prediction.float()
                loss = self.loss_func(pred,labels)
                self.train_loss_history.append(loss.item())
                
                if log_nth > 0 and i % log_nth == 0:
                    print('  (Iteration %d / %d) loss: %f' % (i+1,iter_per_epoch,loss.item()))
                
                #backwrdpass
                optim.zero_grad()
                loss.backward()
                optim.step()
                
            #calculate accuracy
            prob, idx = torch.max(prediction,1)
            new_labels = labels >= 0
            acc = torch.mean((idx == labels)[new_labels].float()).item()
            self.train_acc_history.append(acc)
            print('  (Epoch %d / %d) TRAIN acc/loss: %f/%f' % (epoch+1,num_epochs,acc,loss))
            
            
            #validation
            val_loss_list =[]
            val_acc_list = []
            for i, (val_images, val_labels) in enumerate(val_loader): 
                val_pred = model(val_images)
                val_loss = self.loss_func(val_pred,val_labels)
                val_loss_list.append(val_loss.item())
                
                
                val_prob, val_idx = torch.max(val_pred,1)
                new_labels = val_labels >= 0
                val_acc = torch.mean((val_idx == val_labels)[new_labels].float()).item()
                val_acc_list.append(val_acc)
                
            print('  (Epoch %d / %d) VAL acc/loss: %f/%f' % (epoch+1,num_epochs,np.mean(val_acc_list),np.mean(val_loss_list)))
            self.val_acc_history.append(np.mean(val_acc_list))
            self.val_loss_history.append(np.mean(val_loss_list))
        
        
        model.train()
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
