"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def softmax(z):
    exponents = np.exp(z - np.amax(z, axis=1, keepdims=True))
    return np.true_divide(exponents, np.sum(exponents, axis=1, keepdims=True))


def extend_y(y, C):
    """
    creates a suitable extended label matrix of y of binary values
    :param y: vector of shape (N,), whose values are 0 <= c < C
    :param C: number of classes
    :return: extended matrix of shape (N, C)
    """
    y_extended = np.zeros((y.shape[0], C))
    y_extended[np.arange(y.shape[0]), y] = 1
    return y_extended


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    # set initial values
    (N, D), C = X.shape, W.shape[1]
    y_pred = softmax(X @ W)  # (N, C)
    yy = lambda obs, cl: 1 if y[obs] == cl else 0  # as if y was of shape (N, C)

    # preliminary checks
    if y.shape[0] != y_pred.shape[0]:
        raise IndexError('target and training label shapes do not match')
    if X.shape[0] != y.shape[0] or X.shape[1] != W.shape[0]:
        raise IndexError('shapes of W, X, y do not match')

    # compute loss
    for observation in range(N):  # [0, N)
        for class_label in range(C):  # [0, C)
            loss += yy(observation, class_label) * np.log(y_pred[observation, class_label])

    loss = -loss / N + (reg / 2) * np.sum(np.square(W))

    # compute gradient
    for observation in range(N):
        for dim in range(D):
            for class_label in range(C):
                dW[dim, class_label] += (y_pred[observation, class_label] - yy(observation, class_label)) \
                                        * X[observation, dim]

    dW = np.true_divide(dW, N) + np.multiply(W, reg)

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    # set initial values
    N, D = X.shape
    C = W.shape[1]

    y_pred = softmax(X @ W)  # (N, C)
    #y_pred = np.exp(X @ W)
    #y_pred /= np.sum(y_pred, axis=1, keepdims=True)

    y_extended = extend_y(y, C)

    # compute loss
    loss = np.sum(np.log(y_pred[np.arange(N), y]))
    loss = -loss / N + reg * 0.5 * np.sum(W.T @ W)  # *0.5 is faster than /2

    # compute gradient
    #y_pred[np.arange(N), y] -= 1
    dW = X.T @ (y_pred - y_extended)  # if result's shape does not match, it'll raise an error
    dW = dW / N + reg * W

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 5e-7]
    learning_rates = [2.811768697974231e-07, 3.2374575428176464e-07, 3.3932217718953295e-07, 6.55128556859551e-07]
    regularization_strengths = [5e3]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################

    for lr in learning_rates:
        for reg in regularization_strengths:
            classifier = SoftmaxClassifier()
            classifier.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=2000, batch_size=200)

            y_train_pred = classifier.predict(X_train)
            y_val_pred = classifier.predict(X_val)

            train_accuracy = np.mean(y_train == y_train_pred)
            val_accuracy = np.mean(y_val == y_val_pred)

            results[(lr, reg)] = (train_accuracy, val_accuracy)
            all_classifiers.append((classifier, (lr, reg)))

            if val_accuracy > best_val:
                best_softmax = classifier
                best_val = val_accuracy

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
