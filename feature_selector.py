#pip #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from keras.layers import Multiply, Dropout, Dense, Input
from tensorflow import keras
from keras import Model
from datetime import datetime
from sklearn.svm import SVC, SVR

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

np.set_printoptions(precision=1, suppress=True)

class FeatureSelector:
    
    def __init__(self,
                 selector_nodes : np.ndarray = None,
                 task_nodes : np.ndarray = None, 
                 dout : np.array = None, 
                 task_activation = 'relu'):
        """
        Initializes the selector class with two neural network models.
        
        :param selector_nodes: Numpy array specifying the number of nodes in each layer of the selector model.
        :param task_nodes: Numpy array specifying the number of nodes in each layer of the task model.
        :param dout: Not used
        :param task_activation: Activation function for the task model.
        """
        
        if selector_nodes and task_nodes:
            
            # Sanity check node arrays
            if selector_nodes[0] != selector_nodes[-1] or  selector_nodes[-1] != task_nodes[0]:
                raise ValueError('Node arrays are non-valid.')
                
        self.num_features = task_nodes[0]  
        self.selection_model = self.create_model(nodes=selector_nodes,
                                                 dropout_rates=[0.5] * len(selector_nodes),
                                                 activation='sigmoid')
        
        self.model = self.create_model(nodes=task_nodes,
                                       activation=task_activation)
        
        # Define the Multiply layer outside the functions
        self.multiply_layer = Multiply()
    
    def create_model(self, nodes, dropout_rates=None, activation=None):
        """
        Creates a neural network model with specified layers using the Keras functional API.
        
        :param nodes: Numpy array specifying the number of nodes in each layer.
        :param dropout_rates: Numpy array specifying dropout rates for each layer (optional).
        :param activation: Activation function for the output layer (optional).
        :return: Keras Model.
        """
        
        # Input layer
        inputs = Input(shape=(nodes[0],))
        x = inputs
        
        # Hidden layers
        for i, node in enumerate(nodes[1:-1]):
            x = Dense(node, activation='relu')(x)
            if dropout_rates and i < len(dropout_rates):
                x = Dropout(dropout_rates[i])(x)
        
        # Output layer
        outputs = Dense(nodes[-1], activation=activation)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def compile(self, 
                factor_loss,
                regression = True,
                loss = 'mse',
                optimizer = keras.optimizers.legacy.Adam(),
                metric = 'mae',
                penalization_fn = tf.reduce_mean,
                threshold = 0.01,
                early_stopping = None,
                verbose = 2):
        """
        Compiles the selector model with a loss function and optimizer.
        
        :param factor_loss: Factor that controls the ratio of the selection model's loss.
        :param regression: Boolean indicating whether it's a regression problem (default is True).
        :param loss_fn: Loss function to be used for training. Default is MeanSquaredError.
        :param optimizer: Optimizer to be used for training. Default is Adam optimizer.
        :param metric: Metric to measure during training (default is MeanAbsoluteError).
        :param penalization_fn: Penalization function for selection loss.
        :param threshold: Threshold value. Mask entries larger than this value are counted as selected features.
        :param early_stopping: Tuple (patience, es_threshold) for early stopping (optional).
        :param verbose: Verbosity level for training logs (default is 2).
        """
        self.regression = regression
        
        
        if verbose == 2:
            self.model.summary()
            self.selection_model.summary()
        
        if early_stopping:
            self.early_stopping = True
            self.patience = early_stopping[0]
            self.tracking_window_size = early_stopping[1]
        else:
            self.early_stopping = False
        
        if metric == 'categorical_accuracy':
            self.train_metric = keras.metrics.CategoricalAccuracy()
            self.metric = keras.metrics.CategoricalAccuracy()
        elif metric == 'mae':
            self.train_metric = keras.metrics.MeanAbsoluteError()
            self.metric = keras.metrics.MeanAbsoluteError()
        else:
            raise ValueError('Unknown Metric. Please provide mae or categorical accuracy.')
        
        if loss == 'mse':
            self.loss_fn = keras.losses.MeanSquaredError()
        elif loss == 'categorical_crossentropy':
            self.loss_fn = keras.losses.CategoricalCrossentropy()
        else:
            raise ValueError('Unknown Loss. Please provide mse or categorical crossentropy.')
        
        self.penalization_fn = penalization_fn
        self.optimizer = keras.optimizers.legacy.Adam()
        self.selection_optimizer = keras.optimizers.legacy.Adam()
        self.factor_loss = factor_loss
        self.threshold = threshold
    
    def fit(self, 
            epochs, 
            X_train, y_train, 
            batch_size=32, 
            validation_data=None,
            verbose = 2):
        """
        Trains the selector and task models on the provided dataset.
        
        :param epochs: Number of epochs to train the models.
        :param X_train: Training data inputs.
        :param y_train: Training data labels.
        :param batch_size: Batch size for training. Default is 32.
        :param validation_data: Tuple (X_val, y_val) for validation data. Default is None.
        :param verbose: Verbosity mode for training logs. Default is 2.
        :return: Training history.
        """

        history = {}
        self.verbose = verbose
        self.epochs = epochs
        
        # Prepare the training dataset.
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset)).batch(batch_size)
        
        X_val = validation_data[0]
        y_val = validation_data[1]
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size)
        
        num_train_batches = len(train_dataset)
        
        # Store the selection metrics. 
        self.train_metric_container = np.zeros(epochs)
        self.val_metric_container = np.zeros(epochs)
        self.penalization = np.zeros(epochs)
        self.loss_selection = np.zeros(epochs)
        self.loss_task = np.zeros(epochs)
        self.num = np.zeros(epochs)
        self.masks = np.zeros((self.num_features, epochs))
        self.epoch_time = np.zeros(epochs)

        for epoch in range(epochs):
            now = datetime.now()            

            for x, y in train_dataset:
                loss_selection_model, loss_model, penalization, mask = self.selection_train_step(x, y)
                
                self.masks[:,epoch] += mask.numpy()
                self.loss_selection[epoch] += loss_selection_model
                self.loss_task[epoch] += loss_model
                self.train_metric_container[epoch] += self.train_metric.result().numpy()
                self.train_metric.reset_states()

            
            # Normalize metrics
            self.masks[:,epoch] /= num_train_batches
            self.loss_selection[epoch] /= num_train_batches
            self.loss_task[epoch] /= num_train_batches
            self.penalization[epoch] /= num_train_batches
            self.train_metric_container[epoch] /= num_train_batches
            
            self.num[epoch] = len(np.where(self.masks[:,epoch]>self.threshold)[0])
            
            for x, y in val_dataset:
                val_loss = self.val_step(x, y)
                self.val_metric_container[epoch] += self.metric.result().numpy()
                self.metric.reset_states()

            self.val_metric_container[epoch] /= len(val_dataset)
            
            # Implement early stopping to prevent overfitting and too aggressive feature reduction.
            if self.early_stopping and epoch > self.patience:
                                
                if epoch > self.patience + self.tracking_window_size:
                       if self.regression:
                           index = epoch - self.tracking_window_size
                           # Check if all the next 10 values are smaller than arr[j]
                           if all(self.val_metric_container[index] < self.val_metric_container[index+1:epoch + 1]):
                               break
                       else:
                           index = epoch - self.tracking_window_size
                           # Check if all the next 10 values are smaller than arr[j]
                           if all(self.val_metric_container[index] > self.val_metric_container[index+1:epoch + 1]):
                               break
                            
            time = datetime.now() - now
            self.epoch_time[epoch] = time.total_seconds()
            if self.verbose >= 1:
                template = '\nEpoch: {},\nNum selected: {},\nTrain metric: {:.2f},\nVal metric: {:.2f},\nSelection loss: {:.4f},\nTask loss: {:.2f},\nTask loss*lambda: {:.2f}\npenalization: {:.2f},\nTIME = {}\n'
                template = template.format(epoch + 1, 
                                           self.num[epoch],
                                           self.train_metric_container[epoch], 
                                           self.val_metric_container[epoch],
                                           self.loss_selection[epoch], 
                                           self.loss_task[epoch], 
                                           self.loss_task[epoch]*self.factor_loss, 
                                           self.penalization[epoch],
                                           time) 
                print(template)
                if verbose == 1:
                    print(f'{np.floor(np.log10(np.abs(self.masks[:,epoch]))).astype(int)}')
                    print(self.masks[:,epoch])
                
        self.last_epoch = epoch
 
        history['masks'] = self.masks
        history['num'] = self.num
        history['loss_task'] = self.loss_task
        history['loss_selection'] = self.loss_selection
        history['penalization'] = self.penalization
        history['train_metric'] = self.train_metric_container
        history['val_metric'] = self.val_metric_container
        history['time'] = self.epoch_time
        
        return history 
        
    def eval(self, window_size=None,
             threshold=None,
             p=0.75):
        """
        Evaluates the selected features based on a threshold and a window size.
        
        :param window_size: Window size for evaluating features.
        :param threshold: Threshold value for feature selection (optional).
        :param p: Percentage of epochs to consider for quartile calculation (optional).
        :return: Selected features.
        """
        
        if threshold:
            self.threshold = threshold
        
        if not window_size:
            window_size = int(self.last_epoch*0.5)
        
        quartile = int(p*(self.last_epoch-window_size))
        selected_features  = np.zeros((self.num_features,))
        
        for col in range(window_size, self.last_epoch):
            mask = self.masks[:, col]
            selected_features[np.where(mask > self.threshold)] += 1
        self.selected_features = np.where(selected_features > quartile)
        
        return self.selected_features
        
    def get_mask(self,):
        
        """
        Get the mask for the selected features.
        
        :return: Mask for selected features.
        """
        m = np.zeros((self.num_features,))
        m[self.selected_features] = 1
        return m
   
    @tf.function
    def selection_train_step(self, x, y):
        """
        Performs a training step that includes both the selection and task models.
        
        :param x: Input data.
        :param y: True labels for the data.
        :return: Loss values and metrics for the step.
        """
        
        perm = np.random.permutation(len(x))
        perm_tensor = tf.constant(perm)

        # Shuffle x.
        x_shuffle = tf.gather(x, perm_tensor)
        
        with tf.GradientTape() as tape:
            mask = self.selection_model(x_shuffle, training=True)
            mult_out = self.multiply_layer([mask, tf.cast(x, tf.float32)])
            
            self.train_step(mult_out, y)            
            pred = self.model(mult_out, training=False)
            
            loss_task = self.loss_fn(y, pred)
            penalization = tf.reduce_mean(mask)
            
            loss_selection = self.factor_loss*loss_task + penalization
        
            # Backward pass: compute gradients of the loss w.r.t the model's parameters
            gradients = tape.gradient(loss_selection, self.selection_model.trainable_variables)
        
            # Apply gradients: perform one optimizer update
            self.selection_optimizer.apply_gradients(zip(gradients, self.selection_model.trainable_variables))
        self.train_metric.update_state(y, pred)
        
        return loss_selection, loss_task, penalization, tf.math.reduce_mean(mask, axis=0)
    
    @tf.function
    def train_step(self, x, y):
        """
        Performs a single training step for the task model.
    
        :param x: Input data.
        :param y: True labels for the data.
        :return: Loss value for the step.
        """
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            pred = self.model(x, training=True)  # Logits for this minibatch
    
            # Compute the loss value for this minibatch.
            loss = self.loss_fn(y, pred)
    
            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss, self.model.trainable_weights)
    
            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    
            return loss
    
    @tf.function
    def val_step(self, x, y):
        """
        Performs a single validation step for the task model.
    
        :param x: Input data.
        :param y: True labels for the data.
        :return: Loss value for the validation step.
        """
        # Select features
        mask = self.selection_model(x, training=False)
        mult_out = self.multiply_layer([mask, tf.cast(x, tf.float32)])
    
        pred = self.model(mult_out, training=False)
        loss = self.loss_fn(y, pred)
    
        # Update validation metrics
        self.metric.update_state(y, pred)
    
        return loss
    
    def build_SVM(self,kernel='rbf', C=1.0, epsilon=0.01, is_regression=False, **kwargs):
        if is_regression:
           self.SVM = SVR(kernel=kernel, C=C, epsilon=epsilon, **kwargs)
        else:
            self.SVM = SVC(kernel=kernel, C=C, **kwargs)
        # self.SVM.fit(X_train, y_train)
    
        
