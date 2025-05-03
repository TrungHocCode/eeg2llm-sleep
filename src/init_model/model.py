import os
import numpy as np
import sklearn.metrics as skmetrics
import tensorflow as tf 
from keras import layers, Model, metrics, regularizers
from keras import backend as K
from class_balancing import calculate_class_weights, weighted_binary_crossentropy, focal_loss, augment_minority_samples, augment_signal, MultiSampleDropout, apply_smote_multilabel, ThresholdOptimizationCallback
import timeit
import logging

logger = logging.getLogger("default_log")
class NeuroSight(Model):
    def __init__(
            self,
            config,
            output_dir = "./output",
            use_rnn = False,
            use_attention = False, 
            use_multi_dropout = False,  
            mixup_alpha = None,  
    ):
        super().__init__()
        self.config = config
        self.output_dir = output_dir
        self.use_rnn = use_rnn
        self.use_attention = use_attention
        self.use_multi_dropout = use_multi_dropout
        self.mixup_alpha = mixup_alpha
        self.class_weights = None

        logger.info(f"Initializing NeuroSight with {self.config['n_classes']} classes, use_rnn={use_rnn}, use_attention={use_attention}")
        
        # tham số cho mô hình cnn
        self.conv_layers = []
        first_filter_size = int(self.config["sampling_rate"] / 2.0)
        first_filter_stride = int(self.config["sampling_rate"] / 16.0)

        # block 1
        self.conv_layers.append(
            layers.Conv1D(
                128, first_filter_size, first_filter_stride,
                padding='same', use_bias=False,
                kernel_regularizer=regularizers.l2(self.config.get("l2_weight_decay", 0.01)),
            )
        )
        self.conv_layers.append(layers.BatchNormalization())
        self.conv_layers.append(layers.ReLU())
        self.conv_layers.append(layers.MaxPooling1D(8, 8))
        self.conv_layers.append(layers.Dropout(0.5))

        # block 2
        for i in range(4):  
            self.conv_layers.append(
                layers.Conv1D(
                    256 if i > 1 else 128, 
                    8, 1, padding='same', use_bias=False,
                    kernel_regularizer=regularizers.l2(self.config.get("l2_weight_decay", 0.01)),
                )
            )
            self.conv_layers.append(layers.BatchNormalization())
            self.conv_layers.append(layers.ReLU())
            
            if i % 2 == 1:
                channels = 256 if i > 1 else 128
                ratio = 16
                # Create squeeze-excitation layers once during initialization
                self.conv_layers.append(self._create_se_block(channels, ratio))

        self.conv_layers.append(layers.MaxPooling1D(4,4))
        self.conv_layers.append(layers.Flatten())

        if self.use_multi_dropout:
            self.conv_layers.append(MultiSampleDropout(0.5, 5))
        else:
            self.conv_layers.append(layers.Dropout(0.5))

        if self.use_attention:
            self.attention_layer = layers.Dense(
                1, 
                activation='tanh',
                kernel_regularizer=regularizers.l2(self.config.get("l2_weight_decay", 0.01))
            )
            self.attention_weights = layers.Dense(
                1, 
                activation='softmax',
                kernel_regularizer=regularizers.l2(self.config.get("l2_weight_decay", 0.01))
            )

        if self.use_rnn:
            n_rnn_layers = self.config.get("n_rnn_layers")
            if not n_rnn_layers:
                raise ValueError("n_rnn_layers must be specified in config when use_rnn=True")
            
            self.rnn_layers = []
            for i in range(n_rnn_layers):
                self.rnn_layers.append(
                    layers.Bidirectional(
                        layers.LSTM(
                            self.config.get("n_rnn_units", 128),
                            dropout=0.3,
                            recurrent_dropout=0.3,
                            return_sequences=(i < n_rnn_layers - 1),
                            kernel_regularizer=regularizers.l2(self.config.get("l2_weight_decay", 0.01)),
                        )
                    )
                )
        self.intermediate_dense = layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.config.get("l2_weight_decay", 0.01)),
        )

        self.output_layer = layers.Dense(
            self.config["n_classes"],
            kernel_regularizer=regularizers.l2(self.config.get("l2_weight_decay", 0.01)),
        )
        
        # Thresholds for each class (will be optimized during training)
        self.thresholds = tf.Variable(
            tf.ones(self.config["n_classes"]) * 0.5,
            trainable=False,
            dtype=tf.float32,
            name="classification_thresholds"
        )

    def _create_se_block(self, channels, ratio=16):
        """Create a squeeze-excitation block as a Keras layer"""
        se_global_pool = layers.GlobalAveragePooling1D()
        se_reshape = layers.Reshape((1, channels))
        se_reduce = layers.Dense(channels // ratio, activation='relu', use_bias=False)
        se_expand = layers.Dense(channels, activation='sigmoid', use_bias=False)
        
        def se_block_func(inputs):
            se = se_global_pool(inputs)
            se = se_reshape(se)
            se = se_reduce(se)
            se = se_expand(se)
            return layers.multiply([inputs, se])
            
        # Return a Lambda layer that applies the SE block function
        return layers.Lambda(lambda x: se_block_func(x))

    def _make_se_block(self, channels, ratio=16):
        """Create a squeeze-excitation block for feature recalibration"""
        def se_block(inputs):
            # Squeeze - use fixed channels instead of dynamic tf.shape
            se = layers.GlobalAveragePooling1D()(inputs)
            se = layers.Reshape((1, channels))(se)
            
            # Excitation
            se = layers.Dense(channels // ratio, activation='relu', use_bias=False)(se)
            se = layers.Dense(channels, activation='sigmoid', use_bias=False)(se)
            
            # Scale
            return layers.multiply([inputs, se])
        
        return se_block

    def call(self, inputs, training = False):
        x = inputs

        for layer in self.conv_layers:
            if isinstance(layer, layers.BatchNormalization) or isinstance(layer, layers.Dropout) or isinstance(layer, MultiSampleDropout):
                x = layer(x, training = training)
            else:
                x = layer(x)

        if self.use_rnn:
            batch_size = tf.shape(x)[0]
            seq_inputs = tf.reshape(x, [batch_size, self.config.get("seq_length", 1), -1])
            
            # Pass through bidirectional LSTM layers
            for rnn_layer in self.rnn_layers:
                seq_inputs = rnn_layer(seq_inputs, training=training)
                
            if self.use_attention:
                # Apply attention
                e = self.attention_layer(seq_inputs)
                a = self.attention_weights(e)
                context = tf.reduce_sum(a * seq_inputs, axis=1)
                x = context
            else:
                # If no attention, just use the last LSTM output
                x = seq_inputs

        # Add intermediate dense layer
        x = self.intermediate_dense(x)
        if training:
            x = tf.nn.dropout(x, rate=0.3)
            
        logits = self.output_layer(x)
        return logits
    
    def mixup_data(self, x, y, alpha=0.2):
        """Apply mixup augmentation to the batch"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = tf.shape(x)[0]
        indices = tf.random.shuffle(tf.range(batch_size))
        
        mixed_x = lam * x + (1 - lam) * tf.gather(x, indices)
        # mixed_y = lam * y + (1 - lam) * tf.gather(y, indices)
        
        return mixed_x, y

    def train_step(self, data):
        x, y, sample_weights = data

        if self.mixup_alpha is not None and self.mixup_alpha > 0:
            x, y = self.mixup_data(x, y, self.mixup_alpha)

        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            # Apply class weights or focal loss if needed
            if self.class_weights is not None:
                # Custom weighted loss
                loss_func = lambda y_true, y_pred: weighted_binary_crossentropy(
                    y_true, y_pred, self.class_weights
                )
                loss = loss_func(y, logits)
            else:
                # Standard loss
                loss = self.compiled_loss(y, logits, sample_weight=sample_weights)

        gradients = tape.gradient(loss, self.trainable_variables)
        # Gradient clipping to prevent exploding gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        probs = tf.sigmoid(logits)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, probs, sample_weight=sample_weights)

        return {m.name: m.result() for m in self.metrics}
    

    def test_step(self, data):
        x, y, sample_weights = data
        logits = self(x, training=False)
        
        if self.class_weights is not None:
            # Custom weighted loss for validation too
            loss_func = lambda y_true, y_pred: weighted_binary_crossentropy(
                y_true, y_pred, self.class_weights
            )
            loss = loss_func(y, logits)
        else:
            loss = self.compiled_loss(y, logits, sample_weight=sample_weights)

        probs = tf.sigmoid(logits)
        
        # Apply per-class thresholds for better metrics
        pred_with_thresh = tf.cast(
            tf.greater_equal(probs, self.thresholds), 
            tf.float32
        )
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                # Use threshold-adjusted predictions for metrics
                metric.update_state(y, pred_with_thresh, sample_weight=sample_weights)

        return {m.name: m.result() for m in self.metrics}

    def set_class_weights(self, class_weights):
        """Set class weights for the model"""
        self.class_weights = class_weights
        
    def set_thresholds(self, thresholds):
        """Set classification thresholds for each class"""
        self.thresholds.assign(thresholds)
    
if __name__ == "__main__":
    from config import pretrain, finetune
    
    # Pretraining (CNN)
    model = NeuroSight(config=pretrain, output_dir="./output/cnn")
    model.build(input_shape=(None, pretrain["input_size"], 1))
    model.summary()
    
    # Finetuning (CNN + RNN)
    model_rnn = NeuroSight(config=finetune, use_rnn=True, output_dir="./output/rnn")
    model_rnn.build(input_shape=(None, finetune["input_size"], 1))
    model_rnn.summary()


        


