import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np

class GeoTAETr: #Geometric Transformation Autoencoder with Translation
    def __init__(self, verbose=0, lmb = 0.001):
        # Initialize the encoder, decoder, and autoencoder models as None
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.verbose = verbose
        self.lmb = lmb

    def build_encoder(self, input_shape, embedding_dim):
        if self.verbose > 0:
            print("Building encoder...")
        
        # Build the encoder model
        inputs = layers.Input(shape=input_shape, name='input_data')
        x = layers.Flatten()(inputs)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)

        # Output layer with L2 regularization
        z = layers.Dense(embedding_dim, activation='relu', activity_regularizer=regularizers.l2(1e-5))(x)
        self.encoder = models.Model(inputs, z, name='encoder')
        if self.verbose > 1:
            self.encoder.summary()

    def build_decoder(self, embedding_dim, original_shape):
        if self.verbose > 0:
            print("Building decoder...")
        # Build the decoder model
        inputs = layers.Input(shape=(embedding_dim,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(128, activation='relu')(x)
        
        # Output layer reshaped to the original input shape
        x = layers.Dense(tf.reduce_prod(original_shape).numpy(), activation='sigmoid')(x)
        outputs = layers.Reshape(original_shape)(x)

        self.decoder = models.Model(inputs, outputs, name='decoder')
        if self.verbose > 1:
            self.decoder.summary()

    def emb_and_reconstr_loss(self, inputs, decoded, encoded, embedding):
        """Custom Loss function to regularize it to the embedding space and the reconstruction space"""
        mse = tf.keras.losses.MeanSquaredError()
        reconstruction_loss = mse(inputs, decoded)
        embedding_loss = mse(embedding, encoded)
        return reconstruction_loss + embedding_loss * self.lmb

    def custom_loss(self):
        def loss(y_true, y_pred):
            data_dim = self.data.shape[1]  # e.g., 4
            # Split the combined target tensor into input_data and embedding parts.
            batch_inputs = y_true[:, :data_dim]
            batch_embedding = y_true[:, data_dim:]
            encoded = self.encoder(batch_inputs)
            return self.emb_and_reconstr_loss(batch_inputs, y_pred, encoded, batch_embedding)
        return loss

    def fit(self, data, embedding, epochs=50, batch_size=256):
        if self.verbose > 0:
            print("Fitting the autoencoder model...")

        # Save the data and the embedding
        self.data = data
        self.embedding = embedding
        
        # Fit the autoencoder model to the data
        input_shape = self.data.shape[1:]
        embedding_dim = embedding.shape[1]

        # Build encoder and decoder models
        self.build_encoder(input_shape, embedding_dim)
        self.build_decoder(embedding_dim, input_shape)

        # Create the autoencoder model by connecting encoder and decoder
        encoded = self.encoder.output
        decoded = self.decoder(encoded)

        self.autoencoder = models.Model(self.encoder.input, decoded, name='autoencoder')
        
        # Compile the autoencoder with the custom loss function
        self.autoencoder.compile(optimizer='adam', loss=self.custom_loss())

        # Instead of providing a tuple as target, concatenate data and embedding.
        target = tf.concat([self.data, embedding], axis=1)
        dataset = tf.data.Dataset.from_tensor_slices((self.data, target))
        dataset = dataset.shuffle(buffer_size=self.data.shape[0]).batch(batch_size)
        
        # Inspect one sample (one row) from the dataset:
        for sample in dataset.take(1):
            inputs, targets = sample
            # targets is a tuple: (input_data, embedding)
            print("Sample input:", inputs.numpy()[0])
            print("Sample embedding:", targets[1].numpy()[0])

        self.autoencoder.fit(dataset, epochs=epochs)
        if self.verbose > 0:
            print("Training complete.")

    def encode(self, data):
        if self.verbose > 0:
            print("Encoding data...")
        # Encode the data using the encoder model
        return self.encoder.predict(data)

    def decode(self, encoded_data):
        if self.verbose > 0:
            print("Decoding data...")
        # Decode the encoded data using the decoder model
        return self.decoder.predict(encoded_data)