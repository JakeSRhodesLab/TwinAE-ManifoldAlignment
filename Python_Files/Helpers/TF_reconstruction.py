import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np

class GeoTAETr: #Geometric Transformation Autoencoder with Translation
    def __init__(self, verbose=0):
        # Initialize the encoder, decoder, and autoencoder models as None
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.verbose = verbose

    def build_encoder(self, input_shape, embedding_dim):
        if self.verbose > 0:
            print("Building encoder...")
        # Build the encoder model
        inputs = layers.Input(shape=input_shape)
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
        return reconstruction_loss + embedding_loss

    def custom_loss(self, embedding):
        def loss(y_true, y_pred):
            batch_size = y_true.shape[0]
            batch_embedding = embedding[:batch_size]
            encoded = self.encoder(y_true)
            decoded = self.decoder(encoded)
            return self.emb_and_reconstr_loss(y_true, decoded, encoded, batch_embedding)
        return loss

    def fit(self, data, embedding, epochs=50, batch_size=256):
        if self.verbose > 0:
            print("Fitting the autoencoder model...")
        # Fit the autoencoder model to the data
        input_shape = data.shape[1:]
        embedding_dim = embedding.shape[1]

        # Build encoder and decoder models
        self.build_encoder(input_shape, embedding_dim)
        self.build_decoder(embedding_dim, input_shape)

        # Create the autoencoder model by connecting encoder and decoder
        inputs = layers.Input(shape=input_shape)
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)

        self.autoencoder = models.Model(inputs, decoded, name='autoencoder')
        
        # Compile the autoencoder with the custom loss function
        self.autoencoder.compile(optimizer='adam', loss=self.custom_loss(embedding))

        # Create a dataset to ensure the data and embedding are batched together
        dataset = tf.data.Dataset.from_tensor_slices((data, embedding))
        dataset = dataset.batch(batch_size).shuffle(buffer_size=len(data))

        # Train the autoencoder
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