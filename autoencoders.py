import keras 
import keras.backend as K
import numpy as np 
import random
import math
from sklearn.metrics import mean_squared_error

class Autoencoder: 
    def __init__(self, 
                 input_shape,
                 encoder_args,
                 latent_dim,
                 n_blocks,
                 encoder_latent_layer_type = "simple",
                 verbose=True):
        
        # input_shape: (n_timesteps, n_features)
        # encoder args: a dictionary with arguments: 
        #       (values are lists with 1 value per block or 1 int/string)
        #       ex. if n_blocks = 3: [1,2,3];  1 -> [1,1,1]; "same" -> ["same", "same", "same"]
        #       # filters: n of filters per CONV layer 
        #       # kernel_size
        #       # padding 
        #       # activation
        #       # pooling: (set to 1 to not do pooling)
        # n_blocks: number of block of the encoder (and decoder)
        # latent_dim: size of the latent dimension
        # encoder_latent_layer_type: simple, dense or variational
        #       for simple to work you must enter the correct latent_dim size else the decoder won't work
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder_latent_layer_type = encoder_latent_layer_type
        self.n_blocks = n_blocks
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.inputs = None
        self.outputs = None
        self.encoder_args = self.convert_args(**encoder_args)
        self.decoder_args = self.get_decoder_args()
        self.padding = self.check_padding()
        self.verbose = verbose
        
        
    def my_vae_loss(self, y_true, y_pred):
        xent_loss = self.input_shape[0] * keras.losses.mean_squared_error(K.flatten(self.inputs), K.flatten(self.outputs))
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(xent_loss + kl_loss)
        #vae_loss = kl_loss
        return vae_loss
    
    def sampling(self, args):
        self.z_mean, self.z_log_var = args
        batch = K.shape(self.z_mean)[0]
        dim = K.int_shape(self.z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return self.z_mean + K.exp(0.5 * self.z_log_var) * epsilon
    
    """
    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    """
    """
    def sampling(self, args):
        self.z_mean, self.z_log_var = args
        epsilon = K.random_normal(shape=K.shape(self.z_mean), mean=0., stddev=1.)
        return self.z_mean + K.exp(self.z_log_var / 2) * epsilon
    """

    
    """
    def my_vae_loss(self,x, x_decoded_mean):
        xent_loss = keras.losses.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return xent_loss + kl_loss
    """
    
    def convert_args(self, **args):
        # convert sigle int/string arg int list of int/string
        for argument in args.keys():
            if not isinstance(args[argument], list):
                args[argument] = [args[argument] for i in range(self.n_blocks)]
        return args
        
    def check_padding(self):
        # check the minimum padding required for the input shape, to be divisible by the pooling factors
        if self.input_shape[0] % np.prod(self.encoder_args["pooling"]): 
            padding = abs(self.input_shape[0] - (
                math.ceil(self.input_shape[0] / np.prod(self.encoder_args["pooling"]))*(
                np.prod(self.encoder_args["pooling"]))))
            print("Input shape required {} padding".format(padding))
            return padding
        else: return None
    
    def get_decoder_args(self):
        # encoder args reversed
        decoder_args = dict()
        for argument in self.encoder_args.keys():
            decoder_args[argument] = self.encoder_args[argument][::-1]
        return decoder_args
    
    
    def build(self):
        if self.encoder_latent_layer_type == "variational":
            self.build_encoder_variational()
        else:
            self.build_encoder()
        self.build_decoder()
        self.build_autoencoder()
        return self.encoder, self.decoder, self.autoencoder
            
    
    def build_CNN_block(self, previous_layer, direction, filters, kernel_size, padding, 
                    activation, pooling):
        if direction == "downward":
            block = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding = padding)(previous_layer)
            block = keras.layers.normalization.BatchNormalization()(block)
            block = keras.layers.Activation(activation)(block)
            block = keras.layers.MaxPooling1D(pooling)(block)
        elif direction == "upward":
            block = keras.layers.UpSampling1D(size=pooling)(previous_layer)
            block = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding = padding)(block)
            block = keras.layers.normalization.BatchNormalization()(block)
            block = keras.layers.Activation(activation)(block)
        else: raise Exception("Block direction not valid")
        return block

    
    def build_CNN_blocks(self, previous_layer, direction, **args):
        for i in range(self.n_blocks):
            #print(args["filters"][i])
            block = self.build_CNN_block(previous_layer = previous_layer, 
                                         direction = direction,
                                         filters = args["filters"][i],
                                         kernel_size = args["kernel_size"][i],
                                         padding = args["padding"][i],
                                         activation = args["activation"][i],
                                         pooling = args["pooling"][i])
            previous_layer = block
        return block
    
    
    def build_encoder(self):
        input_layer = keras.layers.Input(shape=(self.input_shape))
        blocks = input_layer
        
        if self.padding:
            blocks = keras.layers.ZeroPadding1D((0, self.padding))(blocks)
        
        blocks = self.build_CNN_blocks(previous_layer = blocks, direction = "downward", **self.encoder_args)
        
        blocks = keras.layers.Conv1D(filters=1, kernel_size=1, padding = "same")(blocks)
        blocks = keras.layers.Activation('linear')(blocks)
        blocks = keras.layers.Flatten()(blocks)
        
        if self.encoder_latent_layer_type == "dense":
            blocks = keras.layers.Dense(self.latent_dim)(blocks)
        
        
        output_layer = blocks
        self.encoder = keras.models.Model(input_layer, output_layer, name = "Encoder")
        if self.verbose:
            print(self.encoder.summary())
        return self.encoder
    
    
    def build_encoder_variational(self):
        
        input_layer = keras.layers.Input(shape=(self.input_shape))
        blocks = input_layer
        
        if self.padding:
            blocks = keras.layers.ZeroPadding1D((0, self.padding))(blocks)
            
        blocks = self.build_CNN_blocks(previous_layer = blocks, direction = "downward", **self.encoder_args)
        
        blocks = keras.layers.Conv1D(filters=1, kernel_size=1, padding = "same")(blocks)
        blocks = keras.layers.Activation('linear')(blocks)
        blocks = keras.layers.Flatten()(blocks)
        
        #blocks = keras.layers.Dense(self.latent_dim)(blocks)
    
        z_mean = keras.layers.Dense(self.latent_dim, name='z_mean')(blocks)
        z_log_var = keras.layers.Dense(self.latent_dim, name='z_log_var')(blocks)
        z = keras.layers.Lambda(self.sampling, output_shape = (self.latent_dim,), name='z')([z_mean, z_log_var])

        #self.z_mean = z_mean
        #self.z_log_var = z_log_var
        
        output_layer = [z, z_mean, z_log_var] # bugged: the latent space is not normally distributed
        output_layer = z
        self.encoder = keras.models.Model(input_layer, output_layer, name='VariationalEncoder')
        if self.verbose:
            print(self.encoder.summary())
        return self.encoder
    
    
    
    def build_decoder(self):
        input_layer = keras.layers.Input(shape=(self.latent_dim,))
    
        blocks = input_layer
        
        # what encoder layer to look for the shape shenaningans
        if self.encoder_latent_layer_type == "dense":
            blocks = keras.layers.Dense(self.encoder.layers[-2].output_shape[1])(blocks)
            
        elif self.encoder_latent_layer_type == "variational":
            blocks = keras.layers.Dense(self.encoder.layers[-4].output_shape[1])(blocks)
        
        if self.encoder_latent_layer_type == "variational":
            blocks = keras.layers.Reshape(self.encoder.layers[-5].output_shape[1:])(blocks)   
        else:
            blocks = keras.layers.Reshape(self.encoder.layers[-3].output_shape[1:])(blocks)
        
        if self.encoder_latent_layer_type == "simple":                          
            blocks = keras.layers.Conv1D(filters=self.latent_dim, kernel_size=8, padding = "same")(blocks)
        
        blocks = self.build_CNN_blocks(blocks, "upward", **self.decoder_args)
        
        blocks = keras.layers.Conv1D(filters=1, kernel_size=1, padding = "same")(blocks)
        blocks = keras.layers.Activation('linear')(blocks)
        
        # crop the final series if the encoder has padding
        if self.padding:
            blocks = keras.layers.Cropping1D((0,self.padding))(blocks)
        
        output_layer = blocks
        self.decoder = keras.models.Model(input_layer, output_layer, name = "Decoder")
        if self.verbose:
            print(self.decoder.summary())
        return self.decoder
    
    def build_autoencoder(self):
        model_input = keras.layers.Input(shape=(self.input_shape), name = "Input")
        
        # DIFFERENTIATION NO LONGER NEEDED
        if self.encoder_latent_layer_type == "variational":
            output_encoder = self.encoder(model_input)#[2]
        else:
            output_encoder = self.encoder(model_input)
    
        
        output_decoder = self.decoder(output_encoder)
        
        self.inputs = model_input
        self.outputs = output_decoder
        
        self.autoencoder = keras.models.Model(model_input, output_decoder, name = "Autoencoder")
        
        if self.encoder_latent_layer_type == "variational":
            self.autoencoder.compile(optimizer='adam', loss=self.my_vae_loss, metrics = ["mse"])
            #self.autoencoder.compile(optimizer='adam', loss="mse")
            
        else:
            self.autoencoder.compile(optimizer='adam', loss='mse')
        
        if self.verbose:
            print(self.autoencoder.summary())
        
        return self.autoencoder
        









class DiscriminativeAutoencoder(Autoencoder):
    def __init__(self, 
                 #output_directory, 
                 input_shape,
                 encoder_args,
                 discriminator_args,
                 latent_dim,
                 n_blocks,
                 n_blocks_discriminator = 2,
                 encoder_latent_layer_type = "simple",
                 verbose=True,
                ):
        super(DiscriminativeAutoencoder, self).__init__(input_shape,
                                                        encoder_args,
                                                        latent_dim,
                                                        n_blocks,
                                                        encoder_latent_layer_type,
                                                        verbose)
        # input_shape: (n_timesteps, n_features)
        # encoder args: a dictionary with arguments: 
        #       (values are lists with 1 value per block or 1 int/string)
        #       ex. if n_blocks = 3: [1,2,3];  1 -> [1,1,1]; "same" -> ["same", "same", "same"]
        #       # filters: n of filters per CONV layer 
        #       # kernel_size
        #       # padding 
        #       # activation
        #       # pooling: (set to 1 to not do pooling)
        # n_blocks: number of blocks of the encoder (and decoder)
        # latent_dim: size of the latent dimension
        # encoder_latent_layer_type: simple, dense or variational
        #       for simple to work you must enter the correct latent_dim size else the decoder won't work
        # n_blocks_discriminator: number of blocks of the discriminator network
        # discriminator_args: a dictionary with arguments:
        #       (same format as the encoder args)
        #       # units: n of unit per block (layer)
        #       # activation
        
        self.n_blocks_discriminator = n_blocks_discriminator
        self.discriminator_args = self.convert_args_discriminator(**discriminator_args)
        self.discriminator = None
        self.history = None
    
    def convert_args_discriminator(self, **args):
        # convert sigle int/string arg int list of int/string
        for argument in args.keys():
            if not isinstance(args[argument], list):
                args[argument] = [args[argument] for i in range(self.n_blocks_discriminator)]
        return args
    
    def build(self):
        if self.encoder_latent_layer_type == "variational":
            self.build_encoder_variational()
        else:
            self.build_encoder()
        self.build_decoder()
        self.build_discriminator()
        self.build_autoencoder()
        return self.encoder, self.decoder, self.discriminator, self.autoencoder
    
    def build_discriminator(self):
        input_layer = keras.layers.Input(shape = (self.latent_dim,))
    
        blocks = input_layer
        blocks = self.build_dense_blocks(previous_layer = blocks, **self.discriminator_args)
        blocks = keras.layers.Dense(1, activation = "sigmoid")(blocks)
        
        output_layer = blocks
        self.discriminator = keras.models.Model(input_layer, output_layer, name = "Discriminator")
        if self.verbose:
            print(self.discriminator.summary())
        return self.discriminator
    
    def build_autoencoder(self):
        model_input = keras.layers.Input(shape=(self.input_shape), name = "Input")
        
        # DIFFERENTIATION NO LONGER NEEDED
        if self.encoder_latent_layer_type == "variational":
            output_encoder = self.encoder(model_input)#[2]
        else:
            output_encoder = self.encoder(model_input)
            
        output_decoder = self.decoder(output_encoder)
        output_discriminator = self.discriminator(output_encoder)
        
        self.inputs = model_input
        self.outputs = output_decoder
        
        optimizer = keras.optimizers.Adam(0.0002, 0.5)
        self.discriminator.trainable = False
        self.discriminator.compile(loss='binary_crossentropy', 
                                   optimizer=optimizer, 
                                   metrics=['accuracy'])
        
        self.autoencoder = keras.models.Model(model_input, 
                                              [output_decoder, output_discriminator], 
                                              name = "Discriminative Autoencoder")
        
        for layer in self.autoencoder.layers:
            if layer.name == "Discriminator":
                layer.trainable = False
        
        if self.encoder_latent_layer_type == "variational":
            self.autoencoder.compile(optimizer=optimizer, 
                                     loss=[self.my_vae_loss, 'binary_crossentropy'],
                                     #loss_weights=[0.99999, 0.00001],
                                     metrics = ["mse"])
            
        else:
            self.autoencoder.compile(loss=['mse', 'binary_crossentropy'], 
                                     loss_weights=[0.999, 0.001], 
                                     optimizer=optimizer)
        if self.verbose:
            print(self.autoencoder.summary())
        return self.autoencoder

    def build_dense_block(self, previous_layer, units, activation):
        block = keras.layers.Dense(units)(previous_layer)
        block = keras.layers.Activation(activation)(block)
        return block
        
    
    def build_dense_blocks(self, previous_layer, **args):
        for i in range(self.n_blocks_discriminator):
            block = self.build_dense_block(previous_layer = previous_layer, 
                                           units = args["units"][i],
                                           activation = args["activation"][i])
            previous_layer = block
        return block
    
    def custom_fit(self, 
                   data, 
                   targets, 
                   epochs = 100, 
                   batch_size = None, 
                   val_data = None, 
                   val_targets = None, 
                   debug = None, 
                   save_checkpoint = None,
                   filepath = None
                   ):
        if not batch_size: batch_size = data.shape[0]
        
        batches_per_epoch = data.shape[0] // batch_size # check if the dataset is integer divisible by the batch size
        batches = [batch_size for batch in range(batches_per_epoch)] # list of batch sizes
        
        # if the dataset is not integer divisible by the batch size the last batch will be equal to the reminder
        if len(data) % batch_size:
            batches.append(data.shape[0] % batch_size)

        # generate batch indexes (start_idx, end_idx, batch_size)
        idxs = []
        start = 0
        for batch in batches:
            end = start + batch
            idxs.append((start,end,np.abs(start - end)))
            start += batch

        
        autoencoder_losses = {"loss": [], "val_loss": []}
        min_val_loss = math.inf
        
        discriminator_losses = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
        
        # in the first epoch the discriminator needs recompiling else the weights won't update
        recompile = 1
        
        for epoch in range(epochs):
            random.shuffle(idxs) # randomize the order of the idxs
            for i in range(len(idxs)):
                ### DEBUG
                if debug: 
                    weights0, _ = self.discriminator.layers[3].get_weights()
                ###
                
                batch = idxs[i][2]
                
                valid = np.ones((batch, 1)) 
                fake = np.zeros((batch, 1))

                Xs = data[idxs[i][0]:idxs[i][1]] # data batch
                
                if self.encoder_latent_layer_type == "variational":
                    latent_fake = self.encoder.predict(Xs)#[2]
                else: 
                    latent_fake = self.encoder.predict(Xs)
                latent_real = np.random.normal(size=(batch, self.latent_dim))

                # Train the discriminator
                self.discriminator.trainable = True
                
                # in the first epoch the discriminator needs recompiling else the weights won't update
                if recompile == 1:
                    self.discriminator.compile(loss='binary_crossentropy', 
                                   optimizer=keras.optimizers.Adam(0.0002, 0.5), 
                                   metrics=['accuracy'])
                    recompile +=1
                
                
                Z = np.concatenate([latent_fake, latent_real])
                y = np.concatenate([fake, valid])
                discriminator_loss = self.discriminator.train_on_batch(Z, y)
                self.discriminator.trainable = False
                
                ### DEBUG
                if debug:
                    weights1, _ = self.discriminator.layers[3].get_weights()
                ###

                # Train the autoencoder
                autoencoder_loss = self.autoencoder.train_on_batch(Xs, [Xs, valid])
                
                ### DEBUG
                if debug:
                    weights2, _ = self.discriminator.layers[3].get_weights()
                    #weights3, _ = self.autoencoder.layers[3].layers[3].get_weights()
                    if np.array_equal(weights1, weights2) and not(np.array_equal(weights1, weights0)):
                        print("WEIGHTS OK")
                    else: 
                        print("WEIGHTS ERROR")
                        return [weights0, weights1, weights2]
                        
                ###
            
            # check the epoch autoencoder loss
            autoencoder_train_pred = self.autoencoder.predict(data)
            autoencoder_epoch_train_reconstruction_loss = mean_squared_error(data.flatten(), 
                                                              autoencoder_train_pred[0].flatten())
            autoencoder_losses["loss"].append(autoencoder_epoch_train_reconstruction_loss)
            
            autoencoder_epoch_val_reconstruction_loss = 0
            if val_data is not None:
                autoencoder_val_pred = self.autoencoder.predict(val_data)
                autoencoder_epoch_val_reconstruction_loss = mean_squared_error(val_data.flatten(), 
                                                            autoencoder_val_pred[0].flatten())
                autoencoder_losses["val_loss"].append(autoencoder_epoch_val_reconstruction_loss)
            
            if save_checkpoint == "validation_loss":
                if autoencoder_epoch_val_reconstruction_loss < min_val_loss:
                    min_val_loss = autoencoder_epoch_val_reconstruction_loss
                    self.autoencoder.save_weights(filepath + "+{:.6f}_.hdf5".format(min_val_loss))
            
            
            
            # discriminator loss & accuracy (of the last batch)
            discriminator_losses["acc"].append(discriminator_loss[1])
            discriminator_losses["loss"].append(discriminator_loss[0])
            
            # Plot the progress
            print("Epoch %d/%d, [DISC loss: %f, acc: %.2f%%] [AUT loss: %f, mse: %f, val_mse: %f]" % (
                epoch + 1, epochs, 
                discriminator_loss[0], 100 * discriminator_loss[1],
                autoencoder_loss[0], 
                autoencoder_epoch_train_reconstruction_loss, 
                autoencoder_epoch_val_reconstruction_loss))
            
        self.history = {"autoencoder": autoencoder_losses, "discriminator": discriminator_losses} 
        return self.history
    
if __name__ == "__main__":
    dataset = np.random.rand(100,194,1)
    n_timesteps = dataset.shape[1]
    
    dataset_val = np.random.rand(20,194,1)
    
    """
    params = {"input_shape": (n_timesteps,1),
          "n_blocks": 6, 
          "latent_dim": 71,
          "encoder_latent_layer_type": "variational",
          "encoder_args": {"filters":[2, 4,8,16,32,64], 
                            "kernel_size":[15,13,11,8,5,3], 
                            "padding":"same", 
                            "activation":"relu", 
                            "pooling":[1,1,1,1,1,1]},
          "discriminator_args": {"units": [100,100],
                                 "activation": "relu"},
          "n_blocks_discriminator": 2,
         }

    aut = DiscriminativeAutoencoder(**params)
    encoder, decoder, discriminator, autoencoder = aut.build()
    
    aut.custom_fit(data = dataset, 
                   targets = dataset, 
                   epochs = 4, 
                   batch_size = 10, 
                   val_data = dataset_val, 
                   val_targets = dataset_val, 
                   debug = None, 
                   save_checkpoint = "validation_loss",
                   filepath = "provaasalvare"
                   )
    
    """
    params = {"input_shape": (n_timesteps,1),
          "n_blocks": 6, 
          "latent_dim": 49,
          "encoder_latent_layer_type": "variational",
          "encoder_args": {"filters":[2,4,8,16,32,64], 
                            "kernel_size":[15,13,11,8,5,3], 
                            "padding":"same", 
                            "activation":"relu", 
                            "pooling":[1,1,1,1,2,2]}
        
         }

    aut = Autoencoder(verbose = False, **params)
    encoder, decoder, autoencoder = aut.build()
    autoencoder.fit(dataset, dataset, epochs=100)
    
    
    