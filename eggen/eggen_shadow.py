"""
Explainable Graph Generation of Social Transactions

# Remove conds as inputs to discriminator. Change the follow variables:
self.disc_real, self.disc_fake, self.gradients, self.W_down_discriminator
"""

import tensorflow as tf
from eggen import utils
import time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import os
from matplotlib import pyplot as plt

class EGGen:    
    """
    EGGen class, an implicit generative model for graphs using random walks.
    """

    # def __init__(self, N, rw_len, walk_generator, n_conds, condgenerator_layers=[20], generator_layers=[50],
    #             discriminator_layers=[35], W_down_generator_size=128, W_down_discriminator_size=128, batch_size=128,
    #             condition_dim=42, noise_dim=16, noise_type="Gaussian", lr_gencond=0.01, lr_gen=0.0001, lr_disc=0.0001,
    #             gencond_iters=1, gen_iters=1, disc_iters=3, wasserstein_penalty=10, l2_penalty_generator=1e-7,
    #             l2_penalty_discriminator=5e-5, temp_start=5.0, min_temperature=0.5, temperature_decay=1-5e-5, seed=15,
    #             gpu_id=0, use_gumbel=True, legacy_generator=False, plot_show=True, sample_batch=1000
    #             ):
    def __init__(self, N, rw_len, walk_generator, n_conds, condgenerator_layers=[20], generator_layers=[50], discriminator_layers=[35], W_down_generator_size=128, W_down_discriminator_size=128, batch_size=128, sample_batch=1000, condition_dim=42, gencond_iters=1, gen_iters=1, disc_iters=3, wasserstein_penalty=10, l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5, lr_gencond=0.01, lr_gen=0.0001, lr_disc=0.0001, noise_dim=16, noise_type="Gaussian", temp_start=5.0, min_temperature=0.5, temperature_decay=1-5e-5, seed=15, use_gumbel=True, legacy_generator=False, gpu_id=0, plot_show=True
        ):

        """
        Initialize EGGen.

        Parameters
        ----------
        N: int
           Number of nodes in the graph to generate.
        rw_len: int
                Length of random walks to generate.
        walk_generator: function
                        Function that generates a single random walk and takes no arguments.
        generator_layers: list of integers, default: [40], i.e. a single layer with 40 units.
                          The layer sizes of the generator LSTM layers
        discriminator_layers: list of integers, default: [30], i.e. a single layer with 30 units.
                              The sizes of the discriminator LSTM layers
        W_down_generator_size: int, default: 128
                               The size of the weight matrix W_down of the generator. See our paper for details.
        W_down_discriminator_size: int, default: 128
                                   The size of the weight matrix W_down of the discriminator. See our paper for details.
        batch_size: int, default: 128
                    The batch size.
        noise_dim: int, default: 16
                   The dimension of the random noise that is used as input to the generator.
        noise_type: str in ["Gaussian", "Uniform], default: "Gaussian"
                    The noise type to feed into the generator.
        learning_rate: float, default: 0.0003
                       The learning rate.
        disc_iters: int, default: 3
                    The number of discriminator iterations per generator training iteration.
        wasserstein_penalty: float, default: 10
                             The Wasserstein gradient penalty applied to the discriminator. See the Wasserstein GAN
                             paper for details.
        l2_penalty_generator: float, default: 1e-7
                                L2 penalty on the generator weights.
        l2_penalty_discriminator: float, default: 5e-5
                                    L2 penalty on the discriminator weights.
        temp_start: float, default: 5.0
                    The initial temperature for the Gumbel softmax.
        min_temperature: float, default: 0.5
                         The minimal temperature for the Gumbel softmax.
        temperature_decay: float, default: 1-5e-5
                           After each evaluation, the current temperature is updated as
                           current_temp := max(temperature_decay*current_temp, min_temperature)
        seed: int, default: 15
              Random seed.
        gpu_id: int or None, default: 0
                The ID of the GPU to be used for training. If None, CPU only.
        use_gumbel: bool, default: True
                Use the Gumbel softmax trick.
        
        legacy_generator: bool, default: False
            If True, the hidden and cell states of the generator LSTM are initialized by two separate feed-forward networks. 
            If False (recommended), the hidden layer is shared, which has less parameters and performs just as good.

        # ++ Add ++                    
        condition_dim: int, default: 16
                   The dimension of the conditions that are used as input to the generator.            
        n_conds: int
                Number of different conditions the model can consider. 
        
        """

        self.params = {
            'condition_dim': condition_dim, # ++ Add ++
            'noise_dim': noise_dim,
            'noise_type': noise_type,
            'CondGenerator_Layers': condgenerator_layers, # ++ Add gen cond ++ 
            'Generator_Layers': generator_layers,
            'Discriminator_Layers': discriminator_layers,
            'W_Down_Generator_size': W_down_generator_size,
            'W_Down_Discriminator_size': W_down_discriminator_size,
            'l2_penalty_generator': l2_penalty_generator,
            'l2_penalty_discriminator': l2_penalty_discriminator,
            # 'learning_rate': learning_rate,
            'lr_gencond': lr_gencond, # ++ Add gen cond ++      
            'lr_gen': lr_gen, # ++ Add ++
            'lr_disc': lr_disc, # ++ Add ++
            'batch_size': batch_size,
            'Wasserstein_penalty': wasserstein_penalty,
            'temp_start': temp_start,
            'min_temperature': min_temperature,
            'temperature_decay': temperature_decay,
            'gencond_iters': gencond_iters, # ++ Add gen cond ++
            'gen_iters': gen_iters,
            'disc_iters': disc_iters,
            'use_gumbel': use_gumbel,
            'legacy_generator': legacy_generator,
            'plot_show': plot_show, # ++ Add ++       
            'sample_batch': sample_batch # ++ Add gen cond ++    
        }

        assert rw_len > 1, "Random walk length must be > 1."

        tf.set_random_seed(seed)

        self.N = N
        self.rw_len = rw_len
        # ++ Add ++
        self.n_conds = n_conds
        self.plot_show = plot_show

        self.noise_dim = self.params['noise_dim']
        self.gencond_layers = self.params['CondGenerator_Layers'] # ++ Add gen cond ++ 
        self.G_layers = self.params['Generator_Layers']
        self.D_layers = self.params['Discriminator_Layers']
        self.tau = tf.placeholder(1.0 , shape=(), name="temperature")

        # W_down and W_up for generator and discriminator
        self.W_down_generator = tf.get_variable('Generator.W_Down',
                                                shape=[self.N, self.params['W_Down_Generator_size']],
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer())
        # # ++ Add ++
        # self.W_down_generator = tf.get_variable('Generator.W_Down',
        #                                         shape=[self.N+self.params['condition_dim'], self.params['W_Down_Generator_size']],
        #                                         dtype=tf.float32,
        #                                         initializer=tf.contrib.layers.xavier_initializer())

        # self.W_down_discriminator = tf.get_variable('Discriminator.W_Down',
        #                                             shape=[self.N, self.params['W_Down_Discriminator_size']],
        #                                             dtype=tf.float32,
        #                                             initializer=tf.contrib.layers.xavier_initializer())
        # ++ Add ++
        self.W_down_discriminator = tf.get_variable('Discriminator.W_Down',
                                                    shape=[self.N+self.params['condition_dim'], self.params['W_Down_Discriminator_size']],
                                                    dtype=tf.float32,
                                                    initializer=tf.contrib.layers.xavier_initializer())

        self.W_up = tf.get_variable("Generator.W_up", shape = [self.G_layers[-1], self.N],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

        self.b_W_up = tf.get_variable("Generator.W_up_bias", dtype=tf.float32, initializer=tf.zeros_initializer,
                                      shape=self.N)

        # Using the generator_recurrent() defined below
        self.generator_function = self.generator_recurrent
        # Using the discriminator_recurrent() defined below
        self.discriminator_function = self.discriminator_recurrent

        # self.fake_inputs = self.generator_function(self.params['batch_size'], reuse=False, gumbel=use_gumbel, legacy=legacy_generator)
        # self.fake_inputs_discrete = self.generate_discrete(self.params['batch_size'], reuse=True,
        #                                                    gumbel=use_gumbel, legacy=legacy_generator)

        # Pre-fetch real random walks
        # dataset = tf.data.Dataset.from_generator(walk_generator, tf.int32, [self.params['batch_size'], self.rw_len])
        # ++ Add ++
        dataset = tf.data.Dataset.from_generator(walk_generator.cond_walk, (tf.int32,tf.int32), ([self.params['batch_size'], self.rw_len],[self.params['batch_size'], self.rw_len]))

        #dataset_batch = dataset.prefetch(2).batch(self.params['batch_size'])
        dataset_batch = dataset.prefetch(100)
        batch_iterator = dataset_batch.make_one_shot_iterator()
        # real_data = batch_iterator.get_next()
        # ++ Add ++
        real_data, realdata_conds = batch_iterator.get_next()

        self.real_inputs_discrete = real_data # Tensor("IteratorGetNext:0", shape=(128, 16), dtype=int32)
        # print("real_inputs_discrete: ", self.real_inputs_discrete)
        self.real_inputs = tf.one_hot(self.real_inputs_discrete, self.N) # Tensor("one_hot:0", shape=(128, 16, N), dtype=float32)
        # print("real_inputs: ", self.real_inputs)
        # ++ Add ++
        self.real_conds_discrete = realdata_conds # Tensor("IteratorGetNext:1", shape=(128, 16), dtype=int32)
        # print("real_conds_discrete: ", self.real_conds_discrete)
        # self.real_conds = tf.one_hot(self.real_conds_discrete, self.params['condition_dim']) # Tensor("one_hot_1:0", shape=(128, 16, 16), dtype=float32)
        self.real_conds = tf.one_hot(self.real_conds_discrete, self.params['condition_dim']) #, on_value=0.8,  off_value=0.05) # Label Smoothing
        # print("real_conds: ", self.real_conds)


        # ++ Add gen cond ++ generate 10000 real eval conds 
        eval_dataset = tf.data.Dataset.from_generator(walk_generator.conds_only, tf.int32, [self.params['sample_batch'], self.rw_len])
        eval_dataset_batch = eval_dataset.prefetch(100)
        eval_batch_iterator = eval_dataset_batch.make_one_shot_iterator()
        evaldata_conds = eval_batch_iterator.get_next()
        self.eval_conds_discrete = evaldata_conds
        self.eval_conds = tf.one_hot(self.eval_conds_discrete, self.params['condition_dim'])

        # # ++ Add ++ (NOT USED) generate 10000 real eval conds 
        # eval_dataset = tf.data.Dataset.from_generator(walk_generator.cond_only_walk, tf.int32, [10000, self.rw_len])
        # eval_dataset_batch = eval_dataset.prefetch(1000)
        # eval_batch_iterator = eval_dataset_batch.make_one_shot_iterator()
        # evaldata_conds = eval_batch_iterator.get_next()
        # self.eval_conds_discrete = evaldata_conds
        # # print("evaldata_conds: ", evaldata_conds)
        # self.eval_conds = tf.one_hot(self.eval_conds_discrete, self.params['condition_dim'])

        # # ++ Add ++ (NOT USED)
        # self.real_inputs_discrete = tf.unstack(real_data)[0]
        # self.real_inputs_discrete = real_data[0] #real_data:  Tensor("IteratorGetNext:0", shape=(128, 16), dtype=int32)
        # print("real_data: ", self.real_inputs_discrete)
        # self.real_inputs = tf.one_hot(self.real_inputs_discrete, self.N) # Tensor("one_hot:0", shape=(128, 16, 700), dtype=float32)
        # print("real_inputs: ", self.real_inputs)
        # self.real_inputs = tf.one_hot(self.real_inputs_discrete[0], self.N)
        # self.real_conds = tf.one_hot(self.real_inputs_discrete[1], self.N)

        # ++ Add gen cond ++      
        self.cond_generator = self.generate_conds
        self.gen_conds, self.y_conds = self.cond_generator(self.real_conds, reuse=tf.AUTO_REUSE) # False) #
        # print("self.gen_conds: ", self.gen_conds)
        # print("self.real_conds: ", self.real_conds)
        self.fake_inputs, self.fake_conds = self.generator_function(self.params['batch_size'], self.gen_conds, rw_len=self.rw_len, reuse=False, gumbel=use_gumbel, legacy=legacy_generator)

        # ++ Add gen cond ++ 
        self.eval_gen_conds, self.eval_y_conds = self.cond_generator(self.eval_conds, reuse=True) # reuse=False)

        # ++ Add ++        
        # self.fake_inputs, self.fake_conds = self.generator_function(self.params['batch_size'], self.real_conds, rw_len=self.rw_len, reuse=False, gumbel=use_gumbel, legacy=legacy_generator)
        # self.fake_inputs_discrete = self.generate_discrete(self.params['batch_size'], self.real_conds, reuse=True,
        #                                                    gumbel=use_gumbel, legacy=legacy_generator)
        # self.eval_conds = tf.tile(self.real_inputs_discrete, [79, 1]) # (1280000, 256)
        # self.eval_conds = tf.cast(self.eval_conds[:10000, :], dtype=tf.float32)
        # tf.reshape(tf.cast(tf.tile(self.real_inputs_discrete, [10000,16]), dtype=tf.float32),
                                    # [10000, self.params['condition_dim']])
        # self.eval_conds_discrete = self.generate_discrete(10000, self.repeated_realcond, reuse=True,
                                                           # gumbel=use_gumbel, legacy=legacy_generator)
        # self.eval_conds = tf.one_hot(self.eval_conds_discrete, self.params['condition_dim'])


        # self.disc_real = self.discriminator_function(self.real_inputs)
        # self.disc_fake = self.discriminator_function(self.fake_inputs, reuse=True)
        # ++ Add ++
        self.disc_real = self.discriminator_function(self.real_inputs, self.real_conds)
        self.disc_fake = self.discriminator_function(self.fake_inputs, self.fake_conds, reuse=True)

        self.disc_cost = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)
        self.gen_cost = -tf.reduce_mean(self.disc_fake)

        # WGAN lipschitz-penalty
        alpha = tf.random_uniform(
            shape=[self.params['batch_size'], 1, 1],
            minval=0.,
            maxval=1.
        )

        self.differences = self.fake_inputs - self.real_inputs
        self.interpolates = self.real_inputs + (alpha * self.differences)
        # self.gradients = tf.gradients(self.discriminator_function(self.interpolates, reuse=True), self.interpolates)[0]
        # ++ Add ++
        self.gradients = tf.gradients(self.discriminator_function(self.interpolates, self.fake_conds, reuse=True), self.interpolates)[0]
        self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), reduction_indices=[1, 2]))
        self.gradient_penalty = tf.reduce_mean((self.slopes - 1.) ** 2)
        self.disc_cost += self.params['Wasserstein_penalty'] * self.gradient_penalty

        # weight regularization; we omit W_down from regularization
        self.disc_l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                     if 'Disc' in v.name
                                     and not 'W_down' in v.name]) * self.params['l2_penalty_discriminator']
        self.disc_cost += self.disc_l2_loss

        # weight regularization; we omit  W_down from regularization
        self.gen_l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                     if 'Gen' in v.name
                                     and not 'W_down' in v.name]) * self.params['l2_penalty_generator']
        self.gen_cost += self.gen_l2_loss

        self.gen_params = [v for v in tf.trainable_variables() if 'Generator' in v.name]
        self.disc_params = [v for v in tf.trainable_variables() if 'Discriminator' in v.name]

        self.gen_train_op = tf.train.AdamOptimizer(learning_rate=self.params['lr_gen'], beta1=0.5,
                                                   beta2=0.9).minimize(self.gen_cost, var_list=self.gen_params)
        self.disc_train_op = tf.train.AdamOptimizer(learning_rate=self.params['lr_disc'], beta1=0.5,
                                                    beta2=0.9).minimize(self.disc_cost, var_list=self.disc_params)

        # ++ Add gen cond ++ Generate conds LSTM
        # Loss and optimizer   
        self.gencond_cost = tf.keras.losses.categorical_crossentropy(y_true=self.y_conds, y_pred=self.gen_conds)
        self.gencond_params = [v for v in tf.trainable_variables() if 'Gen_shadow' in v.name]
        self.cond_train_op = tf.train.AdamOptimizer(learning_rate=self.params['lr_gencond']).minimize(self.gencond_cost, var_list=self.gencond_params) 

        if gpu_id is None:
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
        else:
            gpu_options = tf.GPUOptions(visible_device_list='{}'.format(gpu_id), allow_growth=True)
            config = tf.ConfigProto(gpu_options=gpu_options)

        self.session = tf.InteractiveSession(config=config)
        self.init_op = tf.global_variables_initializer()

    # ++ Add gen cond ++ Generate conds LSTM
    def generate_conds(self, sampled_conds, reuse=None):
        with tf.variable_scope('Gen_shadow') as scope:
            if reuse is True:
                scope.reuse_variables()

            # prepare data for the LSTM
            y_conds = sampled_conds # shape=(128, 16, 42), dtype=float32
            # print("y_conds: ", y_conds.shape)

            # Defining the LSTM model 
            def lstm_cell(lstm_size):
                return tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
            self.gencond_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in self.gencond_layers])

            genconds_outputs, genconds_states = tf.contrib.rnn.static_rnn(cell=self.gencond_lstm, inputs=tf.unstack(sampled_conds, axis=1), dtype='float32')
            
            gen_conds = [tf.keras.layers.Dense(self.params['condition_dim'], activation='softmax')(output) for output in genconds_outputs]
            
            gen_conds = [tf.expand_dims(stack, 1) for stack in gen_conds]
            gen_conds = tf.concat(axis=1, values=gen_conds)

        return gen_conds, y_conds


    # def generate_discrete(self, n_samples, reuse=True, z=None, gumbel=True, legacy=False):
    # ++ Add ++
    def generate_discrete(self, n_samples, conds=None, rw_len=None, reuse=True, z=None, gumbel=True, legacy=False):
        """
        Generate a random walk in index representation (instead of one hot). This is faster but prevents the gradients
        from flowing into the generator, so we only use it for evaluation purposes.

        Parameters
        ----------
        n_samples: int
                   The number of random walks to generate.
        reuse: bool, default: None
               If True, generator variables will be reused.
        z: None or tensor of shape (n_samples, noise_dim)
           The input noise. None means that the default noise generation function will be used.
        gumbel: bool, default: False
            Whether to use the gumbel softmax for generating discrete output.
        legacy: bool, default: False
            If True, the hidden and cell states of the generator LSTM are initialized by two separate feed-forward networks. 
            If False (recommended), the hidden layer is shared, which has less parameters and performs just as good.
        
        Returns
        -------
                The generated random walks, shape [None, rw_len, N]


        """
        # return tf.argmax(self.generator_function(n_samples, reuse, z, gumbel=gumbel, legacy=legacy), axis=-1)
        # return tf.argmax(self.generator_function(n_samples, conds, reuse, z, gumbel=gumbel, legacy=legacy), axis=-1)
        # ++ Add ++
        if conds is None:
            # conds = tf.tile(self.real_conds, [79, 1, 1])[:10000, :] 
            conds = tf.tile(self.gen_conds, [(np.ceil(n_samples/self.params['batch_size']).astype('int32')), 1, 1])[:n_samples, :] # ++ Add gen cond ++ 
        elif conds is True:
            ## conds_gen, _ = self.cond_generator(conds, reuse=reuse)
            ## conds = tf.tile(conds_gen, [(np.ceil(n_samples/self.params['batch_size']).astype('int32')), 1, 1])[:n_samples, :]
            conds = tf.tile(self.eval_gen_conds, [(np.ceil(n_samples/self.params['sample_batch']).astype('int32')), 1, 1])[:n_samples, :]
        else:
            conds, _ = self.cond_generator(conds, reuse=reuse)

        # print("conds: ", conds)
        conds = tf.random_shuffle(conds)

        # print("conds: ", conds[0,0,:])
        # print('rw_len: ', rw_len)

        # generate_discreteinputs, _ = self.generator_function(n_samples, conds, rw_len, reuse, z, gumbel=gumbel, legacy=legacy)
        # return tf.argmax(generate_discreteinputs, axis=-1)
        generate_discreteinputs, generate_discreteconds = self.generator_function(n_samples, conds, rw_len, reuse, z, gumbel=gumbel, legacy=legacy)
        return tf.argmax(generate_discreteinputs, axis=-1), tf.argmax(generate_discreteconds, axis=-1)


    # def generator_recurrent(self, n_samples, reuse=None, z=None, gumbel=True, legacy=False):
    # ++ Add ++
    def generator_recurrent(self, n_samples, conds, rw_len, reuse=None, z=None, gumbel=True, legacy=False):
        """
        Generate random walks using LSTM.
        Parameters
        ----------
        n_samples: int
                   The number of random walks to generate.
        reuse: bool, default: None
               If True, generator variables will be reused.
        z: None or tensor of shape (n_samples, noise_dim)
           The input noise. None means that the default noise generation function will be used.
        gumbel: bool, default: False
            Whether to use the gumbel softmax for generating discrete output.
        legacy: bool, default: False
            If True, the hidden and cell states of the generator LSTM are initialized by two separate feed-forward networks. 
            If False (recommended), the hidden layer is shared, which has less parameters and performs just as good.
        Returns
        -------
        The generated random walks, shape [None, rw_len, N]

        """

        with tf.variable_scope('Generator') as scope:
            if reuse is True:
                scope.reuse_variables()

            # Defining the LSTM model 
            def lstm_cell(lstm_size):
                return tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)

            self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in self.G_layers])

            # initial states h and c are randomly sampled for each lstm cell
            if z is None:
                initial_states_noise = make_noise([n_samples, self.noise_dim], self.params['noise_type']) # n_samples, self.noise_dim (128, 16)
            else:
                initial_states_noise = z
            initial_states = []

            # Noise preprocessing
            for ix,size in enumerate(self.G_layers):
                if legacy: # old version to initialize LSTM. new version has less parameters and performs just as good.
                    h_intermediate = tf.layers.dense(initial_states_noise, size, name="Generator.h_int_{}".format(ix+1),
                                                     reuse=reuse, activation=tf.nn.tanh)
                    h = tf.layers.dense(h_intermediate, size, name="Generator.h_{}".format(ix+1), reuse=reuse,
                                        activation=tf.nn.tanh)

                    c_intermediate = tf.layers.dense(initial_states_noise, size, name="Generator.c_int_{}".format(ix+1),
                                                     reuse=reuse, activation=tf.nn.tanh)
                    c = tf.layers.dense(c_intermediate, size, name="Generator.c_{}".format(ix+1), reuse=reuse,
                                        activation=tf.nn.tanh)
                    
                else:
                    intermediate = tf.layers.dense(initial_states_noise, size, name="Generator.int_{}".format(ix+1),
                                                     reuse=reuse, activation=tf.nn.tanh)
                    print("intermediate: ", intermediate) # shape=(128, 40)
                    h = tf.layers.dense(intermediate, size, name="Generator.h_{}".format(ix+1), reuse=reuse,
                                        activation=tf.nn.tanh)
                    print("h: ", h) # shape=(128, 40)
                    c = tf.layers.dense(intermediate, size, name="Generator.c_{}".format(ix+1), reuse=reuse,
                                        activation=tf.nn.tanh)
                    print("c: ", c) # shape=(128, 40)
                initial_states.append((c, h)) # shape=((128,40),(128,40))
                # print("enumerate ix and size: ", ix, size)
                print("Generator initial_states: ", len(initial_states))

            state = initial_states # shape=((128,40),(128,40))
            # print("Initial generator state: ", state)
            inputs = tf.zeros([n_samples, self.params['W_Down_Generator_size']]) # zeros with shape=(128, 128)
            # print("Initial generator inputs: ", inputs) 
            outputs = []

            # ++ Add ++
            # Setting the conditions for each random walk
            # print("original conds:", conds) # Tensor("one_hot_1:0", shape=(128, 16, 128), dtype=float32)
            if conds is None:
                # conds = tf.zeros([n_samples, self.rw_len, self.params['condition_dim']]) # zeros with shape=(n_samples=128, rwlen=16, condition_dim=16)                
                # conds = tf.random_uniform([n_samples, self.rw_len, self.params['condition_dim']], 0, self.n_conds, seed=39)
                """ randint 0 to n_conds """
                # conds_discrete = np.random.randint(low=0, high=self.n_conds, size=[n_samples, self.rw_len])
                # conds = tf.one_hot(conds_discrete, self.params['condition_dim']) 

                """ self.eval_conds 10000"""
                # conds = tf.tile(self.real_conds, [79, 1, 1])[:10000, :] 
                # print("eval conds: ", conds[:5, :5])
                conds_unstack = tf.unstack(conds, axis=1)    
            else:
                # Unpack the conds tensor into rw_len tensors
                conds_unstack = tf.unstack(conds, axis=1)
                # print("unstacking conds:", conds_unstack)
                # for cond_tensor in conds_unstack[:-1]:                
                #     print("one cond tensor: ", cond_tensor)

            # ++ Add ++            
            # cond = tf.zeros([n_samples, self.params['condition_dim']]) # zeros with shape=(128, 42)
            cond = conds_unstack[0] # shape=(128, 42)
            print("Initial cond: ", cond)
            # # ++ Add ++
            # inputs = tf.zeros([n_samples, self.N]) # zeros with shape=(128, N)
            generator_inputs = tf.concat([inputs, cond], 1) # zeros with shape=(128, N+42)
            # print("Initial generator_inputs: ", generator_inputs)
            # generator_inputs = tf.matmul(generator_inputs, self.W_down_generator)

            # LSTM time steps
            for i in range(rw_len): #self.rw_len):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()                    
                    # print("Loop num: % 2d" %(i))  
                    # ++ Add ++
                    # print("cond tensor: ", conds_unstack[i-1])
                    # cond = self.real_inputs_discrete[1] #tf.zeros([n_samples, 1]) # zeros with shape=(128, 1)
                    # cond = tf.zeros([n_samples, self.params['W_Down_Generator_size']]) # zeros with shape=(128, 128)
                    cond = conds_unstack[i] #-1] # shape=(128, n_conds)
                    # print("gen_input: ", generator_inputs)
                    # print("cond: ", cond)
                    generator_inputs = tf.concat([generator_inputs, cond], 1) # shape=(128, 128+42)
                    # print("gen_input: ", generator_inputs) # Tensor("Generator/concat_2:0", shape=(128, 128+42), dtype=float32)
                    # print("state: ", state) # shape=((128, 40),(128, 40))


                # Get LSTM output
                # output, state = self.stacked_lstm.call(inputs, state)
                # ++ Add ++
                output, state = self.stacked_lstm.call(generator_inputs, state)          

                # print("LSTM output: ", output) #shape=(128, 40)
                # print("LSTM state: ", state) #shape=(128, 40)

                # Blow up to dimension N using W_up
                output_bef = tf.matmul(output, self.W_up) + self.b_W_up
                # print("output_bef: ", output_bef) #shape=(128, N)

                # Perform Gumbel softmax to ensure gradients flow
                if gumbel:
                    output = gumbel_softmax(output_bef, temperature=self.tau, hard = True)
                else:
                    output = tf.nn.softmax(output_bef)
                # print("softmax output: ", output) #shape=(128, N)
                # print("W_down_generator: ", self.W_down_generator) #shape=(N, 128)

                # Back to dimension d
                # inputs = tf.matmul(output, self.W_down_generator)
                # print("size-reduced inputs: ", inputs) #shape=(128, 128)

                # ++ Add ++
                # if i > 0:
                #     tf.get_variable_scope().reuse_variables()                    
                #     cond = conds_unstack[i-1] # shape=(128, n_conds)
                #     # print("cond: ", cond)
                #     gen_output = tf.concat([output, cond], 1) # shape=(128, N+42)
                    # generator_inputs = tf.matmul(gen_output, self.W_down_generator)

                generator_inputs = tf.matmul(output, self.W_down_generator)
                    # print("size-reduced inputs: ", generator_inputs) #shape=(128, 128)
                
                outputs.append(output)

            outputs = tf.stack(outputs, axis=1) #shape=(128, 16, N)
            # print("fake outputs: ", outputs)
        return outputs, conds


    # ++ Add ++
    def discriminator_recurrent(self, inputs, conds, reuse=None):
        """
        Discriminate real from fake random walks using LSTM.
        Parameters
        ----------
        inputs: tf.tensor, shape (None, rw_len, N)
                The inputs to process
        reuse: bool, default: None
               If True, discriminator variables will be reused.

        Returns
        -------
        final_score: tf.tensor, shape [None,], i.e. a scalar
                     A score measuring how "real" the input random walks are perceived.

        """

        with tf.variable_scope('Discriminator') as scope:
            if reuse == True:
                scope.reuse_variables()

            input_reshape = tf.reshape(inputs, [-1, self.N]) # Tensor("Discriminator/Reshape:0", shape=(2048, N), dtype=float32)
            # print("disc recurrent input_reshape: ", input_reshape)
            # output = tf.matmul(input_reshape, self.W_down_discriminator) # Tensor("Discriminator/MatMul:0", shape=(2048, 128), dtype=float32)
            # print("disc recurrent output: ", output)
            # ++ Add ++
            conds_reshape = tf.reshape(conds, [-1, self.params['condition_dim']]) # Tensor("Discriminator_2/Reshape_1:0", shape=(2048, 42), dtype=float32)
            output_cond = tf.concat([input_reshape, conds_reshape], 1) # Tensor("Discriminator_2/concat:0", shape=(2048, N+cond_dim), dtype=float32)
            output = tf.matmul(output_cond, self.W_down_discriminator) # Tensor("Discriminator/MatMul:0", shape=(2048, 128), dtype=float32)
            # print("disc recurrent output: ", output)
            output = tf.reshape(output, [self.params['batch_size'], self.rw_len, -1]) # shape=(128, 16, 144)


            # # ++ Add ++
            # conds_reshape = tf.reshape(conds, [-1, self.params['condition_dim']]) # Tensor("Discriminator_2/Reshape_1:0", shape=(2048, 16), dtype=float32)
            # # print("conds_reshape: ", conds_reshape)
            # output_cond = tf.concat([output, conds_reshape], 1) # Tensor("Discriminator_2/concat:0", shape=(2048, 144), dtype=float32)
            # # print("output_cond: ", output_cond)
            # output_cond = tf.reshape(output_cond, [self.params['batch_size'], self.rw_len, -1]) # Tensor("Discriminator_2/Reshape_2:0", shape=(128, 16, 144), dtype=float32)
            # # print("output_cond reshaped: ", output_cond)
            # output = output_cond

            # # output = tf.reshape(output, [-1, self.rw_len, int(self.W_down_discriminator.get_shape()[-1])]) # Tensor("Discriminator/Reshape_1:0", shape=(128, 16, 128), dtype=float32)
            # # # print("disc recurrent output reshaped: ", output)
            # # print("unstacked output: ", len(tf.unstack(output, axis=1))) # len=16 of <tf.Tensor 'Discriminator_2/unstack:0' shape=(128, 128) dtype=float32>

            def lstm_cell(lstm_size):
                return tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)

            disc_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in self.D_layers])

            output_disc, state_disc = tf.contrib.rnn.static_rnn(cell=disc_lstm_cell, inputs=tf.unstack(output, axis=1),
                                                              dtype='float32')

            last_output = output_disc[-1]

            final_score = tf.layers.dense(last_output, 1, reuse=reuse, name="Discriminator.Out")
            return final_score
    # def discriminator_recurrent(self, inputs, reuse=None):
    #     with tf.variable_scope('Discriminator') as scope:
    #         if reuse == True:
    #             scope.reuse_variables()

    #         input_reshape = tf.reshape(inputs, [-1, self.N])
    #         # print("input_reshape: ", input_reshape)
    #         # print("self.W_down_discriminator: ", self.W_down_discriminator)

    #         output = tf.matmul(input_reshape, self.W_down_discriminator)
    #         # print("output: ", output)
    #         output = tf.reshape(output, [-1, self.rw_len, int(self.W_down_discriminator.get_shape()[-1])])

    #         def lstm_cell(lstm_size):
    #             return tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)

    #         disc_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in self.D_layers])

    #         output_disc, state_disc = tf.contrib.rnn.static_rnn(cell=disc_lstm_cell, inputs=tf.unstack(output, axis=1),
    #                                                           dtype='float32')

    #         last_output = output_disc[-1]

    #         final_score = tf.layers.dense(last_output, 1, reuse=reuse, name="Discriminator.Out")
    #         return final_score

    def train(self, A_orig, val_ones, val_zeros,  max_iters=50000, stopping=None, eval_transitions=15e6,
              transitions_per_iter=150000, max_patience=5, eval_every=500, plot_every=-1, save_directory="./snapshots_shadow",
              model_name=None, continue_training=False): 
        """

        Parameters
        ----------
        A_orig: sparse matrix, shape: (N,N)
                Adjacency matrix of the original graph to be trained on.
        val_ones: np.array, shape (n_val, 2)
                  The indices of the hold-out set of validation edges
        val_zeros: np.array, shape (n_val, 2)
                  The indices of the hold-out set of validation non-edges
        max_iters: int, default: 50,000
                   The maximum number of training iterations if early stopping does not apply.
        stopping: float in (0,1] or None, default: None
                  The early stopping strategy. None means VAL criterion will be used (i.e. evaluation on the
                  validation set and stopping after there has not been an improvement for *max_patience* steps.
                  Set to a value in the interval (0,1] to stop when the edge overlap exceeds this threshold.
        eval_transitions: int, default: 15e6
                          The number of transitions that will be used for evaluating the validation performance, e.g.
                          if the random walk length is 5, each random walk contains 4 transitions.
        transitions_per_iter: int, default: 150000
                              The number of transitions that will be generated in one batch. Higher means faster
                              generation, but more RAM usage.
        max_patience: int, default: 5
                      Maximum evaluation steps without improvement of the validation accuracy to tolerate. Only
                      applies to the VAL criterion.
        eval_every: int, default: 500
                    Evaluate the model every X iterations.
        plot_every: int, default: -1
                    Plot the generator/discriminator losses every X iterations. Set to None or a negative number
                           to disable plotting.
        save_directory: str, default: "../snapshots"
                        The directory to save model snapshots to.
        model_name: str, default: None
                    Name of the model (will be used for saving the snapshots).
        continue_training: bool, default: False
                           Whether to start training without initializing the weights first. If False, weights will be
                           initialized.

        Returns
        -------
        log_dict: dict
                  A dictionary with the following values observed during training:
                  * The generator and discriminator losses
                  * The validation performances (ROC and AP)
                  * The edge overlap values between the generated and original graph
                  * The sampled graphs for all evaluation steps.

        """

        if stopping == None:  # use VAL criterion
            best_performance = 0.0
            patience = max_patience
            print("**** Using VAL criterion for early stopping ****")

        else:  # use EO criterion
            assert "float" in str(type(stopping)) and stopping > 0 and stopping <= 1
            print("**** Using EO criterion of {} for early stopping".format(stopping))

        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        if model_name is None:
            # Find the file corresponding to the lowest vacant model number to store the snapshots into.
            model_number = 0
            while os.path.exists("{}/model_best_{}.ckpt".format(save_directory, model_number)):
                model_number += 1
            save_file = "{}/model_best_{}.ckpt".format(save_directory, model_number)
            open(save_file, 'a').close()  # touch file
        else:
            save_file = "{}/{}_best.ckpt".format(save_directory, model_name)
        print("**** Saving snapshots into {} ****".format(save_file))

        if not continue_training:
            print("**** Initializing... ****")
            self.session.run(self.init_op)
            print("**** Done.           ****")
        else:
            print("**** Continuing training without initializing weights. ****")

        # Validation labels
        actual_labels_val = np.append(np.ones(len(val_ones)), np.zeros(len(val_zeros)))
        # print("actual_labels_val: ", actual_labels_val.shape)

        # Some lists to store data into.
        gencond_losses = [] # ++ Add gen cond ++
        gen_losses = []
        disc_losses = []
        graphs = []
        val_performances = []
        eo=[]
        temperature = self.params['temp_start']

        starting_time = time.time()
        saver = tf.train.Saver()

        transitions_per_walk = self.rw_len - 1
        # # Sample lots of random walks, used for evaluation of model.
        # sample_many_count = int(np.round(transitions_per_iter/transitions_per_walk)) #10000
        # sample_many = self.generate_discrete(sample_many_count, reuse=True)
        # ++ Add ++
        sample_many_count = 10000
        # sample_many = self.generate_discrete(sample_many_count, conds=self.real_conds, reuse=True) #shape=(10000, 16)
        # sample_many = self.generate_discrete(sample_many_count, conds=None, rw_len=self.rw_len, reuse=True) #shape=(10000, 16)
        # ++ Add gen cond ++
        # sample_many = self.generate_discrete(sample_many_count, conds=self.gen_conds, rw_len=self.rw_len, reuse=True) 
        # sample_many = self.generate_discrete(sample_many_count, conds=None, rw_len=self.rw_len, reuse=True) 
        # sample_many, explain_conds = self.generate_discrete(self.params['sample_batch'], conds=True, rw_len=self.rw_len, reuse=True) #10000
        sample_many, explain_conds = self.generate_discrete(sample_many_count, conds=None, rw_len=self.rw_len, reuse=True) #10000
        print("sample_many: ", sample_many)

        n_eval_walks = eval_transitions/transitions_per_walk #1000000.0
        n_eval_iters = int(np.round(n_eval_walks/sample_many_count)) #100

        print("**** Starting training. ****")

        for _it in range(max_iters):

            # print("real_inputs: ", np.nonzero(self.session.run(self.real_inputs)))
            # print("real_conds: ", np.nonzero(self.session.run(self.real_conds)))

            if _it > 0 and _it % (2500) == 0:
                t = time.time() - starting_time
                print('{:<7}/{:<8} training iterations, took {} seconds so far...'.format(_it, max_iters, int(t)))

            # ++ Add gen cond ++   
            # # train LSTM shadow generator
            _gencond_l = []
            for _ in range(self.params['gencond_iters']):
                gencond_loss, _ = self.session.run([self.gencond_cost, self.cond_train_op])
                _gencond_l.append(gencond_loss)

            # Generator training iteration
            # ++ Add ++
            _gen_l = []
            for _ in range(self.params['gen_iters']):
                gen_loss, _ = self.session.run([self.gen_cost, self.gen_train_op],
                                           feed_dict={self.tau: temperature})
                _gen_l.append(gen_loss)

            _disc_l = []
            # Multiple discriminator training iterations.
            for _ in range(self.params['disc_iters']):
                disc_loss, _ = self.session.run(
                    [self.disc_cost, self.disc_train_op],
                    feed_dict={self.tau: temperature}
                )
                _disc_l.append(disc_loss)

            gencond_losses.append(np.mean(_gencond_l)) # ++ Add gen cond ++  
            # gen_losses.append(gen_loss)
            gen_losses.append(np.mean(_gen_l)) # ++ Add ++
            disc_losses.append(np.mean(_disc_l))

            # Evaluate the model's progress.
            if _it > 0 and _it % eval_every == 0:

                # Sample lots of random walks.
                smpls = []
                for _ in range(n_eval_iters):
                    smpls.append(self.session.run(sample_many, {self.tau: 0.5}))
                    # print("eval conds: ", self.session.run(self.real_conds_discrete[:3, :3]))

                # Compute score matrix
                gr = utils.score_matrix_from_random_walks(np.array(smpls).reshape([-1, self.rw_len]), self.N)
                gr = gr.tocsr() # shape=(N, N)                
                # print("gr: ", gr.shape)

                # Assemble a graph from the score matrix
                _graph = utils.graph_from_scores(gr, A_orig.sum()) # shape=(N, N)

                # Compute edge overlap
                edge_overlap = utils.edge_overlap(A_orig.toarray(), _graph)
                graphs.append(_graph)
                eo.append(edge_overlap)

                # print("gr val_ones: ", gr[tuple(val_ones.T)].A1.shape)
                # print("gr val_zeros: ", gr[tuple(val_zeros.T)].A1.shape)
                edge_scores = np.append(gr[tuple(val_ones.T)].A1, gr[tuple(val_zeros.T)].A1)
                # print("edge_scores: ", edge_scores.shape)

                # Compute Validation ROC-AUC and average precision scores.
                val_performances.append((roc_auc_score(actual_labels_val, edge_scores),
                                               average_precision_score(actual_labels_val, edge_scores)))

                # Update Gumbel temperature
                temperature = np.maximum(self.params['temp_start'] * np.exp(-(1-self.params['temperature_decay']) * _it),
                                         self.params['min_temperature'])

                # print("**** Iter {:<6} Val ROC {:.3f}, AP: {:.3f}, EO {:.3f}, Gen Loss {:.3f}, Disc Loss: {:.3f} ****".format(_it,
                #                                                                val_performances[-1][0],
                #                                                                val_performances[-1][1],
                #                                                                edge_overlap/A_orig.sum(),
                #                                                                gen_losses[-1],
                #                                                                disc_losses[-1]
                #                                                                ))
                print("**** Iter {:<6} Val ROC {:.3f}, AP: {:.3f}, EO {:.3f}, Gen Loss {:.3f}, Disc Loss: {:.3f}, Gen Cond Loss {:.3f} ****".format(_it,
                                                                               val_performances[-1][0],
                                                                               val_performances[-1][1],
                                                                               edge_overlap/A_orig.sum(),
                                                                               gen_losses[-1],
                                                                               disc_losses[-1],
                                                                               gencond_losses[-1]
                                                                               ))

                if stopping is None:   # Evaluate VAL criterion
                    if np.sum(val_performances[-1]) > best_performance:
                        # New "best" model
                        best_performance = np.sum(val_performances[-1])
                        patience = max_patience
                        _ = saver.save(self.session, save_file)
                    else:
                        patience -= 1

                    if patience == 0:
                        print("**** EARLY STOPPING AFTER {} ITERATIONS ****".format(_it))
                        break
                elif edge_overlap/A_orig.sum() >= stopping:   # Evaluate EO criterion
                    print("**** EARLY STOPPING AFTER {} ITERATIONS ****".format(_it))
                    break
            
            # if plot_every > 0 and (_it+1) % plot_every == 0:
            # ++ Add ++
            if self.plot_show is True and plot_every > 0 and (_it+1) % plot_every == 0:        
                if len(disc_losses) > 10:
                    plt.plot(disc_losses[9::], label="Critic loss")
                    plt.plot(gen_losses[9::], label="Generator loss")
                else:
                    plt.plot(disc_losses, label="Critic loss")
                    plt.plot(gen_losses, label="Generator loss")
                plt.legend()
                plt.show()

        print("**** Training completed after {} iterations. ****".format(_it))
        # ++ Add ++
        if self.plot_show is True:
            plt.plot(disc_losses[9::], label="Critic loss")
            plt.plot(gen_losses[9::], label="Generator loss")
            plt.legend()
            plt.show()

        if stopping is None:
            saver.restore(self.session, save_file)
        #### Training completed.
        log_dict = {"disc_losses": disc_losses, 'gen_losses': gen_losses, 'val_performances': val_performances,
                    'edge_overlaps': eo, 'generated_graphs': graphs}
        return log_dict


def make_noise(shape, type="Gaussian"):
    """
    Generate random noise.

    Parameters
    ----------
    shape: List or tuple indicating the shape of the noise
    type: str, "Gaussian" or "Uniform", default: "Gaussian".

    Returns
    -------
    noise tensor

    """

    if type == "Gaussian":
        noise = tf.random_normal(shape)
    elif type == 'Uniform':
        noise = tf.random_uniform(shape, minval=-1, maxval=1)
    else:
        print("ERROR: Noise type {} not supported".format(type))
    return noise


def sample_gumbel(shape, eps=1e-20):
    """
    Sample from a uniform Gumbel distribution. Code by Eric Jang available at
    http://blog.evjang.com/2016/11/tutorial-categorical-variational.html
    Parameters
    ----------
    shape: Shape of the Gumbel noise
    eps: Epsilon for numerical stability.

    Returns
    -------
    Noise drawn from a uniform Gumbel distribution.

    """
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
      """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        # y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)), y.dtype)
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y