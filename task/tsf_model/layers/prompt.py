import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class SoftPrompts(tf.keras.layers.Layer):
    '''Soft prompts with prompt pooling.'''
    def __init__(
            self,
            key_dims,
            embedding_dims,
            nr_of_query_patches,
            prompt_pool_size,
            nr_of_most_similar_prompts,
            custom_prompt_keys=None,
            **kwargs):
        '''
        A prompt pool is a list key-value pairs.
        Prompt value is a trainable weight with the shape of
            (prompt_length, embedding_dims)
        A prompt key is a vector that has shape of
            (1, key_dims).

        args:
            key_dims (int) - key dimension.
            embedding_dims (int) -  embedding dimension.
            nr_of_query_patches (int) - nr of timesteps of the input query.
            prompt_pool_size (int) - number of the prompts in the pool.
            nr_of_most_similar_prompts (int) - number of the most similar
                prompts to be returned.
            custom_prompt_keys (list) - custom prompt keys.
        '''

        super().__init__(**kwargs)
        self.embedding_dims = embedding_dims
        self.nr_of_most_similar_prompts = nr_of_most_similar_prompts
        self.prompt_pool_size = max(prompt_pool_size, nr_of_most_similar_prompts * 2)
        self.key_dims = key_dims
        self.reshaper = tf.keras.layers.Reshape((key_dims,))
        self.custom_prompt_keys = custom_prompt_keys
        self.nr_of_query_patches = nr_of_query_patches

    def build(self, input_shape):
        # ---- stable config values ----
        key_dims = self.key_dims
        embedding_dims = self.embedding_dims
        prompt_pool_size = self.prompt_pool_size
        nr_similar = self.nr_of_most_similar_prompts
        query_patches = self.nr_of_query_patches

        # ---- derive prompt length safely ----
        # total prompt-related input should be half of the actual input.
        prompt_length = (query_patches // 2) // nr_similar

        # ---- prompt keys ----
        if self.custom_prompt_keys is not None:
            custom_keys = tf.convert_to_tensor(
                self.custom_prompt_keys,
                dtype=self.dtype or tf.float32
            )

            key_initializer = tf.keras.initializers.Constant(custom_keys)
            key_shape = tuple(custom_keys.shape)
        else:
            key_initializer = tf.keras.initializers.RandomNormal()
            key_shape = (prompt_pool_size, key_dims)

        self.prompt_keys = self.add_weight(
            name="prompt_keys",
            shape=key_shape,
            initializer=key_initializer,
            trainable=False
        )

        # ---- prompt values ----
        self.prompt_values = self.add_weight(
            name="prompt_values",
            shape=(prompt_pool_size, prompt_length, embedding_dims),
            initializer="random_uniform",
            trainable=True
        )

        # ---- IMPORTANT: mark built ----
        super().build(input_shape)

    def call(self, inputs):
        '''
        Calculates the most similar prompts by keys.
        Returns the values of them.

        inputs: (None, timesteps, covariates)
        returns: (None, timesteps, features)
        '''
        # reduce multivariate to univariate
        # since expected input is already instance normalized,
        # there should not be problem with difference of
        # scales between the covariates.
        x = self.reshaper(inputs)

        # Compute cosine similarity between inputs and prompt keys
        x_norm = tf.nn.l2_normalize(x, axis=1)
        prompt_key_norm = tf.nn.l2_normalize(
            self.prompt_keys,
            axis=1)  # (key_dims, timesteps)
        similarity_scores = tf.matmul(
            x_norm,
            prompt_key_norm,
            transpose_b=True)  # (batch_size, key_dims)

        # Get indices of most similar prompts
        top_indices = tf.math.top_k(
            similarity_scores,
            k=self.nr_of_most_similar_prompts).indices

        # Gather the values of the most similar prompts
        most_similar_prompt_values = \
            tf.gather(self.prompt_values, top_indices)

        # Reshape to required shape using tf.concat
        most_similar_prompt_values = \
            tf.concat(
                tf.unstack(most_similar_prompt_values, axis=1),
                axis=1)

        return most_similar_prompt_values

    def get_config(self):
        config = super().get_config()
        config.update({
            'key_dims': self.key_dims,
            'embedding_dims': self.embedding_dims,
            'nr_of_query_patches': self.nr_of_query_patches,
            'prompt_pool_size': self.prompt_pool_size,
            'nr_of_most_similar_prompts': self.nr_of_most_similar_prompts,
            'custom_prompt_keys': self.custom_prompt_keys
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
