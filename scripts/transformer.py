"""
This script contains model structure of the transformer model used to embedding
patient visits.

Some codes bellow are borrowed (such as CustomSchedule class) from the
tensorflow website: https://www.tensorflow.org/text/tutorials/transformer
for reference purposes.
"""

# custom learning rate, setup warmup steps
# define loss function
# mask some labels but restore the whole prediction
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=500):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2, output_type=tf.int32))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

def random_mask(data_batch,fraction = .2, MAX_LENGTH = 250):
    L = np.argmin(data_batch)
    mask_position = np.random.choice(L,int(fraction * L))
    data_batch = [0 if i in set(mask_position) else data_batch[i] for i in range(len(data_batch))]
    return data_batch


### defining objects for the model

if __name__ == "__main__":
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    d_model=50
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                        beta_1=0.9,
                                        beta_2=0.98,
                                        epsilon=1e-9)

    transformer = model.Transformer(num_layers=6, # layer, L
        d_model=200, # embeddings of  feature, H
        num_heads=10, # numbers of head A
        dff=2048,
        input_vocab_size=len(namelist),
        embedding_matrix = np.concatenate((np.zeros(shape = (1,50)),embd_data)))

    import multiprocessing
    import logging
    import time


    def train(EPOCHS=EPOCHS):

        checkpoint_path = 'C:/Data/Lab/PatEmbedding/eMERGE/pat_embd/checkpoints/testrun/'

        ckpt = tf.train.Checkpoint(transformer=transformer,
                                   optimizer=optimizer)

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

        # If a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        for epoch in range(EPOCHS):
            train_loss.reset_states()
            train_accuracy.reset_states()
            # inp -> portuguese, tar -> english
            for (batch,data) in enumerate(train_batches):
                tar, inp = data[0], data[1]
                train_step(inp, tar)

                if batch % 200 == 0:
                    print(f'Epoch {epoch + 1} Batch \
{batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

            ckpt_save_path = ckpt_manager.save()

        return None

    EPOCHS = 10
    p = multiprocessing.Process(target=train(EPOCHS = EPOCHS))
    p.start()
    p.join()
