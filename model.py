import pandas as pd 
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def load_dataset(filename):
    #TODO: Load dataset. Return a N x d dataset of features to learn from. 
    data = pd.read_csv("alldata.csv")
    data_frame = data_frame.drop('Year', axis=1)
    data_frame = data_frame.drop('Region/Country/Area', axis=1)
    data_frame.fillna(0, inplace=True)

    features=data_frame.iloc[:, :].values
    return features
    # return tfd.Beta([.5]*3,[.5]*3).sample(10)

def train(dataset, k=3, epochs=1000, learn_prior=False, print_step=10):
    """
    Returns trained clustering model.
    
    dataset: Numpy array of N samples of d features. [N,d]
    k: Number of components/clusters.
    epochs: Number of epochs to train for.
    learn_prior: Make prior a learnable.
    print_step: Printing period of loss.
    """
    d = tf.shape(dataset)[-1] # Number of Features.

    w = tf.Variable(tf.ones([k,2*d])) # [k,2d]
    pis = tf.Variable([1.]*k, trainable=learn_prior) # [k]

    model = tf.keras.Sequential(
                tfp.layers.DistributionLambda(lambda t: 
                    tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=pis),
                                          components_distribution=tfd.Independent(
                                            tfd.Beta(tf.abs(w[:,:d]),tf.abs(w[:,d:]))))
                    )
                )

    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(W):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(-model(None).log_prob(dataset))
        gradients = tape.gradient(loss, [W])
        optimizer.apply_gradients(zip(gradients, [W]))
        return loss

    for epoch in range(epochs):
        _loss = train_step(w)
        if epoch%print_step==0:
            print('STEP:{} NLL:{}'.format(epoch, _loss))
            print('W:{}'.format(w))
    
    print('Done training') 
    return model

def log_post(dataset, model):

    log_cat = tf.math.log(model(None).mixture_distribution.probs)[tf.newaxis,:]
    log_dem = model(None).log_prob(dataset)[:,tf.newaxis]
    log_lik = model(None).components_distribution.log_prob(dataset[:,tf.newaxis,:])

    log_post = log_cat - log_dem + log_lik

    return log_post

def test(dataset, model):
    """
    Returns model's test metrics.

    dataset: Numpy array of N samples of d features. [N,d]
    model: Trained Tensorflow Probability model
    """
    #TODO: Compute p-value and clustering graphs.
    MAP = tf.argmax(log_post(dataset, model),1)
    return MAP  

def main():
    dataset = load_dataset(None)
    train_model = train(dataset)
    MAP = test(dataset, train_model) 
