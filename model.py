import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def load_dataset(year='None'):
    #TODO:Return a N x d dataset of features to learn from and other features for plotting as well. 
    # If year is None, then return all the data; else only return data corresponding to that year
    data = pd.read_csv("alldata.csv")
    # data_frame = data_frame.drop('Year', axis=1)
    # data_frame = data_frame.drop('Region/Country/Area', axis=1)
    # data_frame.fillna(0, inplace=True)
    data_frame = data

    data_frame.fillna(0, inplace=True)
    if year is not 'None':
        data_frame = data_frame.loc[data_frame['Year'] == year]
    # print(data_frame.shape)
    
    cols = ["GDP per capita (US dollars)","GDP in current prices (millions of US dollars)",'Population mid-year estimates (millions)', "Students enrolled in primary education (thousands)", "Students enrolled in secondary education (thousands)",'Students enrolled in tertiary education (thousands)']
    data_frame[cols] = data_frame[cols].replace({',': ''}, regex=True)
    data_frame[cols] = data_frame[cols].astype(float)

    country = data_frame.loc[:,"Region/Country/Area"].to_numpy()
    gdp_per_capita = data_frame.loc[:,"GDP per capita (US dollars)"].values
    gdp_curr = data_frame.loc[:,"GDP in current prices (millions of US dollars)"].values
    percent_exp = data_frame.loc[:,"Public expenditure on education (% of GDP)"].values
    
    train_df = data_frame.loc[:,'Population mid-year estimates (millions)':"Students enrolled in tertiary education (thousands)"]
    # print(train_df.head())
    train_df['% primary'] = (train_df.iloc[:,1]/train_df.iloc[:,0])*0.001
    train_df['% secondary'] = (train_df.iloc[:,2]/train_df.iloc[:,0])*0.001
    train_df['% tertiary'] = (train_df.iloc[:,3]/train_df.iloc[:,0])*0.001
    # print(train_df.head())
    features=train_df.iloc[:, 4:7].values
    return features,country,gdp_per_capita,gdp_curr,percent_exp
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

def test(feat, country ,gdp_per_capita, gdp_curr, percent_exp, model):
    """
    Returns model's test metrics.

    dataset: Numpy array of N samples of d features. [N,d]
    model: Trained Tensorflow Probability model
    """
    #TODO: Compute p-value and clustering graphs.
    MAP = tf.argmax(log_post(feat, model),1)
    
    # Plotting
    cluster_assign = np.random.randint(0,2,size=gdp_per_capita.size)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(gdp_per_capita,gdp_curr,percent_exp, marker="s", c=cluster_assign, s=50, cmap="viridis")
    # ax.legend()
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    plt.xlabel('GDP_per_Capita (in Dollars)')
    plt.ylabel('GDP_curr (in Millions)')
    plt.title("Percent Expenditure on Education vs GDP")
    # Change the next line to plt.show() if you want the plot to show up
    plt.savefig("3dplot_with_legend_and_labels_viridis.png")#bbox_inches='tight'

    return MAP  

def main():
    feat, country, gdp_per_capita, gdp_curr, percent_exp = load_dataset(2010)
    # Changed dataset to feat (training feratures, %prim,secon,tert enrollment)
    train_model = train(feat)
    MAP = test(feat, country, gdp_per_capita, gdp_curr, percent_exp, train_model) 

if __name__ == "__main__":
    main()