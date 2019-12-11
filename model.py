import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 
import numpy as np
import scipy.stats as stats
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def load_dataset(year='None'):
    #TODO:Return a N x d dataset of features to learn from and other features for plotting as well. 
    # If year is None, then return all the data; else only return data corresponding to that year
    data = pd.read_csv("alldata.csv")
    data_frame = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

    data_frame.fillna(0, inplace=True)
    if year is not 'None':
        data_frame = data_frame.loc[data_frame['Year'] == year]
    
    cols = ["GDP per capita (US dollars)","GDP in current prices (millions of US dollars)",'Population mid-year estimates (millions)', "Students enrolled in primary education (thousands)", "Students enrolled in secondary education (thousands)",'Students enrolled in tertiary education (thousands)']
    data_frame[cols] = data_frame[cols].replace({',': ''}, regex=True)
    data_frame[cols] = data_frame[cols].astype(float)

    country = data_frame.loc[:,"Region/Country/Area"].to_numpy()
    gdp_per_capita = data_frame.loc[:,"GDP per capita (US dollars)"].values
    gdp_curr = data_frame.loc[:,"GDP in current prices (millions of US dollars)"].values
    percent_exp = data_frame.loc[:,"Public expenditure on education (% of GDP)"].values
    
    train_df = data_frame.loc[:,'Population mid-year estimates (millions)':"Students enrolled in tertiary education (thousands)"]
    train_df['% primary'] = (train_df.iloc[:,1]/train_df.iloc[:,0])*0.001
    train_df['% secondary'] = (train_df.iloc[:,2]/train_df.iloc[:,0])*0.001
    train_df['% tertiary'] = (train_df.iloc[:,3]/train_df.iloc[:,0])*0.001
    features=train_df.iloc[:, 4:7].values
    return features,country,gdp_per_capita,gdp_curr,percent_exp
    # return tfd.Beta([.5]*3,[.5]*3).sample(10)

def train(dataset, k=3, epochs=200000, learn_prior=True, print_step=10):
    """
    Returns trained clustering model.
    
    dataset: Numpy array of N samples of d features. [N,d]
    k: Number of components/clusters.
    epochs: Number of epochs to train for.
    learn_prior: Make prior a learnable.
    print_step: Printing period of loss.
    """
    
    d = tf.shape(dataset)[-1] # Number of Features.

    w = tf.Variable(tf.abs(tf.random.normal([k,2*d]))) # [k,2d]
    pis = tf.Variable(tf.abs(tf.random.normal([k])), trainable=learn_prior) # [k]
   

    print('Dataset:{}'.format(dataset))

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

    return log_cat - log_dem + log_lik

def ptest(x, k, MAP, limits):
    """
    Computes p-values for each cluster and each difference between limits.
    
    Args:
    x: Values
    k: Number of classes.
    MAP: Cluster assignment for values.
    limits: List of thresholds.
    """
    assert k==len(limits)-1
    p_val = np.zeros([k,len(limits)-1])    
    for k in range(k):
        for i in range(len(limits)-1):
            total_class = np.sum(tf.logical_and(limits[i]<=x,x<=limits[i+1]))
            total_cluster = np.sum(tf.logical_and(limits[i]<=x[MAP==k],x[MAP==k]<=limits[i+1]))
            rv = stats.hypergeom(len(MAP),total_class,np.sum(MAP==k))
            p_val[k,i] = 1-rv.cdf(total_cluster)

    return p_val

def test(feat, country ,gdp_per_capita, gdp_curr, percent_exp, model):
    """
    Returns model's test metrics.

    dataset: Numpy array of N samples of d features. [N,d]
    model: Trained Tensorflow Probability model
    """
    #TODO: Compute p-value and clustering graphs.
    MAP = tf.argmax(log_post(feat, model),1)
    
    # Plotting
    # cluster_assign = np.random.randint(0,2,size=gdp_per_capita.size)
    cluster_assign = MAP
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    #scatter = ax.scatter(gdp_per_capita,gdp_curr,percent_exp, marker="s", c=cluster_assign, s=50, cmap="viridis")
    ax.scatter(feat[:,0],feat[:,1],feat[:,2], marker="s", c=cluster_assign, s=50, cmap="viridis")
    # scatter = ax.scatter(gdp_per_capita, percent_exp, marker="s", c=cluster_assign, s=50, cmap="viridis", label=str(cluster_assign))
    # ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.set_xlabel('Primary')
    ax.set_ylabel('Secondary')
    ax.set_zlabel('Tertiary')
    ax.title.set_text('Enrolled Population Percentage')

    ax2 = fig.add_subplot(122)
    ax2.scatter(gdp_per_capita, percent_exp, marker="s", c=cluster_assign, s=50, cmap="viridis", label=str(cluster_assign))
    ax2.set_xlabel('GDP_per_Capita (in Dollars)')
    ax2.set_ylabel('Percent Expenditure on Education')
    ax2.title.set_text("Percent Expenditure on Education vs GDP")
    # Change the next line to plt.show() if you want the plot to show up
    plt.savefig("enrrolled_pop_percent.png")#bbox_inches='tight'
    
    return MAP, plt  

def main():
    feat, country, gdp_per_capita, gdp_curr, percent_exp = load_dataset(2010)
    # Changed dataset to feat (training feratures, %prim,secon,tert enrollment)
    train_model = train(feat)
    MAP, plt = test(feat, country, gdp_per_capita, gdp_curr, percent_exp, train_model) 
    print(ptest(gdp_per_capita,3,MAP, [0,3500,45000,100000000000]))
    print(ptest(percent_exp,3,MAP, [0,3,5.40,10]))
    plt.show()
if __name__ == "__main__":
    main()
