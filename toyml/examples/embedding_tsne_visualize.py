import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

item_id = 'item_id'
tag = 'tag'
item_embed = 'item_embed'
df = pd.read_csv('embedding.txt',
                 sep='\u0001',
                 names=[item_id, tag, item_embed],
                 dtype={item_id: str, tag: str, item_embed: str})

id_data = df[item_id].values
embed_data = df[item_embed].values
y = df[tag].values

X = []
for embed in embed_data:
    vec = []
    for v in embed.split('\u0002'):
        if len(v) > 0:
            vec.append(float(v))
    X.append(vec)

# t-sne visualize high-dimensional embedding
X_tsne = TSNE(n_components=2,
              random_state=33,
              n_jobs=-1).fit_transform(X)
df = pd.DataFrame()
df['comp-1'] = X_tsne[:, 0]
df['comp-2'] = X_tsne[:, 1]
df['y'] = y

sns.scatterplot(x="comp-1",
                y="comp-2",
                hue=df.y.tolist(),
                palette=sns.color_palette("hls", len(pd.unique(df['y']))),
                data=df).set(title="embedding T-SNE projection")

plt.savefig("visualize.png")
