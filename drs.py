import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from flask import Flask,render_template,jsonify,request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

"""run in cmd"""
# $env:FLASK_APP = "drs.py"
# flask run --host=192.168.0.114

####### dataset #######
food_nutrition = pd.read_csv("dataset/food_nutrition.csv")
disease_nutrition = pd.read_csv("dataset/disease_nutrition.csv",encoding='unicode_escape')

"""**Part-1**"""
####### Methods #######
def get_disease(disease_name):
	if(disease_name not in list(disease_nutrition["disease"])):
		return False
	else:
		return disease_nutrition[disease_nutrition.disease==disease_name]["disease"].values[0]

def get_disease_id(disease):
	return disease_nutrition[disease_nutrition.disease == disease]["disease_id"].values[0]

def get_disease_ie(disease):
	return disease_nutrition[disease_nutrition.disease == disease]["ineficient_nutritions"].values[0]


@app.route("/food",methods=["POST"])
####### get-set Data #######
def predict():
    global dis_list
    input_disease = request.form['disease']
    disease_name = get_disease(input_disease)
    print(disease_name)
    if(disease_name==False):
      print("Disease Not Found")
    else:
      disease_id = get_disease_id(disease_name)
      i= disease_id-101
      fd= disease_nutrition.iloc[i]
      pre_list = [fd["Precaution_1"],fd["Precaution_2"],fd["Precaution_3"],fd["Precaution_4"]]
      dict={"disease_precautions":pre_list,"disease_name":input_disease}
      disease_ie = get_disease_ie(disease_name)
      dis_list = list(disease_ie.split(" "))
      for ele in dis_list:
        if(ele==""):
          dis_list.remove(ele)
      prediction = displayFood()
      abc = {"prediction": prediction,"precaution": dict}
      return abc
      

"""**Preprocessing data**"""
def displayFood():
    columns_to_cluster = dis_list
    #MinMaxScaler
    #Transform features by scaling each feature to a given range.Here is an example to scale a data matrix to the [0, 1] range:
    mms = MinMaxScaler()
    food_scaled = mms.fit_transform(food_nutrition[columns_to_cluster])
    print("Scaled Food Value: ", food_scaled[0,:])
    columns_to_cluster_scaled = dis_list
    df_food_scaled = pd.DataFrame(food_scaled, columns=columns_to_cluster_scaled)

    """**Training the model**"""
    n_clusters = range(2,11)
    ssd = []
    sc = []
    for n in n_clusters:
        km = KMeans(n_clusters=n, max_iter=300, n_init=10, init='k-means++', random_state=42)
        km.fit(food_scaled)
        preds = km.predict(food_scaled) 
        centers = km.cluster_centers_ 
        ssd.append(km.inertia_) 
        score = silhouette_score(food_scaled, preds, metric='euclidean')
        sc.append(score)
        print("Number of Clusters = {}, Silhouette Score = {}".format(n, score))

    k=6
    model = KMeans(n_clusters=k, random_state=42).fit(food_scaled)
    pred = model.predict(food_scaled)
    print('10 first clusters: ', model.labels_[:10])

    """**Visualizing the clusters**"""

    df_food_scaled['cluster'] = model.labels_
    df_food_scaled['cluster'].value_counts()
    minor_cluster = df_food_scaled['cluster'].value_counts().tail(1)
    print("Amount of food in the smallest cluster: ", int(minor_cluster.values))

    df_food_joined = pd.concat([food_nutrition,df_food_scaled], axis=1).set_index('cluster')
    for cluster in range(k):
       df_food_joined.loc[cluster, ['Description']].sample(frac=1).head(10)
    """**Applying PCA to visualize the clusters**"""

    pca = PCA(n_components=3, random_state=42)
    food_pca = pca.fit_transform(food_scaled)
    pca.explained_variance_ratio_.sum()

    df_pca = pd.DataFrame(food_pca, columns=['C1', 'C2', 'C3'])
    df_pca['cluster'] = model.labels_
    df_pca.head()

    sampled_clusters_pca = pd.DataFrame()

    for c in df_pca.cluster.unique():
        df_cluster_sampled_pca = df_pca[df_pca.cluster == c].sample(n=int(minor_cluster), random_state=42)
        sampled_clusters_pca = pd.concat([sampled_clusters_pca,df_cluster_sampled_pca], axis=0)
    sampled_clusters_pca.cluster.value_counts()

    """**Predicting users clusters**"""

    user_pred = model.predict(food_scaled)
    print('10 first users clusters: ', user_pred[:10])
    user_cluster = pd.DataFrame(food_scaled, columns=columns_to_cluster_scaled)
    user_cluster['cluster'] = user_pred
    user_cluster['cluster'].value_counts()
    df_user_food_joined = pd.concat([food_nutrition,user_cluster], axis=1).set_index('cluster')
    for cluster in user_cluster['cluster'].unique():
        df_user_food_joined.loc[cluster, ['Description']].sample(frac=1).head(10)

    """**Recommending Food**"""

    df_user_food_joined.reset_index(inplace=True)
    cluster_pct = df_user_food_joined.cluster.value_counts(normalize=True)*20
    if int(cluster_pct.round(0).sum()) < 20:
        cluster_pct[cluster_pct < 0.5] = cluster_pct[cluster_pct < 0.5] + 1.0
    print('Total food: ', int(cluster_pct.round(0).sum()))
    df_food_joined.reset_index(inplace=True)
    df_user_food_joined['cluster_pct'] = df_user_food_joined['cluster'].apply(lambda c: cluster_pct[c])
    df_user_food_joined.drop(columns=columns_to_cluster_scaled, inplace=True)
    final_Food = pd.DataFrame()

    for ncluster, pct in cluster_pct.items():
        foods = df_food_joined[df_food_joined['cluster'] == ncluster].sample(n=int(round(pct, 0)))
        final_Food = pd.concat([final_Food,foods], ignore_index=True)
        if len(final_Food) > 20 :
            flag = 20 - len(final_Food)
            final_Food = final_Food[:flag]

    list_of_foods = final_Food['Description'].to_list()
    prediction = {'food_list': list_of_foods}
    return prediction