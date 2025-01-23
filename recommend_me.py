#manage dependencies

import pandas as pd
import numpy as np

from torch import torchvision, torchaudio


#Crie um sistema de recomendação de receitas considerando que um dado cliente irá informar 5 tags. 
# O sistema deve conter um score das melhores receitas para cada cliente. 
# E, caso as tags mudem, as receitas recomendadas e scores também devem mudar. 
# Além disso, na análise apresente resultados para as top 5 receitas para 3 clientes diferentes. Interprete os resultados.

### Dataset
###########################333
#load dataset
name = 'receitas.json'
original_dataset = pd.read_json(name)

#excluir nulls e duplicadas

original_dataset.dropna(how='all', inplace=True)
original_dataset.drop_duplicates(subset=['title', 'date'], inplace=True)
df = original_dataset.dropna(subset=['categories'])

#create a taglist
taglist = set(color for sublist in df['categories'] for color in sublist)
taglist = list(taglist)

###importar modelo pré-treinado
##################################
import gensim.downloader as api

#modelo pre treinado
model = api.load('word2vec-google-news-300')

from sentence_transformers import SentenceTransformer

st_model = SentenceTransformer('all-MiniLM-L6-v2')

def return_taglist():
    return taglist

#### Funções
################################################################33

#pt1: embeddings
#gerar clusters probabilisticamente pq palavras são ambíguas e etc
#gerar embeddings das tags
def embed_tags(tags):

  tag_embeddings = st_model.encode(tags)
  # print(tag_embeddings[0])
  return tag_embeddings

#clusterizar as tags usando um critério de informação (mistura gaussiana e bic scores)
def cluster_tags(tag_embeddings, min_clusters, max_clusters):
  from sklearn.mixture import GaussianMixture

  # Example: tag_embeddings.shape -> (num_tags, embedding_size)
  tag_embeddings = np.array(tag_embeddings)

  # Fit GMM with automatic selection of number of clusters using BIC
  bic_scores = []
  models = []

  for n in range(min_clusters, max_clusters):  # Search for optimal clusters in a range
      gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
      gmm.fit(tag_embeddings)
      bic_scores.append(gmm.bic(tag_embeddings))
      models.append(gmm)
      # print(bic_scores)

  #lowest BIC
  best_model = models[np.argmin(bic_scores)]

  # cluster labels
  labels = best_model.predict(tag_embeddings)

  # print(labels)
  return labels

####################################################333
#pt2: matching
# map words to clusters
def map_clusters(words, labels):
    clusters = {}
    distincts = list(set(labels))
    for value in distincts:
      cluster_words = [words[i] for i, v in enumerate(labels) if v == value]
      clusters[value] = cluster_words

    return clusters

def map_instance_cluster(instance, words, labels, islist=False):
  try:
    if islist:
      i = [words.index(x) for x in instance]
      return [labels[j] for j in i]
    else:
        i = words.index(instance)
        return labels[i]
  except Exception as  e:
    print(e)
    return None

#gerar clusters e métricas de similaridade entre eles

def generate_readable_clusters(example_tag, labels):
  return  map_clusters(example_tag, labels)

from sklearn.metrics.pairwise import cosine_similarity

### métricas de similaridade
def cluster_similarity(cluster1, cluster2):
    # Get embeddings for each word in the clusters
        # tag_embeddings = st_model.encode(taglist)

    embeddings1 = st_model.encode(cluster1)
    # embeddings2 = [model[tag] for tag in cluster2]
    embeddings2 = st_model.encode(cluster2)

    # Calculate pairwise cosine similarities and take the average
    if not embeddings1.size == 0:
      if not embeddings2.size ==0: # Ensure there are embeddings
        similarities = cosine_similarity(embeddings1, embeddings2)
        avg_similarity = np.mean(similarities)
        return avg_similarity
    else:
        return 0  # Return zero if no words have embeddings

### métricas de similaridade entre clusters
def pairwise_cluster_similarity(c):
#c for clusters_dict generated in generate_readable_clusters
  pcs = dict()
  for key in c:
    for unmatched_key in c:
      if not key == unmatched_key:
        similarity = cluster_similarity(c[key], c[unmatched_key])
        # print(key, unmatched_key, similarity)
        pcs[(key, unmatched_key)] = similarity
      else:
        pcs[(key, key)] = 1

  return pcs

#criar score para matching user x instance

def get_match_score(instance_tags, input_tags, weights, taglist, labels, pcs):
  try:
    #let pcs be "global"
      matches = 0
      total_weight = sum(weights)
      # print(instance_tags, input_tags, weights)

      # matches e calculo de scores por média dos pesos
      #correspondencias diretas valem mais, até 100%
      #na análise de cluster, vale a análise de similaridade e a quantidade, tipo [similaridade * peso * quantidade / (soma dos pesos + total de elementos)] para cada input tag
      for idx, tag in enumerate(input_tags):
          # print(tag, cluster_tag)
          if isinstance(tag, str):
            if tag in instance_tags:
              matches += weights[idx]
              # print(matches)
          else:
              p =set(instance_tags)
              count = [instance_tags.count(x) for x in p]
              similarity_score = list()
              similarity_score = [pcs[(tag,y)] for y in p]
              dot = np.dot(similarity_score, count)
              matches = matches + ((weights[idx] * dot) / len(instance_tags))

      return (matches / total_weight) * 100

  except Exception as e:
      print(e)
      pass
  
########################################################
### pt3: recomendação

def graded_instances(dataset, input_tags, weights, taglist, labels, tag_column, cluster_col, pcs):
    cluster_input_tags = map_instance_cluster(input_tags, taglist, labels, islist=True)
    # Computar match spara instances
    for index, row in dataset.iterrows():
        instance, clustered_instance = row[tag_column], row[cluster_col]
        # print(instance, clustered_instance)
        exact_match = get_match_score(instance, input_tags, weights, taglist, labels, pcs)
        similarity_match = get_match_score(clustered_instance, cluster_input_tags, weights, taglist, labels, pcs)
        dataset.at[index, 'score'] = exact_match + similarity_match
        # print(exact_match + similarity_match)

    return dataset

#pipeline de execução

def transform(dataset, target_column, treat=False, verbose=False):
  try:
    #nome do dataset
    #gerar tags do dataset original
    #modelo é tratado como uma variável global daqui
    # st_model = SentenceTransformer('all-MiniLM-L6-v2')
    if treat:
        dataset = pd.read_json(dataset)

        #excluir nulls e duplicadas
        dataset.dropna(how='all', inplace=True)
        dataset.drop_duplicates(subset=['title', 'date'], inplace=True)
        df = dataset.dropna(subset=['categories'])
        # dataset['categories'] = dataset['categories'].apply(eval)

        #create a taglist
        taglist = set(color for sublist in df['categories'] for color in sublist)
        taglist = list(taglist)

    tag_embeddings = st_model.encode(taglist)

    #labels
    labels = cluster_tags(tag_embeddings,  min_clusters=10, max_clusters=20)
    c = generate_readable_clusters(taglist, labels)
    if verbose:
        for key in c:
            print(key, c[key], '\n') #veja meus clusters

    #score de similaridade
    pcs = pairwise_cluster_similarity(c)
    #target_colum, para o dataset original, é 'categories'
    dataset['clustered_tags'] = dataset[target_column].apply(lambda x: map_instance_cluster(x, taglist, labels, islist=True))

    return dataset, pcs, labels, c

  except Exception as e:
    print(e)
    # raise
    return None

def recommend_me(user):
   try:
    dataset, pcs, labels, c = transform(df)
    return graded_instances(dataset, user['tags'], user['weights'], taglist, labels, 'categories','clustered_tags', pcs),c
   
   except Exception as e:
    print(e)
    return None
