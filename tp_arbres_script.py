#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from matplotlib import rc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree, datasets
from tp_arbres_source import (rand_gauss, rand_bi_gauss, rand_tri_gauss,
                              rand_checkers, rand_clown,
                              plot_2d, frontiere)
#%%
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
params = {'axes.labelsize': 6,
          'font.size': 12,
          'legend.fontsize': 12,
          'text.usetex': False,
          'figure.figsize': (10, 12)}
plt.rcParams.update(params)

sns.set_context("poster")
sns.set_palette("colorblind")
sns.set_style("white")
_ = sns.axes_style()

############################################################################
# Data Generation: example
############################################################################

np.random.seed(1)

n = 100
mu = [1., 1.]
sigma = [1., 1.]
rand_gauss(n, mu, sigma)


n1 = 20
n2 = 20
mu1 = [1., 1.]
mu2 = [-1., -1.]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
data1 = rand_bi_gauss(n1, n2, mu1, mu2, sigma1, sigma2)

n1 = 50
n2 = 50
n3 = 50
mu1 = [1., 1.]
mu2 = [-1., -1.]
mu3 = [1., -1.]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
sigma3 = [0.9, 0.9]
data2 = rand_tri_gauss(n1, n2, n3, mu1, mu2, mu3, sigma1, sigma2, sigma3)

n1 = 50
n2 = 50
sigma1 = 1.
sigma2 = 5.
data3 = rand_clown(n1, n2, sigma1, sigma2)


n1 = 114 
n2 = 114
n3 = 114
n4 = 114
sigma = 0.1
data4 = rand_checkers(n1, n2, n3, n4, sigma)

plot_2d(data4)
#%%
############################################################################
# Displaying labeled data
############################################################################

plt.close("all")
plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(141)
plt.title('First data set')
plot_2d(data1[:, :2], data1[:, 2], w=None)

plt.subplot(142)
plt.title('Second data set')
plot_2d(data2[:, :2], data2[:, 2], w=None)

plt.subplot(143)
plt.title('Third data set')
plot_2d(data3[:, :2], data3[:, 2], w=None)

plt.subplot(144)
plt.title('Fourth data set')
plot_2d(data4[:, :2], data4[:, 2], w=None)

#%%
############################################
# ARBRES
############################################


# Q2. Créer deux objets 'arbre de décision' en spécifiant le critère de
# classification comme l'indice de gini ou l'entropie, avec la
# fonction 'DecisionTreeClassifier' du module 'tree'.

dt_entropy = tree.DecisionTreeClassifier(criterion="gini")
dt_gini = tree.DecisionTreeClassifier(criterion="entropy")

# Effectuer la classification d'un jeu de données simulées avec rand_checkers des échantillons de
# taille n = 456 (attention à bien équilibrer les classes)

data = rand_checkers(n1=114, n2=114, n3=114, n4=114, sigma=0.1)
n_samples = len(data)
X = data[:, :2] 
Y = data[:,2].astype(int)  #and be careful with the type (cast to int)

dt_gini.fit(X, Y)
dt_entropy.fit(X, Y) 

#%%

print("Gini criterion") #calcule les performances du score sur les données d'apprentissage
print(dt_gini.get_params())
print(dt_gini.score(X, Y))

print("Entropy criterion")
print(dt_entropy.get_params())
print(dt_entropy.score(X, Y))


#%%
# Afficher les scores en fonction du paramètre max_depth
# Créez un ensemble de test
#X_test, Y_test, X_train, Y_train = train_test_split(X, Y, test_size=0.2, random_state=42)

dmax = 12
scores_entropy = np.zeros(dmax)
scores_gini = np.zeros(dmax)

best_depth_entropy = None
best_score_entropy = 0.0

best_depth_gini = None
best_score_gini = 0.0


plt.figure(figsize=(15, 10))
for i in range(dmax):
    # Arbre de décision avec le critère entropy
    dt_entropy = tree.DecisionTreeClassifier(criterion="entropy", max_depth = i + 1, random_state = 0)
    dt_entropy.fit(X, Y)
    Y_pred_entropy = dt_entropy.predict(X)
    scores_entropy[i] = accuracy_score(Y, Y_pred_entropy)
    if scores_entropy[i] > best_score_entropy:
        best_score_entropy = scores_entropy[i]
        best_depth_entropy = i + 1
        
    # Arbre de décision avec le critère gini
    dt_gini = tree.DecisionTreeClassifier(criterion="gini", max_depth = i + 1, random_state = 0)
    dt_gini.fit(X, Y)
    Y_pred_gini = dt_gini.predict(X)
    scores_gini[i] = accuracy_score(Y, Y_pred_gini)
    if scores_gini[i] > best_score_gini:
        best_score_gini = scores_gini[i]
    best_depth_gini = i + 1

    plt.subplot(3, 4, i + 1)
    frontiere(lambda x: dt_gini.predict(x.reshape((1, -1))), X, Y, step=50, samples=False)
    plt.title(f'Depth{ i + 1 }')
plt.draw()

#%%
plt.figure()
plt.plot(range(1, dmax + 1), 1 - scores_entropy, label="Accuracy score (Entropie)")
plt.plot(range(1, dmax + 1), 1 - scores_gini, label="Accuracy score (Gini)")
plt.legend()
plt.xlabel('Max depth')
plt.ylabel('Accuracy Score')
plt.title("Accuracy score as a function of max depth")
plt.draw()
print("Scores with entropy criterion: ", scores_entropy)
print("Scores with Gini criterion: ", scores_gini)

#%%
# Q3 Afficher la classification obtenue en utilisant la profondeur qui minimise le pourcentage d’erreurs obtenues avec l’entropie
random.seed(1234)
# Créer un arbre de décision avec la meilleure profondeur pour l'entropie
best_tree_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=best_depth_entropy, random_state=0)
best_tree_entropy.fit(X, Y)
# Afficher la classification obtenue avec la profondeur optimale (Entropy)
plt.figure(figsize=(8, 8))
plt.figure()
frontiere(lambda x: dt_entropy.predict(x.reshape((1, -1))), X, Y, step=100, samples=True) #l'évaluation en X de prédict, elle prend X pour le prédire
plt.title("Best frontier with entropy criterion")
plt.draw()
print("Best scores with entropy criterion: ", dt_entropy.score(X, Y))

#%%
# Q4.  Exporter la représentation graphique de l'arbre: Need graphviz installed
# Voir https://scikit-learn.org/stable/modules/tree.html#classification

from sklearn.tree import export_graphviz
import graphviz

best_tree_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=best_depth_entropy, random_state=0)
best_tree_entropy.fit(X, Y)
# On exporte l'arbre au format DOT
donnes = export_graphviz(best_tree_entropy)
# On crée un objet Graphviz à partir du fichier DOT
graph = graphviz.Source(donnes)
graph.render("best_tree_entropy", format = "pdf")

#%%
# Q5 :  Génération d'une base de test
# Créer un nouvel échantillon de test avec 160 données (40 de chaque classe)
new_data = rand_checkers(n1=40, n2=40, n3=40, n4=40, sigma=0.1)

# Séparer les caractéristiques et les étiquettes
X_new = new_data[:, :2]
Y_new = new_data[:, 2].astype(int)

# Évaluer les modèles sur le nouvel échantillon de test
error_rate_entropy = 1 - best_tree_entropy.score(X_new, Y_new)

best_tree_gini = DecisionTreeClassifier(criterion="gini", max_depth=best_depth_gini, random_state=0)
best_tree_gini.fit(X_new, Y_new)

error_rate_gini = 1 - best_tree_gini.score(X_new, Y_new)

print("Proportion d'erreurs sur le nouvel échantillon (Entropy): {:.2f}%".format(error_rate_entropy * 100))
print("Proportion d'erreurs sur le nouvel échantillon (Gini): {:.2f}%".format(error_rate_gini * 100))

dmax = 12
scores_entropy = np.zeros(dmax)
scores_gini = np.zeros(dmax)
plt.figure(figsize=(15, 10))

for i in range(dmax):
    
    # Créer un arbre de décision avec une profondeur maximale variable (entropy)
    dt_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=i + 1)
    dt_entropy.fit(X_new, Y_new)
    scores_entropy[i] = 1 - dt_entropy.score(X_new, Y_new)

    dt_gini = DecisionTreeClassifier(criterion="gini", max_depth = i + 1)
    dt_gini.fit(X_new, Y_new)
    scores_gini[i] = 1 - dt_gini.score(X_new, Y_new)
    
    plt.subplot(3, 4, i + 1)
    frontiere(lambda x: dt_gini.predict(x.reshape((1, -1))), X_new, Y_new, step=50, samples=False)

plt.figure()
plt.plot(range(1, dmax + 1), scores_entropy, label="Entropy")
plt.plot(range(1, dmax + 1), scores_gini, label="Gini")
plt.xlabel('Max depth')
plt.ylabel('Error Rate')
plt.title("Testing error")
plt.legend()
plt.show()

print("Proportion d'erreurs sur le nouvel échantillon (Entropy):", scores_entropy)
print("Proportion d'erreurs sur le nouvel échantillon (Gini):", scores_gini)

#%%
# Q6. même question avec les données de reconnaissances de texte 'digits'
from sklearn import datasets
from sklearn.model_selection import train_test_split
# Import the digits dataset
digits = datasets.load_digits()

#X = digits.data
#Y = digits.target 

n_samples = len(digits.data)

# Diviser l'ensemble de données en ensembles d'entraînement et de test (80% - 20%)
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

digits.images.reshape((n_samples, -1))
X_test = digits.data[:n_samples*8 // 10]  # digits.images.reshape((n_samples, -1))
Y_test = digits.target[:n_samples*8 // 10]
X_train = digits.data[n_samples*8 // 10:]
Y_train = digits.target[n_samples*8 // 10:]

dmax = 12
scores_entropy = np.zeros(dmax)
scores_gini = np.zeros(dmax)

for i in range(dmax):
    # Arbre de décision pour l'entropie
    dt_entropy = DecisionTreeClassifier(criterion="entropy", max_depth = i + 1)
    dt_entropy.fit(X_train, Y_train)
    scores_entropy = dt_entropy.score(X_train, Y_train)
    
    #Arbre de décision pour gini
    dt_gini =  DecisionTreeClassifier(criterion="gini", max_depth = i + 1)
    dt_gini.fit(X_train, Y_train)
    scores_entropy = dt_entropy.score(X_train, Y_train)
    
plt.figure()
plt.plot(scores_entropy, label = "score entropy")
plt.plot(scores_gini, label = "score gini")
plt.legend()
plt.xlabel("Max Depth")
plt.ylabel("Accuracy score")
plt.draw()

#%%
# Let's see what happens
dmax = 20
scores_entropy = np.zeros(dmax)
scores_gini = np.zeros(dmax)

for i in range(dmax):
    dt_entropy = tree.DecisionTreeClassifier(criterion="entropy", max_depth = i + 1)
    dt_entropy.fit(X_train,Y_train)
    scores_entropy[i] = dt_entropy.score(X_test, Y_test)

    dt_gini = tree.DecisionTreeClassifier(criterion="gini", max_depth = i+1)
    dt_gini.fit(X_train,Y_train)
    scores_gini[i] = dt_gini.score(X_test,Y_test)


plt.figure()
plt.plot(scores_entropy, label="entropy")
plt.plot(scores_gini, label="gini")
plt.legend()
plt.xlabel("Max Depth")
plt.ylabel("Accuracy Score")
plt.draw()
print("Entropy criterion scores : ", scores_entropy)
print("Gini criterion scores : ", scores_gini)

#%%
# Q7. estimer la meilleur profondeur avec un cross_val_score

from sklearn.model_selection import cross_val_score

# Profondeurs maximales à tester
max_depths = np.arange(1, 16, 1)

# scores de validation croisée
cv_scores = []

# Tester chaque profondeur maximale
for depth in max_depths:
    # Initialiser et entraîner l'arbre de décision avec le critère d'Entropy
    tree_classifier = DecisionTreeClassifier(criterion="entropy", max_depth=depth, random_state=0)
    
    # Effectuer une validation croisée avec 5 plis
    scores = cross_val_score(tree_classifier, X, Y, cv=5)
    
    # Calculer la moyenne des scores de validation croisée
    mean_score = scores.mean()
    
    # Ajouter le score moyen à la liste des scores
    cv_scores.append(mean_score)

# Trouver la meilleure profondeur maximale avec le score le plus élevé
best_depth_index = cv_scores.index(max(cv_scores))
best_depth = max_depths[best_depth_index]


for depth, score in zip(max_depths, cv_scores):
    print("Profondeur maximale = {}, Score de validation croisée moyen = {:.4f}".format(depth, score))
    
print("Meilleure profondeur maximale (Entropy) = {}, Score de validation croisée moyen = {:.4f}".format(best_depth, max(cv_scores)))

#%% 
# Q8 afficher la courbe d’apprentissage

