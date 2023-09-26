# TP2 Apprentissage Statistique - Arbres

## Auteur : Sarah Matoub.

contact : sarah.matoub@etu.umontpellier.fr

Ce dépot git contient les fichiers suivants :

- Les fichiers `tp_arbre_script.py` et `tp_arbre_source.py` contenant respectivement les codes **Python** de ce TP ainsi que les codes source des fonctions utilisées.

- Le fichier `.gitignore` afin de garder ce dépôt propre.

- Le fichier `tp2_sarahmatoub.qmd` contenant la rédaction du TP qui sera compilée en un fichier `tp2_sarahmatoub.html`.

- Un fichier `requirements.txt` contenant les librairies nécessaires pour ce TP.

- Le dossier `Plots` contenant la figure obtenue à la question 4 de ce TP.

## Lancement de la mission

Ce TP a été effectué sur **VsCode** , avant de pouvoir
expérimenter mon code, vous devez d'abord vous assurer d'avoir un
environnement de travail fonctionnel (si cela est déja fait vous pouvez passer
à la deuxième étape).

### Première étape :

- Installer `VsCode` en suivant les instructions sur [ce lien](https://code.visualstudio.com/download).

- Installer l'extension `Python` en suivant les instructions sur [ce lien](https://www.pythontutorial.net/getting-started/setup-visual-studio-code-for-python/).

- Enfin, installez `Quarto` pour pouvoir compiler le fichier `.qmd` en suivant les instructions sur [ce lien](https://quarto.org/docs/get-started/).

### Deuxième étape :

- Ouvrez un terminal et clônez le référentiel via la commande suivante :

```bash
$ git clone https://github.com/sarahmatoub/TP-apprentissage-stat.git
```

- Créez un nouvel environnement de travail nommé "tp_env":

```bash
$ conda create -n tp_env python=3.9.12
```

- Après avoir basculé dans votre nouvel environnement python, téléchargez les
  modules présents dans le fichier requirements.txt via la commande `pip`
  suivante:

```bash
$ pip install -r requirements.txt 
``` 

- Compilez le fichier `tp2_sarahmatoub.qmd` à l'aide de la commande :

```bash
$ quarto render tp2_sarahmatoub.qmd --to html
```
