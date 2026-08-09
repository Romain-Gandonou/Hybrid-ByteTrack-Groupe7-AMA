
Projet : Amélioration du suivi multi-objets des motos dans des scènes routières urbaines denses
1. Contexte

Les systèmes de vision par ordinateur sont de plus en plus utilisés pour la surveillance intelligente du trafic routier. Le suivi multi-objets (Multi-Object Tracking - MOT) permet d'attribuer une identité unique aux véhicules et de suivre leur trajectoire au cours du temps dans une vidéo.

Les méthodes modernes comme ByteTrack obtiennent de très bonnes performances sur plusieurs benchmarks, mais certaines difficultés persistent dans des environnements routiers complexes.

2. Sujet

Amélioration du suivi des motos dans des scènes routières urbaines denses avec YOLO11 + ByteTrack.

3. Problématique

Dans les scènes de circulation dense, plusieurs motos peuvent :

circuler très proches les unes des autres ;
se croiser ;
être partiellement occultées par d'autres véhicules ;
disparaître temporairement puis réapparaître.

Ces situations peuvent provoquer des erreurs de suivi comme :

perte d'une trajectoire ;
changement d'identité d'un véhicule (ID Switch) ;
fragmentation des trajectoires.

La question étudiée est :

Est-ce qu'une adaptation du détecteur YOLO11 sur des données spécifiques de circulation dense permet d'améliorer les performances du suivi réalisé par ByteTrack ?

4. Hypothèse

Un modèle YOLO11 spécialisé sur des scènes contenant davantage de motos proches et partiellement occultées pourrait fournir de meilleures détections à ByteTrack et ainsi améliorer :

la conservation des identités ;
la continuité des trajectoires ;
les métriques de suivi.
5. Approche proposée

Pipeline initial :

Vidéo de circulation
        ↓
YOLO11 pré-entraîné
        ↓
Détection des motos/véhicules
        ↓
ByteTrack
        ↓
Suivi avec IDs

Pipeline amélioré :

Vidéo de circulation
        ↓
YOLO11 fine-tuné sur notre dataset
        ↓
Détection améliorée
        ↓
ByteTrack
        ↓
Suivi amélioré
6. Modèle de référence (Baseline)

Système initial :

YOLO11 pré-entraîné + ByteTrack

Ce système sera évalué avant toute modification.

7. Amélioration proposée

Fine-tuning de YOLO11 sur un dataset adapté contenant :

circulation urbaine dense ;
motos ;
véhicules proches ;
occultations.

Possibilité d'ajouter des données locales béninoises afin d'améliorer l'adaptation au contexte local.

8. Dataset envisagé

Sources possibles :

datasets publics de trafic routier ;
datasets spécialisés en suivi multi-objets ;
vidéos locales de circulation au Bénin.

Critères :

vidéos ;
annotations ;
présence d'identités pour le tracking ;
scènes denses ;
motos.
9. Métriques d'évaluation

Les performances seront mesurées avec :

MOTA (Multi-Object Tracking Accuracy)
IDF1 (qualité de conservation des identités)
HOTA (qualité globale du tracking)
ID Switch (nombre de changements d'identité)
FPS (vitesse de traitement)
10. Résultat attendu

Obtenir un système capable de mieux suivre les motos dans des situations difficiles par rapport au système de base.