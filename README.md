# Hybrid-ByteTrack — Suivi Multi-Objets pour Vidéoprotection Intelligente

Projet Intégrateur 1 — PIIA Cohorte 2, Académie des Mathématiques Appliquées (AMA)

**Groupe 7** : Amen QUENUM · Ablo Romain GANDONOU · Emiline ADANGNISSODE · Wilfried BONOU

---

## 1. Contexte et objectif

Dans un trafic dense (motos et voitures qui se croisent et se masquent mutuellement),
ByteTrack perd fréquemment l'identité d'un véhicule lors d'une occlusion prolongée
(ID Switch), car il ne s'appuie que sur la trajectoire géométrique.

**Hybrid-ByteTrack** ajoute à ByteTrack un module léger de vérification par
apparence (couleur) au moment où un objet réapparaît après disparition, pour
réduire ces erreurs sans sacrifier la vitesse temps réel.

## 2. Architecture du pipeline
Détection : YOLOv8n fine-tuné (classes car / motorcycle)
Tracking : ByteTrack (baseline) → génère les trajectoires brutes
Correction : notre_module_hybride.py → détecte et fusionne les ID Switches
probables par similarité d'histogramme couleur (HSV)
Évaluation : motmetrics → calcule MOTA / IDF1 / ID Switches
Comparaison : baseline vs hybride, sur vérité terrain MOTChallenge

## 3. Structure du dépôt

```
.
├── Projet_Suivi_ByteTrack.md
├── README.md
├── requirements.txt
├── results
│   ├── check_bdd_0090c713-9d58a186_frame50.jpg
│   ├── check_bdd_00ac3256-0f8e2cda_frame50.jpg
│   ├── check_bdd_012e9465-1031243b_frame50.jpg
│   ├── comparison_mota_idf1.csv
│   └── tracking
│       ├── baseline
│       ├── hybride
│       └── notre_hybride
└── scripts
    ├── build_yolo_dataset.py
    ├── demo_video.py
    ├── eval_mot.py
    ├── notre_module_hybride.py
    ├── prepare_bdd_sequences.py
    ├── run_tracking.py
    ├── train_yolo.py
    └── verify_bdd_gt.py
```

**Note** : `data/` et `runs/` (poids entraînés, datasets bruts) ne sont pas
versionnés sur GitHub (~21 Go) — voir section 5 pour les régénérer.

## 4. Installation

```bash
git clone <URL_DU_DEPOT>
cd Suivi_ByteTrack
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 5. Reproduire le pipeline complet

Les données brutes (~21 Go) ne sont pas versionnées sur GitHub. Pour reproduire :

```bash
# 1. Construire le dataset d'entraînement (télécharge BDD100K via Kaggle, nécessite un compte Kaggle configuré)
python3 scripts/build_yolo_dataset.py

# 2. Fine-tuner le détecteur (long sur CPU — plusieurs heures)
python3 scripts/train_yolo.py

# 3. Préparer les séquences d'évaluation avec vérité terrain
python3 scripts/prepare_bdd_sequences.py
python3 scripts/verify_bdd_gt.py   # vérification visuelle, optionnel mais recommandé

# 4. Lancer le tracking (baseline ByteTrack + référence BoT-SORT)
python3 scripts/run_tracking.py

# 5. Appliquer notre module hybride
python3 scripts/notre_module_hybride.py

# 6. Évaluer et comparer
python3 scripts/eval_mot.py
```

## 6. Résultats (BDD100K, 3 séquences, ~600 frames)

| Méthode | MOTA | IDF1 | ID Switches |
|---|---|---|---|
| ByteTrack (baseline) | 0.152 | 0.282 | 36 |
| BoT-SORT (référence externe) | 0.151 | 0.281 | 37 |
| **Notre module hybride** | **0.153** | 0.263 | **32 (-11%)** |

Notre module réduit les ID Switches de 11 % avec un MOTA légèrement amélioré,
mais au coût d'un IDF1 plus faible : certaines fusions par couleur associent
parfois deux véhicules distincts de teinte proche (ex. deux voitures blanches).
Ce compromis sera affiné dans les prochaines itérations (ajout d'un critère de
forme, seuil de similarité adaptatif).

## 7. Limites actuelles et travail en cours

- Résultats mesurés sur BDD100K (caméra embarquée), qui contient peu d'occlusions
  prolongées comparé au trafic dense visé par le projet (marchés, carrefours béninois).
- Un dataset filmé au Bénin (embouteillages réels) est en cours d'annotation
  manuelle (CVAT) pour validation sur le scénario cible exact.
- Le détecteur est entraîné uniquement sur données occidentales — sa généralisation
  aux scènes ouest-africaines (motos "zémidjans", densité, luminosité) reste à
  confirmer quantitativement (démonstration qualitative disponible).

## 8. Prochaines étapes

- [ ] Finaliser l'annotation du dataset béninois (CVAT)
- [ ] Ajouter un critère de forme/aspect-ratio au module hybride
- [ ] Remesurer MOTA/IDF1 sur données béninoises réelles
- [ ] Optimiser le seuil de similarité couleur (réduire les faux positifs de fusion)
