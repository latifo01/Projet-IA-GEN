# Generative AI Preprocessing Agent

Un utilitaire CLI et Dashboard Web (via Streamlit) conçu pour nettoyer, transformer et préparer vos datasets avant modélisation de manière semi-automatisée, à l'aide de LangGraph et OpenAI.

L'objectif de cet outil n'est pas de jeter du LLM à l'aveugle sur des tableaux de données. Il orchestre plutôt un ensemble de **règles déterministes strictes** (pour les tâches évidentes comme la gestion des colonnes vides ou de l'imputation par la médiane) et s'appuie sur **gpt-5-mini** uniquement pour le contexte métier et les cas ambigus (anomalies sémantiques, cardinaux intermédiaires, détection de features).

Surtout : **l'humain garde le contrôle**. À chaque étape critique du pipeline, l'exécution s'interrompt pour soumettre les choix des agents à l'utilisateur.

---

## Fonctionnement du Pipeline

Le traitement est découpé entre 4 agents spécialisés. Le graphe d'état (LangGraph) garantit que les informations transitent correctement et que les interruptions humaines sont respectées.

1. **L'Analyseur (Agent 1)**
   Audit initial des types, suppression des colonnes inutiles (IDs, colonnes constantes ou full-NaN) et définition de la stratégie de split (stratifiée, temporelle, etc.) adaptée au contexte déduit.
   *`-> Pause : l'utilisateur valide le split et le domaine métier.`*

2. **Le Transformateur (Agent 2)**
   Applique l'encodage (One-Hot, Frequency) et l'imputation (Mode, Médiane). Le LLM intervient ici pour résoudre les cas limites, détecter d'éventuels leakages, et proposer du feature engineering pertinent.
   *`-> Pause : l'utilisateur valide les propositions de transformation.`*

3. **Le Chasseur d'Outliers (Agent 3)**
   Sélectionne dynamiquement la méthode statistique adaptée à la distribution de la colonne (IQR, Z-score, ou seuil bimodal selon le test de D'Agostino-Pearson) pour conserver l'information tout en isolant les anomalies.
   *`-> Pause : l'utilisateur vérifie la logique d'écrêtage (clipping/removal).`*

4. **Le Rapporteur (Agent 4)**
   Génère les CSV finaux (train et test), entraîne une baseline rapide (Régression logistique vs Dummy), vérifie les contraintes, construit le dashboard Streamlit et calcule un **score de qualité sur 100** des données fraîchement nettoyées.

> 🛡️ **Garantie anti-leakage** : L'outil est designé pour splitter les données *avant* le moindre calcul d'imputation ou d'encoding. Toutes les transformations (médianes, OHE, boundaries d'outliers) sont fittées exclusivement sur le split de _train_ puis appliquées aveuglément sur le _test_.

---

## 🛠️ Installation

Le projet utilise `uv` comme gestionnaire de dépendances. Assurez-vous d'avoir Python 3.12+ d'installé.

```bash
# Installer les dépendances cœur (si vous voulez juste le CLI)
uv sync

# Installer les dépendances de développement, les notebooks et le tracking (expériences)
uv sync --extra dev --extra tracking

# Créer votre fichier de configuration d'environnement
cp .env.example .env
```

Vous devrez configurer vos clés API dans le fichier `.env` nouvellement créé, en copiant ce format :
```env
# LLM Provider Configuration
# Fill in your credentials below

# OpenAI 
OPENAI_API_KEY='entrez votre api openai'
# LangSmith (LangChain)
LANGCHAIN_API_KEY='entrez votre api langchain'
LANGCHAIN_PROJECT=projet Gen AI
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
WANDB_API_KEY='entrez votre api wandb'

# Google AI (Gemini) - for future use
# GOOGLE_API_KEY=your-google-key-here

# Anthropic (Claude) - for future use
# ANTHROPIC_API_KEY=your-anthropic-key-here

# Ollama (local) - for future use
# OLLAMA_BASE_URL=http://localhost:11434

# Default LLM provider: openai | google | anthropic | ollama
LLM_PROVIDER=openai
LLM_MODEL=gpt-5-mini
```

---

## 🚀 Utilisation

### Via l'interface Web (Recommandé)

C'est l'interface la plus agréable pour interagir avec les interruptions et voir visuellement les distributions et la qualité des données :

```bash
uv run streamlit run app_streamlit.py
```

### Via la Ligne de Commande (CLI)

Utile pour automatiser sur un serveur ou scripter.

```bash
uv run python run.py chemin/vers/le/dataset.csv nom_de_la_colonne_cible
```

#### Exemples de split customisés depuis le CLI :

```bash
# Split temporel (ex: pour des séries temporelles ou des logs)
uv run python run.py data/logs.csv status --strategy temporal --time-col timestamp --test-size 0.3

# Split groupé (garantit que l'ID d'un client n'est pas coupé entre train et test)
uv run python run.py data/patients.csv is_sick --strategy group --group-col patient_id
```

### Via l'API REST locale

Permet d'intégrer le pipeline à un backend tiers ou un frontend React.

```bash
uv run uvicorn src.api.app:app --reload
```

Par defaut, l'API ne lit que les CSV places sous `data/` et refuse toutes les
URL. Configurez `DATASET_ALLOWED_ROOTS` (separe par `;` sous Windows ou `:` sous
Linux) et, uniquement si necessaire, `DATASET_ALLOWED_HOSTS` pour autoriser des
hotesses HTTPS precises. La taille maximale est 25 Mio, ajustable avec
`MAX_DATASET_BYTES`.

## Confidentialite et limites

- Les lignes brutes ne sont jamais placees dans les prompts : seuls le schema,
  les types et des statistiques agregees sont transmis au fournisseur LLM.
- `OPENAI_API_KEY` n'est chargee qu'au premier appel LLM. Les regles
  deterministes et leurs tests fonctionnent hors ligne avec un faux client
  injectable.
- Le cache LLM persistant est desactive par defaut, car une reponse peut contenir
  des donnees reconstruites. `LLM_CACHE_ENABLED=1` l'active explicitement ; il
  reste alors sous la responsabilite de l'operateur (retention, chiffrement du
  disque et suppression).
- L'outil assiste la preparation des donnees mais ne garantit ni leur conformite
  reglementaire, ni l'absence de biais, ni la qualite d'un modele aval. Les
  validations humaines restent obligatoires.

---

## 🏗️ Structure du Dépôt

L'architecture est granulaire et conçue pour être facilement testée (les tests unitaires sont très denses) :

```text
.
├── config/                  # Layout des configurations (Modèle, logs, prompts versionnés)
├── data/
│   ├── cache/               # Base SQLite pour le state LangGraph et les embeddings LLM
│   └── outputs/             # Vos CSV nettoyés finissent ici (ainsi que les scalers .joblib)
├── notebooks/               # Benchmarks, démos interactives, tests de promping
├── src/
│   ├── agents/              # Le code métier des 4 agents et le StateGraph (pipeline.py)
│   ├── api/                 # Endpoints FastAPI
│   ├── llm/                 # Enrobage custom autour d'OpenAI (Cache local pour économiser de la thune)
│   ├── prompt_engineering/  # Parsing Pydantic et gestion des templates YAML
│   ├── tracking/            # Intégration W&B
│   └── utils/               # Loggers
├── tests/                   # +40 tests unitaires sur les règles déterministes (!llm)
├── app_streamlit.py         # Dashboard final
├── pyproject.toml           # Définition uv du package
└── run.py                   # Entrée CLI
```

---

## ✨ Points techniques intéressants

*   **Cache LLM local optionnel** : la clé est un SHA-256 des prompts et paramètres. Le stockage SQLite des réponses est explicitement opt-in avec `LLM_CACHE_ENABLED=1`.
*   **Fallback & Sécurité** : Les réponses du LLM sont validées par `Pydantic`. S'il hallucine ou se trompe de format JSON, une boucle de réparation est déclenchée (jusqu'à 3 _retries_ exponentiels) avant de lever une erreur, assurant la résilience du script.
*   **Transparence des Prompts** : Les prompts ne sont pas noyés dans le code source. Ils sont centralisés et versionnés dans `config/prompt_templates.yaml`, ce qui facilite les itérations et le benchmark.

---

## 🧪 Tests

Les comportements statistiques déterministes (détection de bimodalité de Sarle, logiques de fallback sur les splits, vérifications de cibles) sont fiabilisés par un solide harnais de tests (pytest) :

```bash
# Lancer les tests
uv run pytest tests/ -v
```

---


