"""
Projet : HR Analytics - Job Change of Data Scientists
Auteur : [Votre nom]

Objectif :
- Charger le fichier CSV
- Explorer les données
- Analyser les valeurs manquantes et la variable cible
- Vérifier les doublons et les catégories
- Nettoyer les données
- Enregistrer le dataset nettoyé et les graphiques

Utilisation :
    python analyse_hr.py --data aug_train.csv

Dépendances :
    pip install pandas numpy matplotlib seaborn
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Configuration générale des graphiques
sns.set_theme(style="whitegrid")


def charger_donnees(chemin_csv):
    """Charge le fichier CSV et renvoie un DataFrame."""
    chemin = Path(chemin_csv)

    if not chemin.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin}\n"
            "Indique le bon chemin avec --data, par exemple : "
            "python analyse_hr.py --data aug_train.csv"
        )

    df = pd.read_csv(chemin)
    print("\n=== 1. CHARGEMENT DES DONNÉES ===")
    print(f"Fichier chargé : {chemin}")
    print(f"Nombre de lignes : {df.shape[0]}")
    print(f"Nombre de colonnes : {df.shape[1]}")
    return df


def explorer_donnees(df):
    """Affiche un aperçu, les types et les statistiques descriptives."""
    print("\n=== 2. EXPLORATION INITIALE ===")
    print("\nPremières lignes :")
    print(df.head())

    print("\nDimensions :", df.shape)
    print("\nNoms des colonnes :")
    print(df.columns.tolist())

    print("\nInformations générales :")
    df.info()

    print("\nStatistiques des colonnes numériques :")
    print(df.describe())

    print("\nStatistiques de toutes les colonnes :")
    print(df.describe(include="all"))


def analyser_valeurs_manquantes(df, dossier_sortie):
    """Compte et visualise les valeurs manquantes avant nettoyage."""
    print("\n=== 3. ANALYSE DES VALEURS MANQUANTES ===")

    tableau = pd.DataFrame({
        "Valeurs_manquantes": df.isnull().sum(),
        "Pourcentage": df.isnull().mean() * 100,
    }).sort_values("Pourcentage", ascending=False)

    print(tableau.round(2))

    # Enregistrer le tableau pour pouvoir l'utiliser dans le rapport
    tableau.to_csv(dossier_sortie / "rapport_valeurs_manquantes_avant.csv")

    pourcentages = tableau.loc[tableau["Pourcentage"] > 0, "Pourcentage"]
    if not pourcentages.empty:
        ax = pourcentages.sort_values().plot(
            kind="barh", figsize=(10, 6)
        )
        ax.set_title("Pourcentage de valeurs manquantes par variable")
        ax.set_xlabel("Valeurs manquantes (%)")
        ax.set_ylabel("Variables")
        plt.tight_layout()
        plt.savefig(dossier_sortie / "valeurs_manquantes_avant.png", dpi=150)
        plt.show()
        plt.close()


def analyser_variable_cible(df, dossier_sortie):
    """Analyse la répartition de target et crée un graphique."""
    print("\n=== 4. ANALYSE DE LA VARIABLE CIBLE ===")

    if "target" not in df.columns:
        print("La colonne 'target' n'existe pas dans ce fichier.")
        return

    effectifs = df["target"].value_counts(dropna=False).sort_index()
    pourcentages = (df["target"].value_counts(normalize=True, dropna=False) * 100).sort_index()

    print("\nEffectifs :")
    print(effectifs)
    print("\nPourcentages (%) :")
    print(pourcentages.round(2))

    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x="target")
    plt.title("Répartition des candidats selon leur intention de changer d'emploi")
    plt.xlabel("Target : 0 = Non, 1 = Oui")
    plt.ylabel("Nombre de candidats")
    plt.tight_layout()
    plt.savefig(dossier_sortie / "repartition_target.png", dpi=150)
    plt.show()
    plt.close()


def verifier_doublons(df):
    """Affiche le nombre de doublons complets et d'identifiants répétés."""
    print("\n=== 5. VÉRIFICATION DES DOUBLONS ===")
    print("Lignes entièrement dupliquées :", df.duplicated().sum())

    if "enrollee_id" in df.columns:
        print(
            "Identifiants répétés :",
            df["enrollee_id"].duplicated().sum()
        )


def examiner_categories(df):
    """Affiche les modalités des variables catégorielles."""
    print("\n=== 6. EXAMEN DES VARIABLES CATÉGORIELLES ===")

    colonnes = [
        "gender",
        "relevent_experience",
        "enrolled_university",
        "education_level",
        "major_discipline",
        "experience",
        "company_size",
        "company_type",
        "last_new_job",
    ]

    for col in colonnes:
        if col in df.columns:
            print(f"\n--- {col} ---")
            print(df[col].value_counts(dropna=False))


def nettoyer_donnees(df):
    """
    Nettoie les données de manière prudente :
    - travaille sur une copie ;
    - remplace les valeurs manquantes de certaines colonnes catégorielles par
      'Unknown' (on ne devine pas les informations absentes) ;
    - harmonise la modalité '10/49' de company_size en '10-49' ;
    - supprime uniquement les lignes entièrement identiques.
    """
    print("\n=== 7. NETTOYAGE DES DONNÉES ===")
    df_clean = df.copy()

    lignes_avant = len(df_clean)

    colonnes_categorielles = [
        "gender",
        "enrolled_university",
        "education_level",
        "major_discipline",
        "experience",
        "company_size",
        "company_type",
        "last_new_job",
    ]

    for col in colonnes_categorielles:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("Unknown")

    # Normaliser une catégorie incohérente repérée pendant l'exploration
    if "company_size" in df_clean.columns:
        df_clean["company_size"] = df_clean["company_size"].replace({
            "10/49": "10-49"
        })

    # Supprimer uniquement les lignes complètement identiques.
    # Les identifiants répétés seuls ne sont pas supprimés automatiquement.
    doublons = df_clean.duplicated().sum()
    if doublons > 0:
        df_clean = df_clean.drop_duplicates()

    print(f"Lignes avant nettoyage : {lignes_avant}")
    print(f"Lignes après nettoyage : {len(df_clean)}")
    print(f"Lignes entièrement dupliquées supprimées : {doublons}")
    print("\nValeurs manquantes restantes par colonne :")
    print(df_clean.isnull().sum().sort_values(ascending=False))

    return df_clean


def verifier_nettoyage(df_clean, dossier_sortie):
    """Affiche et enregistre les contrôles après nettoyage."""
    print("\n=== 8. CONTRÔLE FINAL ===")
    print("Dimensions finales :", df_clean.shape)
    print("\nTypes des colonnes :")
    print(df_clean.dtypes)

    print("\nDoublons complets restants :", df_clean.duplicated().sum())

    if "experience" in df_clean.columns:
        print("\nModalités de experience :")
        print(df_clean["experience"].value_counts(dropna=False))

    if "target" in df_clean.columns:
        print("\nRépartition de target :")
        print(df_clean["target"].value_counts(dropna=False))

    print("\nAperçu du dataset nettoyé :")
    print(df_clean.head())

    # Export du dataset nettoyé
    fichier_nettoye = dossier_sortie / "hr_analytics_nettoye.csv"
    df_clean.to_csv(fichier_nettoye, index=False)
    print(f"\nDataset nettoyé enregistré dans : {fichier_nettoye}")

    # Rapport des valeurs manquantes après nettoyage
    rapport_final = pd.DataFrame({
        "Valeurs_manquantes": df_clean.isnull().sum(),
        "Pourcentage": df_clean.isnull().mean() * 100,
    }).sort_values("Pourcentage", ascending=False)

    rapport_final.to_csv(dossier_sortie / "rapport_valeurs_manquantes_apres.csv")
    print(f"Rapport final enregistré dans : {dossier_sortie / 'rapport_valeurs_manquantes_apres.csv'}")


def main():
    """Fonction principale : orchestre toutes les étapes dans l'ordre."""
    parser = argparse.ArgumentParser(
        description="Exploration et nettoyage du dataset HR Analytics."
    )
    parser.add_argument(
        "--data",
        default="aug_train.csv",
        help="Chemin vers le fichier CSV d'entraînement (défaut : aug_train.csv).",
    )
    parser.add_argument(
        "--output",
        default="resultats_hr",
        help="Dossier où enregistrer les résultats (défaut : resultats_hr).",
    )
    args = parser.parse_args()

    dossier_sortie = Path(args.output)
    dossier_sortie.mkdir(parents=True, exist_ok=True)

    df = charger_donnees(args.data)
    explorer_donnees(df)
    analyser_valeurs_manquantes(df, dossier_sortie)
    analyser_variable_cible(df, dossier_sortie)
    verifier_doublons(df)
    examiner_categories(df)

    df_clean = nettoyer_donnees(df)
    verifier_nettoyage(df_clean, dossier_sortie)

    print("\n=== FIN DU SCRIPT ===")
    print("Exploration et nettoyage terminés.")


if __name__ == "__main__":
    main()
