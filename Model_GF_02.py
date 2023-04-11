# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 19:42:11 2023

@author: naouf
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog
from scipy.integrate import solve_ivp


def creer_interface_principale():
    fenetre = tk.Tk()
    fenetre.title("Analyse des constantes cinétiques")
    fenetre.configure(bg='white')

    # Créer des labels et des champs de saisie pour chaque donnée
    label_t = tk.Label(fenetre, text="Durée de la réaction en minutes")
    entry_t = tk.Entry(fenetre)
    
    label_ic = tk.Label(fenetre, text="Densité de courant de corrosion (A/m2)")
    entry_ic = tk.Entry(fenetre)

    label_V = tk.Label(fenetre, text="Volume de la solution (m3)")
    entry_V = tk.Entry(fenetre)

    label_S = tk.Label(fenetre, text="Surface de l’électrode (m2)")
    entry_S = tk.Entry(fenetre)

    label_pH_initial = tk.Label(fenetre, text="pH initial de la solution")
    entry_pH_initial = tk.Entry(fenetre)

    label_HSO4_initial = tk.Label(fenetre, text="Concentration initiale de H2SO4 (mol/m3)")
    entry_HSO4_initial = tk.Entry(fenetre)

    label_O2_initial = tk.Label(fenetre, text="Concentration initiale d’oxygène (mol/m3)")
    entry_O2_initial = tk.Entry(fenetre)

    label_H2O2_initial = tk.Label(fenetre, text="Concentration initiale de H2O2 (mol/m3)")
    entry_H2O2_initial = tk.Entry(fenetre)

    label_VM_initial = tk.Label(fenetre, text="Concentration initiale de VM (mol/m3)")
    entry_VM_initial = tk.Entry(fenetre)

    # Positionner les labels et les champs de saisie dans la fenêtre
    label_t.grid(row=0, column=0)
    entry_t.grid(row=0, column=1)

    label_ic.grid(row=1, column=0)
    entry_ic.grid(row=1, column=1)

    label_V.grid(row=2, column=0)
    entry_V.grid(row=2, column=1)

    label_S.grid(row=3, column=0)
    entry_S.grid(row=3, column=1)

    label_pH_initial.grid(row=4, column=0)
    entry_pH_initial.grid(row=4, column=1)

    label_HSO4_initial.grid(row=5, column=0)
    entry_HSO4_initial.grid(row=5, column=1)

    label_O2_initial.grid(row=6, column=0)
    entry_O2_initial.grid(row=6, column=1)

    label_H2O2_initial.grid(row=7, column=0)
    entry_H2O2_initial.grid(row=7, column=1)

    label_VM_initial.grid(row=8, column=0)
    entry_VM_initial.grid(row=8, column=1)
    # Créer le bouton "Soumettre"
    bouton_soumettre = tk.Button(
        fenetre, text="Soumettre",
        command=lambda: soumettre_donnees(
            entry_t, entry_ic, entry_V, entry_S, entry_pH_initial,
            entry_HSO4_initial, entry_O2_initial, entry_H2O2_initial, entry_VM_initial
        )
    )
    bouton_soumettre.grid(row=9, column=0, columnspan=2)
    return entry_t, entry_ic, entry_V, entry_S, entry_pH_initial, entry_HSO4_initial, entry_O2_initial, entry_H2O2_initial, entry_VM_initial


def soumettre_donnees(entry_t, entry_ic, entry_V, entry_S, entry_pH_initial, entry_HSO4_initial, entry_O2_initial, entry_H2O2_initial, entry_VM_initial):
  
    # Récupérer les données des champs de saisie et les convertir en types appropriés
    t = float(entry_t.get())
    ic = float(entry_ic.get())
    V = float(entry_V.get())
    S = float(entry_S.get())
    pH_initial = float(entry_pH_initial.get())
    HSO4_initial = float(entry_HSO4_initial.get())
    O2_initial = float(entry_O2_initial.get())
    H2O2_initial = float(entry_H2O2_initial.get())
    VM_initial = float(entry_VM_initial.get())


def lire_fichier_excel(fichier):
    # Lire le fichier Excel en utilisant pandas
    df = pd.read_excel(fichier)

    # Extraire les colonnes pertinentes pour votre analyse
    t = df['Temps']
    dVM_dt = df['dVM_dt']  # Assurez-vous d'avoir cette colonne dans votre fichier Excel

    return t, dVM_dt

def afficher_courbe_experimentale(t, dVM_dt):
        # Créer un graphique avec les données expérimentales
    # Ici, on trace les points des données expérimentales en utilisant des cercles ('o') pour les marqueurs et la couleur bleue ('b')
    plt.plot(t, dVM_dt, 'o', color='blue', label='Données expérimentales')

    # Ajouter des légendes pour les axes
    # Ici, on définit les étiquettes pour l'axe des abscisses (x) et l'axe des ordonnées (y) avec la couleur noire
    plt.xlabel('Durée de la réaction en minutes', color='black')
    plt.ylabel('Variation de la concentration de VM (dVM_dt)', color='black')

    # Ajouter un titre au graphique
    # Ici, on donne un titre à notre graphique pour décrire ce qu'il représente
    plt.title('Courbe expérimentale de la variation de la concentration de VM dans le temps')

    # Afficher la légende
    # Ici, on demande à afficher la légende du graphique, qui contient les informations sur les tracés
    plt.legend()

    # Modifier la couleur des axes
    ax = plt.gca()  # Récupérer l'objet Axes actuel
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    # Modifier la couleur des étiquettes des axes
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    # Afficher le graphique
    # Ici, on demande à afficher le graphique à l'écran
    plt.show()

def modele(t, ka, kb, kc, kd):
      
# Définition des constantes de vitesse
    k1 = 6.3e4
    k2 = 2.0e-6
    k28 = 3.3e4
    k47 = 1.58e5
    k48 = 1.0e7
    k3 = 3.2e5
    k4 = 1.2e3
    k5 = 3.6e2
    k6 = 1.0e4
    k7 = 5.0e4
    k29 = 5.2e6
    k30 = 8.3e2
    k49 = 1.0e7
    k37 = 7.1e6
    k38 = 1.01e7
    k31 = 9.7e4
    k32 = 5.0e-4
    k33 = 1.3e-4
    k8 = 0  # Négligeable
    k9 = 5.0e4
    k10 = 2.29e8
    k16 = 3.89e9
    k17 = 4.47e7
    k51 = 3.47e8
    k39 = 1.4e4
    k40 = 3.5e2
    k45 = 3.0e5
    k46 = 1.4e4
    k34 = 1.2e4
    k50 = 3.5e6
    k11 = 3.0e5
    k12 = 1.0e10
    k18 = 1.0e10
    k19 = 1.0e10
    k52 = 1.0e10
    k20 = 2.9e7
    k21 = 7.62e3
    k22 = 8.0e3
    k13 = 2.0e-6
    k23 = 1.0e7
    k24 = 1.0e4
    k25 = 1.0e4
    k26 = 3.1e4
    k27 = 1.0e7
    k14 = 2.3e-3
    k35 = 2.0e3
    k36 = 1.0e7
    k15 = 2.3e-3

# Conditions initiales des concentrations des espèces chimiques
    n = 2
    F = 96500
    Fe2_initial = 0 
    Fe3_initial = 0 
    HO_initial = 0
    HO2_initial = 0
    O2_minus_initial = 0
    SO4_2_initial = 0 
    SO4_minus_initial= 0
    FeSO4_initial = 0
    FeSO4_plus_initial = 0
    FeSO4_2_minus_initial = 0
    FeOH2_plus_initial = 0
    FeOH2_plus2_initial = 0
    FeHO2_2_plus_initial = 0
    Fe2OH2_4_plus_initial = 0
    FeOH_OH2_plus_initial = 0
    H_plus_initial = 10**(-pH_initial) 
    OH_minus_initial = 10**(pH_initial - 14)

    # Votre code pour définir les conditions initiales
    
    conditions_initiales = [Fe2_initial, Fe3_initial, HO_initial, HO2_initial, O2_minus_initial, SO4_2_initial, SO4_minus_initial, FeSO4_initial, FeSO4_plus_initial, FeSO4_2_minus_initial, FeOH2_plus_initial, FeOH2_plus2_initial, FeHO2_2_plus_initial, Fe2OH2_4_plus_initial, FeOH_OH2_plus_initial, H_plus_initial, OH_minus_initial]
    
    # Fonction pour définir le vecteur de temps
    temps_debut = 0
    temps_fin = 100
    nombre_de_points = 1000
    
    t = np.linspace(temps_debut, temps_fin, nombre_de_points)
    
    # Résoudre les équations différentielles
    solution = solve_ivp(equations_differenielles, (t[0], t[-1]), conditions_initiales, t_eval=t, args=(ka, kb, kc, kd))
    
        # Calculer dVM_dt en utilisant les concentrations résolues des espèces chimiques
    HO = solution.y[0]
    SO4_minus = solution.y[1]
    HO2 = solution.y[2]
    O2_minus = solution.y[3]
     
    dVM_dt = (ka * HO + kb * SO4_minus + kc * HO2 + kd * O2_minus) * VM_initial
    
    # Définir les équations différentielles pour les concentrations des espèces chimiques
def equations_differenielles(t, y, ka, kb, kc, kd):
    # Votre code pour définir les 20 équations différentielles
        
    dFe2_dt = (ic * S) / (n * F * V) - k1 * Fe2 * H2O2 + k2 * Fe3 * H2O2 - k3 * HO * Fe2 - k4 * HO2 * Fe2 + k5 * HO2 * Fe3 - k6 * O2_minus * Fe2 + k7 * O2_minus * Fe3 - k8 * Fe2 * O2 + k9 * O2_minus * Fe3 - k10 * Fe2 * SO4_2 - k11 * SO4_minus * Fe2 + k12 * FeSO4 + k13 * FeOH2_plus * H2O2 + k14 * FeOH2_plus2 + k15 * FeOH_OH2_plus
    dFe3_dt = k1 * Fe2 * H2O2 - k2 * Fe3 * H2O2 + k3 * HO * Fe2 + k4 * HO2 * Fe2 - k5 * HO2 * Fe3 + k6 * O2_minus * Fe2 - k7 * O2_minus * Fe3 + k8 * Fe2 * O2 - k9 * O2_minus * Fe3 + k11 * SO4_minus * Fe2 - k16 * Fe3 * SO4_2 - k17 * Fe3 * SO4_2**2 + k18 * FeSO4_plus + k19 * FeSO4_2_minus - k20 * Fe3 - k21 * Fe3 - k22 * Fe3**2 + k23 * FeOH2_plus * H_plus + k24 * FeOH2_plus2 * H_plus**2 + k25 * FeOH2_plus2 * H_plus**2 - k26 * Fe3 * H2O2 + k27 * FeOH2_plus2 * H_plus
    dH2O2_dt = -k1 * Fe2 * H2O2 - k2 * Fe3 * H2O2 - k28 * H2O2 * HO + k4 * HO2 * Fe2 + k6 * O2_minus * Fe2 + k29 * HO**2 + k30 * HO2**2 + k31 * HO2 * O2_minus - k32 * HO2 * H2O2 - k33 * O2_minus * H2O2 - k34 * SO4_minus * H2O2 -k13 * FeOH2_plus * H2O2 - k26 * Fe3 * H2O2 + k27 * FeOH2_plus2 * H_plus - k35 * FeOH2_plus * H2O2 + k36 * FeOH_OH2_plus * H_plus
    dHO_dt = k1 * Fe2 * H2O2 - k28 * H2O2 * HO - k3 * Fe2 * HO - k29 * HO**2 - k37 * HO * HO2 - k38 * HO * O2_minus + k32 * HO2 * H2O2 + k33 * O2_minus * H2O2 - k39 * SO4_2 * HO - k40 * HSO4 * HO + k45 * SO4_minus + k46 * SO4_minus * OH_minus
    dHO2_dt = k2 * Fe3 * H2O2 + k28 * H2O2 * HO - k47 * HO2 + k48 * O2_minus * H_plus - k4 * HO2 * Fe2 - k5 * HO2 * Fe3 - k30 * HO2**2 + k49 * O2_minus * H_plus - k37 * HO * HO2 - k31 * HO2 * O2_minus - k32 * HO2 * H2O2 + k34 * SO4_minus * H2O2 - k50 * SO4_minus * HO2 + k13 * FeOH2_plus * H2O2 + k14 * FeHO2_2_plus + k15 * FeOH_OH2_plus
    dO2_minus_dt = k47 * HO2 - k48 * O2_minus * H_plus - k6 * O2_minus * Fe2 - k7 * O2_minus * Fe3 - k49 * O2_minus * H_plus - k38 * HO * O2_minus - k31 * HO2 * O2_minus - k33 * O2_minus * H2O2 + k8 * Fe2 * O2_minus - k9 * Fe3 * O2_minus
    dSO4_2_dt = -k10 * Fe2 * SO4_2 - k16 * Fe3 * SO4_2 - k17 * Fe3 * SO4_2**2 - k51 * H_plus * SO4_2 - k39 * SO4_2 * HO + k45 * SO4_minus + k46 * SO4_minus * OH_minus + k34 * SO4_minus * H2O2 + k50 * SO4_minus * HO2 + k11 * SO4_minus * Fe2 + k12 * FeSO4 + k18 * FeSO4_plus + k19 * FeSO4_2_minus + k52 * HSO4
    dHSO4_dt = k51 * H_plus * SO4_2 - k40 * HSO4 * HO - k52 * HSO4  
    dO2_dt = k5 * HO2 * Fe3 + k7 * O2_minus * Fe3 + k30 * HO2**2 + k37 * HO * HO2 + k38 * HO * O2_minus + k31 * HO2 * O2_minus + k32 * HO2 * H2O2 + k33 * O2_minus * H2O2 - k8 * Fe2 * O2 + k9 * Fe3 * O2_minus + k50 * SO4_minus * HO2
    dSO4_minus_dt = k39 * SO4_2 * HO + k40 * HSO4 * HO - k45 * SO4_minus - k46 * SO4_minus * HO - k34 * SO4_minus * H2O2 - k50 * SO4_minus * HO2 - k11 * SO4_minus * Fe2
    dFeSO4_dt = k10 * Fe2 * SO4_2 - k12 * FeSO4
    dFeSO4_plus_dt = k16 * Fe3 * SO4_2 - k18 * FeSO4_plus
    dFeSO4_2_minus_dt = k17 * Fe3 * SO4_2**2 - k19 * FeSO4_2_minus
    dFeOH2_plus_dt = k20 * Fe3 - k13 * FeOH2_plus * H2O2 - k23 * FeOH2_plus * H_plus + k36 * FeOH_OH2_plus * H_plus
    dFeOH2_plus2_dt = k21 * Fe3 - k24 * FeOH2_plus2 * (H_plus ** 2)
    dFeHO2_2_plus_dt = -k27 * FeHO2_2_plus * H_plus - k14 * FeHO2_2_plus
    dFe2OH2_4_plus_dt = k22 * Fe3**2 - k25 * Fe2OH2_4_plus * H_plus**2
    dFeOH_OH2_plus_dt = k35 * FeOH2_plus * H2O2 - k36 * FeOH_OH2_plus * H_plus - k15 * FeOH_OH2_plus
    dH_plus_dt = k2 * Fe3 * H2O2 + k47 * HO2 - k48 * O2_minus * H_plus + k5 * HO2 * Fe3 - k49 * O2_minus * H_plus + k34 * SO4_minus * H2O2 + k50 * SO4_minus * HO2 - k51 * H_plus * SO4_2 + k45 * SO4_minus + k20 * Fe3 + k21 * Fe3 + k22 * Fe3**2 + k13 * FeOH2_plus * H2O2 - k23 * FeOH2_plus * H_plus - k24 * FeHO2_2_plus * H_plus**2 - k25 * Fe2OH2_4_plus * H_plus**2 + k26 * Fe3 * H2O2 - k27 * FeHO2_2_plus * H_plus - k35 * FeOH2_plus * H2O2 - k36 * FeOH_OH2_plus * H_plus + k52 * HSO4
    dOH_minus_dt = k1 * Fe2 * H2O2 + k3 * Fe2 * HO + k4 * HO2 * Fe2 + k6 * O2_minus * Fe2 + k38 * HO * O2_minus + k31 * HO2 * O2_minus + k33 * O2_minus * H2O2 + k39 * SO4_2 * HO - k46 * SO4_minus * OH_minus + k13 * FeOH2_plus * H2O2 + k15 * FeOH_OH2_plus 
    Fe2 = Fe2_initial + dFe2_dt
    Fe3 = Fe3_initial + dFe3_dt
    H2O2 = H2O2_initial +  dH2O2_dt
    HO = HO_initial + dHO_dt
    HO2 = HO2_initial + dHO2_dt
    O2_minus = O2_minus_initial + dO2_minus_dt
    SO4_2 = SO4_2_initial + dSO4_2_dt
    HSO4 = HSO4_initial + dHSO4_dt
    O2 = O2_initial + dO2_dt
    SO4_minus = SO4_minus_initial + dSO4_minus_dt  
    FeSO4 = FeSO4_initial + dFeSO4_dt
    FeSO4_plus = FeSO4_plus_initial + dFeSO4_plus_dt
    FeSO4_2_minus = FeSO4_2_minus_initial + dFeSO4_2_minus_dt
    FeOH2_plus = FeOH2_plus_initial + dFeOH2_plus_dt
    FeOH2_plus2 = FeOH2_plus2_initial + dFeOH2_plus2_dt
    FeHO2_2_plus = FeHO2_2_plus_initial + dFeHO2_2_plus_dt
    Fe2OH2_4_plus = Fe2OH2_4_plus_initial + dFe2OH2_4_plus_dt
    H_plus = H_plus_initial + dH_plus_dt
    OH_minus = OH_minus_initial + dOH_minus_dt

        # Retourner les dérivées des concentrations sous forme d'une liste
        return [dFe2_dt, dFe3_dt, dH2O2_dt, dHO_dt, dHO2_dt, dO2_minus_dt, dSO4_2_dt, dHSO4_dt, dO2_dt, dSO4_minus_dt, dFeSO4_dt, dFeSO4_plus_dt, dFeSO4_2_minus_dt, dFeOH2_plus_dt, dFeOH2_plus2_dt, dFeHO2_2_plus_dt, dFe2OH2_4_plus_dt, dFeOH_OH2_plus_dt, dH_plus_dt, dOH_minus_dt, Fe2, Fe3, H2O2, HO, HO2, O2_minus, SO4_2, HSO4, O2, SO4_minus, FeSO4, FeSO4_plus, FeSO4_2_minus, FeOH2_plus, FeOH2_plus2, FeHO2_2_plus, Fe2OH2_4_plus, FeOH_OH2_plus, H_plus, OH_minus]


def effectuer_fitting(t, dVM_dt):
    # Définir les bornes pour les constantes cinétiques ka, kb, kc et kd (si nécessaire)
    bornes_ka = (0, np.inf)  # Exemple de bornes pour ka
    bornes_kb = (0, np.inf)  # Exemple de bornes pour kb
    bornes_kc = (0, np.inf)  # Exemple de bornes pour kc
    bornes_kd = (0, np.inf)  # Exemple de bornes pour kd

    bornes = (bornes_ka[0], bornes_kb[0], bornes_kc[0], bornes_kd[0]), (bornes_ka[1], bornes_kb[1], bornes_kc[1], bornes_kd[1])

    # Effectuer le fitting avec la fonction curve_fit
    params_opt, params_cov = curve_fit(modele, t, dVM_dt, bounds=bornes)

    # Extraire les constantes cinétiques optimisées et leurs incertitudes
    ka_opt, kb_opt, kc_opt, kd_opt = params_opt
    incertitudes = np.sqrt(np.diag(params_cov))

    return params_opt, incertitudes

def afficher_resultats(params, uncertainties):
    # Créer une nouvelle fenêtre pour afficher les résultats
    fenetre_resultats = tk.Toplevel()
    fenetre_resultats.title("Résultats du fitting")

    # Extraire les constantes cinétiques et leurs incertitudes
    ka_opt, kb_opt, kc_opt, kd_opt = params
    incertitudes_ka, incertitudes_kb, incertitudes_kc, incertitudes_kd = uncertainties

    # Afficher les résultats dans la fenêtre
    label_ka = tk.Label(fenetre_resultats, text=f"ka optimisé : {ka_opt:.4e} ± {incertitudes_ka:.4e}")
    label_kb = tk.Label(fenetre_resultats, text=f"kb optimisé : {kb_opt:.4e} ± {incertitudes_kb:.4e}")
    label_kc = tk.Label(fenetre_resultats, text=f"kc optimisé : {kc_opt:.4e} ± {incertitudes_kc:.4e}")
    label_kd = tk.Label(fenetre_resultats, text=f"kd optimisé : {kd_opt:.4e} ± {incertitudes_kd:.4e}")

    label_ka.pack()
    label_kb.pack()
    label_kc.pack()
    label_kd.pack()

    # Ajouter un bouton pour fermer la fenêtre
    bouton_fermer = tk.Button(fenetre_resultats, text="Fermer", command=fenetre_resultats.destroy)
    bouton_fermer.pack()

def importer_fichier():
    # Créer une fenêtre Tkinter temporaire (elle ne sera pas affichée)
    fenetre_cachee = tk.Tk()
    fenetre_cachee.withdraw()

    # Ouvrir la boîte de dialogue pour sélectionner le fichier Excel
    fichier = filedialog.askopenfilename(title="Ouvrir un fichier", filetypes=[("Fichiers Excel", "*.xlsx")])

    # Détruire la fenêtre Tkinter temporaire
    fenetre_cachee.destroy()
    if fichier:
        return fichier
    else:
        print("Aucun fichier sélectionné.")
    return None

    return fichier

if __name__ == "__main__":
    # Ajoutez la logique pour appeler vos fonctions ici
    pass