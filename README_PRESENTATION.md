# Raport Proiect Data Mining: Detecția Stării Sistemelor Distribuite

## 1. Introducere și Obiectiv

**Scopul Proiectului:** Dezvoltarea unui sistem de Machine Learning capabil să clasifice starea de sănătate a unei arhitecturi distribuite (`Healthy`, `Degraded`, `Cascading Failure`, `Total Outage`) pe baza metricilor de performanță (CPU, RAM, Latență) și a configurației arhitecturale.

**Setul de Date:** `distributed_system_architecture_stress_dataset.csv`
- **Metrici:** CPU, Memory, Request Rate, Payload Size, etc.
- **Arhitectură:** Monolith vs Microservices, Database Count, etc.
- **Target:** `system_state` (4 clase).

---

## 2. Parcursul Dezvoltării (The Journey)

Proiectul a evoluat iterativ, trecând prin mai multe etape de rafinare pentru a asigura realismul și robustețea soluției.

### Faza 1: Abordarea Inițială (Naivă)
- Am antrenat modelele (Random Forest, Gradient Boosting, Logistic Regression) pe setul brut.
- **Rezultat:** Acuratețe de **100% (1.0)**.
- **Diagnosticul Expertului:** *Overfitting* masiv sau *Target Leakage*. Modelele "trișau" folosind variabile care erau consecințe directe ale stării (ex: `error_rate_percent`, `circuit_breaker_open`).

### Faza 2: Creșterea Realismului
- Am eliminat coloanele "diagnostic" (`error_rate_percent`, `circuit_breaker_open`, `retry_storm_detected`) pentru a forța modelul să învețe pattern-uri subtile din metricile de infrastructură (CPU, RAM, Network).
- **Rezultat:** Acuratețea a scăzut la **~72%**. Modelele nu mai reușeau să detecteze deloc clasa critică `Total Outage` (Recall 0.00).

### Faza 3: Tratarea Dezechilibrului (Class Imbalance)
- Clasa `Total Outage` avea doar 18 exemple în setul de antrenament (vs 60.000 pentru Healthy), fiind imposibil de învățat.
- **Soluție:** Am implementat **SMOTE (Synthetic Minority Over-sampling Technique)**.
- **Impact:** Setul de antrenament a fost echilibrat artificial, fiecare clasă având acum ~62.000 de exemple.

### Faza 4: Optimizare pentru Siguranță (Recall Focus)
- Într-un sistem critic, este inacceptabil să ratezi un "Total Outage".
- Am schimbat metrica de optimizare în `GridSearchCV` de la `accuracy` la **`recall_macro`**.
- Am reintrodus `p95_latency_ms` ca feature valid (fiind un indicator de performanță, nu neapărat o eroare fatală).

---

## 3. Metodologie Tehnică

### Pipeline de Preprocesare
1.  **Curățare:** Imputare valori lipsă (Mediana/Mod), eliminare duplicate.
2.  **Feature Selection:** Eliminare variabile cu *leakage*.
3.  **Encoding:** One-Hot Encoding pentru categorii, Label Encoding pentru target.
4.  **Scaling:** `StandardScaler` pentru normalizarea metricilor numerice.
5.  **Balancing:** Aplicare **SMOTE** pe setul de training.

### Modele Utilizate și Tuning
Am folosit `GridSearchCV` (Cross-Validation cu 3 fold-uri) pentru a găsi hiperparametrii optimi:

| Model | Parametri Optimi Identificați | Observații |
| :--- | :--- | :--- |
| **Random Forest** | `max_depth: 20`, `n_estimators: 50` | Robust, dar tinde să favorizeze clasele majoritare. |
| **Gradient Boosting** | `lr: 0.05`, `n_estimators: 100` | Foarte precis, dar sensibil la noise. |
| **Logistic Regression** | `C: 0.1`, `solver: lbfgs` | Simplu, liniar, dar surprinzător de eficient pe date echilibrate. |

---

## 4. Rezultate Finale și Analiză

Evaluarea finală pe setul de Test (20.000 instanțe) a arătat rezultate remarcabile:

### Performanță Generală
*   **Gradient Boosting:** Acuratețe **87.49%**, Recall Macro **0.94**.
*   **Logistic Regression:** Acuratețe **87.22%**, Recall Macro **0.85**.
*   **Random Forest:** Acuratețe **87.19%**, Recall Macro **0.85**.

### Detecția Stărilor Critice (`Total Outage`)
Aici s-a dat "bătălia" reală. Pe cele 3 cazuri de test de *Total Outage*:
*   **Gradient Boosting:** A detectat **3/3** (Recall 1.00). Perfect!
*   **Random Forest:** A detectat **2/3** (Recall 0.67).
*   **Logistic Regression:** A detectat **2/3** (Recall 0.67), dar cu multe alarme false (Precision 0.22).

### Simulare Producție (Setul de Predicție)
Pe setul de 60.000 de instanțe noi ("Prediction"), am evaluat capacitatea modelelor de a generaliza (Recall Macro):
*   Random Forest: 0.8533
*   Gradient Boosting: 0.9102
*   **Logistic Regression: 0.9361**

**Câștigător:** Deși Gradient Boosting a fost mai precis pe test, **Logistic Regression** a demonstrat cel mai mare Recall mediu pe setul mare de predicție (0.9361), fiind ales ca modelul final pentru robustețea sa în detectarea anomaliilor (chiar dacă generează mai multe alarme false, preferăm *False Positives* decât *False Negatives* într-un sistem de monitorizare).

---

## 5. Concluzii

1.  **Realism vs. Perfecțiune:** Eliminarea variabilelor evidente a transformat o problemă trivială într-una complexă, scăzând inițial acuratețea dar crescând utilitatea reală a modelului.
2.  **Puterea SMOTE:** Fără SMOTE, detectarea `Total Outage` era imposibilă (0%). Cu SMOTE, am atins 67%-100%.
3.  **Recall este Rege:** Pentru monitorizarea infrastructurii, optimizarea pentru Recall este vitală.
4.  **Simplitatea câștigă uneori:** Logistic Regression, fiind un model mai simplu, a generalizat excelent pe setul masiv de predicție, depășind algoritmi mai complecși la capitolul Recall Macro.

---
*Acest proiect demonstrează un flux complet de Data Science: de la analiza exploratorie și curățare, la strategii avansate de echilibrare și tuning, finalizând cu o soluție robustă gata de producție.*
