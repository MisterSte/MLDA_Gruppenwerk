# MLDA_Gruppenwerk

Dieses Projekt wird eine 1,0

**Detaillierte Aufgabenliste für das Machine Learning Projekt**

**II. Dateneinlesen und initiale Inspektion**

3.  **Datensatz laden:** `pd.read_csv('insurance.csv')` in einen DataFrame (z.B. `df`).
4.  **Erste Datenübersicht:**
    * `df.head()` und `df.tail()` anzeigen.
    * `df.info()` zur Überprüfung von Datentypen und Non-Null-Werten.
    * `df.describe(include='all')` für deskriptive Statistiken aller Spalten.
5.  **Fehlende Werte identifizieren:** `df.isnull().sum()` anzeigen und Strategie festlegen (z.B. entfernen, imputieren), falls vorhanden.
6.  **Duplikate prüfen und behandeln:** `df.duplicated().sum()`, ggf. `df.drop_duplicates()`.

**III. Explorative Datenanalyse (EDA) & Visualisierung**

7.  **Verteilung der Zielvariable (`charges`) analysieren:**
    * Histogramm: `sns.histplot(df['charges'], kde=True)`.
    * Boxplot: `sns.boxplot(x=df['charges'])`.
    * Überlegen Sie, ob eine Transformation (z.B. Log-Transformation) für `charges` sinnvoll ist, um die Schiefe zu reduzieren. Wenn ja, führen Sie diese durch und verwenden Sie die transformierte Variable als Ziel für das Training, denken Sie aber daran, die Vorhersagen zurück zu transformieren.
8.  **Numerische Merkmale analysieren (`age`, `bmi`, `children`):**
    * Histogramme/Dichte-Plots für jedes Merkmal.
    * Boxplots für jedes Merkmal.
9.  **Kategoriale Merkmale analysieren (`sex`, `smoker`, `region`):**
    * Häufigkeitszählungen: `df['column_name'].value_counts()`.
    * Balkendiagramme: `sns.countplot(x='column_name', data=df)`.
10. **Beziehungen zur Zielvariable (`charges`) untersuchen:**
    * Numerische Merkmale vs. `charges`:
        * Scatter-Plots: z.B. `sns.scatterplot(x='age', y='charges', data=df)`. Untersuchen Sie auch mit `hue='smoker'`.
    * Kategoriale Merkmale vs. `charges`:
        * Boxplots: z.B. `sns.boxplot(x='smoker', y='charges', data=df)`.
        * Violinplots: z.B. `sns.violinplot(x='region', y='charges', data=df)`.
11. **Korrelationsanalyse:**
    * Korrelationsmatrix für numerische Features (inkl. `charges`): `df_numeric.corr()`.
    * Heatmap der Korrelationsmatrix: `sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm')`.
12. **EDA-Erkenntnisse dokumentieren:** Fassen Sie wichtige Beobachtungen zusammen (z.B. welche Variablen scheinen `charges` stark zu beeinflussen?).

**IV. Datenvorverarbeitung (Preprocessing)**

13. **Kategoriale Merkmale kodieren:**
    * `sex`: Binär kodieren (0/1) oder One-Hot-Encoding.
    * `smoker`: Binär kodieren (0/1).
    * `region`: One-Hot-Encoding (z.B. mit `pd.get_dummies(df, columns=['region'], drop_first=True)` oder `OneHotEncoder` von Scikit-learn).
14. **Merkmalsmatrix (X) und Zielvektor (y) definieren:**
    * `X` enthält alle unabhängigen (kodierten) Merkmale.
    * `y` enthält die Zielvariable `charges` (ggf. transformiert).
15. **Datensatz aufteilen:**
    * `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)` (oder eine andere `random_state` für Reproduzierbarkeit).
16. **Feature Scaling (numerische Merkmale):**
    * `StandardScaler` (oder `MinMaxScaler`) initialisieren.
    * Scaler auf `X_train` anpassen: `scaler.fit(X_train[['age', 'bmi', 'children']])`.
    * `X_train` und `X_test` transformieren: `X_train[['age', 'bmi', 'children']] = scaler.transform(X_train[['age', 'bmi', 'children']])`, entsprechend für `X_test`. Achten Sie darauf, nur die numerischen Spalten zu skalieren, die nicht aus dem One-Hot-Encoding stammen.

**V. Modelltraining und Hyperparameter-Optimierung (Iterativer Prozess)**

* **Allgemein:** Definieren Sie eine Kreuzvalidierungsstrategie (z.B. `KFold(n_splits=5, shuffle=True, random_state=42)`).

17. **Lineare Regression:**
    * Modell initialisieren: `LinearRegression()`.
    * Modell auf `X_train`, `y_train` trainieren.
    * *(Optional mit `statsmodels` für detaillierte Statistiken: `sm.OLS(y_train, sm.add_constant(X_train)).fit()`)*.
18. **Ridge Regression:**
    * Modell initialisieren: `Ridge()`.
    * Hyperparameter-Raster für Alpha definieren: z.B. `{'alpha': [0.01, 0.1, 1, 10, 100]}`.
    * `GridSearchCV` mit Ridge, dem Parameter-Raster und CV-Strategie verwenden, um optimales Alpha zu finden.
    * Bestes Ridge-Modell mit optimalem Alpha auf gesamten `X_train` trainieren.
19. **Lasso Regression:**
    * Modell initialisieren: `Lasso()`.
    * Hyperparameter-Raster für Alpha definieren.
    * `GridSearchCV` mit Lasso, dem Parameter-Raster und CV-Strategie verwenden.
    * Bestes Lasso-Modell mit optimalem Alpha auf gesamten `X_train` trainieren.
    * Koeffizienten des Lasso-Modells inspizieren, um Feature Selection zu bewerten (`lasso_model.coef_`).
20. **Entscheidungsbaum-basierte Methode (z.B. Random Forest Regressor):**
    * Modell initialisieren: `RandomForestRegressor(random_state=42)`.
    * Hyperparameter-Raster definieren: z.B. `{'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}`.
    * `GridSearchCV` verwenden.
    * Bestes Modell auf gesamten `X_train` trainieren.
    * Merkmalswichtigkeiten analysieren (`rf_model.feature_importances_`).

**VI. Modellevaluierung**

21. **Vorhersagen auf dem Testset generieren:** Für jedes trainierte Modell `model.predict(X_test)`.
22. **Metriken berechnen und vergleichen:**
    * Mean Absolute Error (MAE): `mean_absolute_error(y_test, y_pred)`.
    * Mean Squared Error (MSE): `mean_squared_error(y_test, y_pred)`.
    * Root Mean Squared Error (RMSE): `np.sqrt(mean_squared_error(y_test, y_pred))`.
    * R-squared (R²): `r2_score(y_test, y_pred)`.
    * Erstellen Sie eine Tabelle oder einen DataFrame, um die Metriken aller Modelle übersichtlich darzustellen.
23. **Residuenanalyse (besonders für lineare Modelle):**
    * Residuen berechnen: `y_test - y_pred`.
    * Scatter-Plot der Vorhersagen gegen Residuen (sollte keine klare Struktur zeigen).
    * Histogramm der Residuen (sollte annähernd normalverteilt sein).
24. **Vorhersagen zurücktransformieren (falls `charges` transformiert wurde):** Bevor Sie die Metriken interpretieren, die sich auf Geldbeträge beziehen (MAE, RMSE), stellen Sie sicher, dass die Vorhersagen und `y_test` in der ursprünglichen Skala sind.

**VII. Interpretation, Schlussfolgerung und Dokumentation**

25. **Ergebnisse interpretieren:**
    * Welches Modell liefert die besten Ergebnisse gemäß den Metriken?
    * Welche Merkmale sind laut Lasso-Koeffizienten oder Merkmalswichtigkeiten der Bäume am relevantesten?
    * Stimmen die Ergebnisse mit den Erkenntnissen aus der EDA überein?
26. **Schlussfolgerungen ziehen:** Beantworten Sie die ursprüngliche Fragestellung.
27. **Nächste Schritte vorschlagen:** Was könnte man noch tun, um das Modell zu verbessern oder die Analyse zu erweitern?
28. **Jupyter Notebook strukturieren und finalisieren:** Mit Markdown-Texten, Erklärungen und sauberen Code-Zellen.