import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

from src.utils.design_utils import triangle_multi_mask
from pandas.plotting import scatter_matrix

plt.style.use('default')


def main():
    # Intro - Read and edit Dataset
    df = pd.read_csv('insurance.csv', na_values='?')
    df.dropna(inplace=True)
    df.info()
    df.head()

    # correlation matrix
    corr_matrix = df.corr(numeric_only=True)
    multi_mask = triangle_multi_mask(corr_matrix, 0)  # Erstellt Maske für Dreieck Darstellung
    sns.heatmap(corr_matrix, annot=True, cmap="flare", mask=multi_mask)

    #plt.show()

    # Scatterplots
    fig, ax = plt.subplots(2,2, figsize=(15, 15))

    # commented out for performance
    #sns.scatterplot(data=df, x=df['age'], y=df['charges'], alpha=0.5, hue=df['sex'], ax=ax[0,0])
    #sns.scatterplot(data=df, x=df['age'], y=df['bmi'], alpha=0.5, hue=df['sex'], ax=ax[0,1])
    #sns.scatterplot(data=df, x=df['age'], y=df['bmi'], alpha=0.5, hue=df['children'], ax=ax[1,0])
    #sns.scatterplot(data=df, x=df['age'], y=df['bmi'], alpha=0.5, hue=df['region'], ax=ax[1,1])

    run = False  # performance schalter
    if run:
        # Analysis: effect of number of children
        for i in pd.unique(df['children']):  # Iterate through all numbers of children (unique values)

            # Challenge: Get only affected rows of dataframe

            # Filter das df auf den aktuellen Wert i
            subset = df[df['children'] == i]

            # Scatterplot erstellen
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"Children: {i}, n={len(subset)})")
            ax[0].scatter(x = subset['bmi'], y=subset['charges'], alpha=0.5)
            ax[0].set_xlabel('BMI')
            ax[0].set_ylabel('charges')
            ax[1].scatter(x = subset['age'], y=subset['charges'], alpha=0.5)
            ax[1].set_xlabel('AGE')
            ax[1].set_ylabel('charges')
            ax[2].scatter(x = subset['age'], y=subset['bmi'], alpha=0.5)
            ax[2].set_xlabel('AGE')
            ax[2].set_ylabel('bmi')
            plt.show()


    # These plots show no correlations based on amount of children.
    # Amount of children has likely no effect on result

    # Analysis:
    # age-charges showed the highest correlation value in the corr_matrix -> OLS
    X = sm.add_constant(df['age'])
    y = df['charges']

    model = sm.OLS(y,X)
    estimate = model.fit()
    print(estimate.summary())
    # The linear regression for age/cahrges shows clear signs of correlation:
    # p value extremely low (almost 0)
    # beta_0 at 3165 (intercept)
    # beta_1 = 257 (slope)

    # plot regression line onto data
    fig, ax = plt.subplots(figsize=(10,10))
    plt.scatter(x=df['age'], y=df['charges'])
    plt.plot(df['age'], estimate.fittedvalues, color='r')
    plt.show()


if __name__ == "__main__":
    main()