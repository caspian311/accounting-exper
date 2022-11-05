import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('./test.csv')
model = smf.ols(formula = 'TotalCost ~ Rentals', data = df).fit()
print(model.summary())

img = sm.graphics.plot_partregress('TotalCost', 'Rentals', ['Rentals'], data=df)
img.savefig('test.png')
