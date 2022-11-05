import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices

df = sm.datasets.get_rdataset("Guerry", "HistData").data
vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
df = df[vars]
df = df.dropna()

y, X = dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type='dataframe')

print(y[:3])
print(X[:3])

mod = sm.OLS(y, X)
res = mod.fit()
print(res.summary())

img = sm.graphics.plot_partregress('Lottery', 'Wealth', ['Region', 'Literacy'], data=df, obs_labels=False)
print(type(img))
print(img)
img.savefig('test.png')
