import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('./test.csv')
model = smf.ols(formula = 'AdjustedTotalCost ~ Rentals + D11 + D12 + D13 + D14 + D15 + D16 + D17 + HDD + CDD', data = df).fit()
print(model.summary())

img = sm.graphics.plot_partregress('AdjustedTotalCost', 'Rentals', ['Rentals', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'HDD', 'CDD'], data=df)
img.savefig('test.png')
