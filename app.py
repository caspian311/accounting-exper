import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

my_sample = pd.read_csv('./test.csv')
vars = ['Rentals', 'TotalCost']
df = my_sample[vars]
print(df)

my_sample['year_f'] = pd.Categorical(my_sample.Year)
my_model = smf.ols(formula = 'TotalCost ~ Rentals', data = my_sample).fit()
print(my_model.summary())

img = sm.graphics.plot_partregress('TotalCost', 'Rentals', ['Rentals'], data=df)
img.savefig('test.png')
