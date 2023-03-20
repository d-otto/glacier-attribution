import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import statsmodels.api as sm

#%%

folders = {'BA': Path(r'C:\sandbox\glacier-attribution\data\external\mb\argentiere\ablation'),
           'BW': Path(r'C:\sandbox\glacier-attribution\data\external\mb\argentiere\accumulation')}
concat = []
for balance_code, folder in folders.items():
    ps = folder.iterdir()
    for p in ps:
        df = pd.read_csv(p, header=0)
        df['BALANCE_CODE'] = balance_code
        concat.append(df)
df = pd.concat(concat, ignore_index=True)
df.to_csv(Path(r'C:\sandbox\glacier-attribution\data\external\mb\argentiere\argentiere_point_mb.csv'))

#%%

d = df.groupby('year_end').mean()
fig, ax = plt.subplots(1,1)
ax.plot(d.index, d.annual_smb)
fig.show()

#%%

X = df.loc[:, ['altitude', 'year_end']]
X['year_end'] = X['year_end'] - X['year_end'].min()
X = sm.add_constant(X)
y = df['annual_smb']

ols = sm.OLS(y, X).fit()
ols.summary()

#%%

sns.set_style('ticks')
fig, ax = plt.subplots(1, 1, figsize=(8,6))
sns.scatterplot(df, x='altitude', y='annual_smb', hue='year_end', palette='viridis', ax=ax)
sns.regplot(x='altitude', y='annual_smb', color='black', data=df, scatter=False, ci=None, x_partial='year_end', ax=ax)
ax.set_title('Argentiere Annual MB (1975-2019)', loc='left')
ax.set_title('b = -22.92 + -0.32 * yr + 0.0076 * meters', loc='right')
ax.grid(which='both', axis='both')
ax.set_axisbelow(True)
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.xaxis.set_minor_locator(MultipleLocator(50))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.25))
plt.tight_layout()
plt.savefig('argentiere_point_smb.png')
fig.show()
