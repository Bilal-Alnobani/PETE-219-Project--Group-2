######### Poro-perm code goes here: #########

### importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import anderson
from scipy.stats import kstest

pp = pd.read_csv('poro_perm_data.csv')


##### cleaning the poro_perm data
#### removing negative values
data_clean = (pp["Porosity (%)"] > 0) & (pp["Permeability (mD)"] > 0)
pp = pp[data_clean]
#### removing nans
pp = pp.dropna()
#### removing the extra apostrophie in Facies
last_col = pp.columns[-1]
pp[last_col] = pp[last_col].str.replace("'","")


############################### Porosity scatter plot
por_plot = pp.plot("Depth (ft)", "Porosity (%)", kind = 'scatter')
plt.title('Porosity scatter plot')
plt.show()

############################# Permeability scatter plot
per_plot = pp.plot("Depth (ft)", "Permeability (mD)", kind = 'scatter')
plt.title('Permeabiliy scatter plot')
plt.show()

### Test for skewness for the porosity histogram
por_skew = skew(pp["Porosity (%)"], axis =0, bias = True)
### Test for kurtosis for the porosity histogram
por_kur = kurtosis(pp["Porosity (%)"], axis =0, bias = True)

### Test for skewness for the permeability histogram
per_skew = skew(pp["Permeability (mD)"], axis =0, bias = True)
### Test for kurtosis for the permeability histogram
per_kur = kurtosis(pp["Permeability (mD)"], axis =0, bias = True)


################################ Porosity histogram
poro_plot = plt.hist(pp["Porosity (%)"], 10)
plt.xlabel("Porosity (%)")
plt.ylabel("n")
plt.title("Histogram for Porosity")
plt.text(30,29, 'Skewness: ' +str(np.round(por_skew,3)),fontsize = 12)
plt.text(30,27, 'Kurtosis: ' +str(np.round(por_kur,3)),fontsize = 12)
plt.show()

################################# Permeability histogram
perm_plot = plt.hist(pp["Permeability (mD)"], 10)
plt.xlabel("Permeability (mD)")
plt.ylabel("n")
plt.title("Histogram for Permeabiliy")
plt.text(250,25, 'Skewness: ' +str(np.round(per_skew,3)),fontsize = 12)
plt.text(250,23, 'Kurtosis: ' +str(np.round(per_kur,3)),fontsize = 12)
plt.show()


##### Anderson Darling test for porosity
result = anderson(pp['Porosity (%)'])
print('Statistic: %.3f' % result.statistic)
p = 0

for i in range(len(result.critical_values)):
    slevel, cvalues = result.significance_level[i], result.critical_values[i]
    if result.statistic < result.critical_values[i]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (slevel, cvalues))
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (slevel, cvalues))



##### Anderson Darling test for permeability
result = anderson(pp['Permeability (mD)'])
print('Statistic: %.3f' % result.statistic)
p = 0

for i in range(len(result.critical_values)):
    slevel, cvalues = result.significance_level[i], result.critical_values[i]
    if result.statistic < result.critical_values[i]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (slevel, cvalues))
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (slevel, cvalues))



##### Kolmogorov-Smirnov test for Porosity
result = kstest(pp['Porosity (%)'], 'norm')
print('KS Statistic: %.3f' % result.statistic)

alpha = 0.05
if result.pvalue > alpha:
    print('P-value > %.2f, data looks normal (fail to reject H0)' % alpha)
else:
    print('P-value <= %.2f, data does not look normal (reject H0)' % alpha)


##### Kolmogorov-Smirnov test for Permeability
result = kstest(pp['Permeability (mD)'], 'norm')
print('KS Statistic: %.3f' % result.statistic)

alpha = 0.05
if result.pvalue > alpha:
    print('P-value > %.2f, data looks normal (fail to reject H0)' % alpha)
else:
    print('P-value <= %.2f, data does not look normal (reject H0)' % alpha)
