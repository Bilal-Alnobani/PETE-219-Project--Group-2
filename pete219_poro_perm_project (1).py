### importing libraries
import pylab
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import anderson
from scipy.stats import kstest
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN



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

###################### PP plot for porosity
stats.probplot(pp["Porosity (%)"], dist = 'norm', plot = pylab)
pylab.show()

####################### PP plot for permeability
stats.probplot(pp["Permeability (mD)"], dist = 'norm', plot = pylab)
pylab.show()

################################## k means clustering with porosity scatter plot
x1 = pp["Porosity (%)"].to_numpy().reshape(-1,1)
y1 = pp["Permeability (mD)"].to_numpy().reshape(-1,1)

#plot
plt.scatter(x1,y1)
plt.xlabel("Poro (%)")
plt.ylabel("Permea (mD)")
plt.title("scatter plot using kmeans clustering")

x1y1 = np.hstack((x1,y1))
km = KMeans(
    n_clusters = 3, init = 'random',
    n_init = 10, max_iter = 300,
    tol = 1e-04, random_state =0
)
y_km = km.fit_predict(x1y1)

# plot clusters
plt.scatter(
    x1y1[y_km ==0,0], x1y1[y_km == 0,1],
    s = 100, c = 'green',
    marker = 'x', edgecolor = 'black',
    label = 'cluster 1')

plt.scatter(
    x1y1[y_km ==1,0], x1y1[y_km == 1,1],
    s = 100, c = 'purple',
    marker = 'x', edgecolor = 'black',
    label = 'cluster 2')

plt.scatter(
    x1y1[y_km ==2,0], x1y1[y_km == 2,1],
    s = 100, c = 'orange',
    marker = 'x', edgecolor = 'black',
    label = 'cluster 3')
plt.legend()

## scoring how well the kmeans clustering worked against true values
from sklearn.metrics import silhouette_score
sil = silhouette_score(x1y1, y_km)
acc = sil*100

## converting string labels
labels, uniques = pd.factorize(pp.iloc[:,-1])
plt.scatter(pp["Porosity (%)"], pp["Permeability (mD)"], c = labels, cmap = "turbo")
plt.xlabel("Porosity (%)")
plt.ylabel("Permeability (mD)")
plt.text(1,250,"Accuracy: " + str(np.round(acc,3)), fontsize = 8)
plt.show()

########################################## DBScan 

xx = pp[["Porosity (%)", "Permeability (mD)"]].values
dbscan = DBSCAN(eps = 15, min_samples = 10)
cluster = dbscan.fit_predict(xx)

#plot
plt.scatter(pp["Porosity (%)"], pp["Permeability (mD)"], c = cluster, cmap = "viridis", marker = "x")

######################################### actual distribution of facies

#giving each type of facies its color identity
fac = {"channel" : 'green', "overbanks" : "orange", "crevasse splay" : "purple"}
# scatter plot
for facies, color in fac.items():
    sub = pp[pp["Facies"] == facies]
    plt.scatter(sub["Porosity (%)"], sub["Permeability (mD)"], label = facies, color = color, alpha = 0.7)
plt.legend()
plt.xlabel("Porosity (%)")
plt.ylabel("Permeability (mD)")
plt.title("Actual distribution of facies")
plt.show()

############### Regression plot

por_x = pp.loc[:, "Porosity (%)"].to_numpy().reshape(-1,1)
per_y = pp.loc[:, "Permeability (mD)"].to_numpy().reshape(-1,1)
plt.scatter(por_x,per_y)
plt.xlabel('porosity (%)')
plt.ylabel('permeability (mD)')
plt.title("Porosity vs Permeability regression plot")
model = LinearRegression()
model.fit(por_x,per_y)
r_sq = model.score(por_x,per_y)
y_pred = model.predict(por_x)
plt.plot(por_x,y_pred, color = "orange")
plt.text(2,300, "R^2 value: " + str(np.round(r_sq,3)), fontsize = 12)
plt.show()

######## Regression plot for channel facies

ch = pp.loc[:,"Facies"] == "channel"
ch = pp[ch]
por_ch = ch.loc[:, "Porosity (%)"].to_numpy().reshape(-1,1)
per_ch = ch.loc[:, "Permeability (mD)"].to_numpy().reshape(-1,1)
plt.scatter(por_ch,per_ch)
plt.xlabel('porosity (%)')
plt.ylabel('permeability (mD)')
plt.title('channel')
model = LinearRegression()
model.fit(por_ch,per_ch)
r_sq_ch = model.score(por_ch, per_ch)
y_pred_ch = model.predict(por_ch)
plt.plot(por_ch, y_pred_ch, color = "green")
plt.text(13,300, "R^2 value: " + str(np.round(r_sq_ch,3)), fontsize =10)
plt.show()


######## Regression plot for overbank facies

ob =pp.loc[:,"Facies"] == "overbanks"
ob = pp[ob]
por_ob = ob.loc[:, "Porosity (%)"].to_numpy().reshape(-1,1)
per_ob = ob.loc[:, "Permeability (mD)"].to_numpy().reshape(-1,1)
plt.scatter(por_ob,per_ob)
plt.xlabel('porosity (%)')
plt.ylabel('permeability (mD)')
plt.title('overbanks')
model = LinearRegression()
model.fit(por_ob,per_ob)
r_sq_ob = model.score(por_ob, per_ob)
y_pred_ob = model.predict(por_ob)
plt.plot(por_ob, y_pred_ob, color = "green")
plt.text(20,30, "R^2 value: " + str(np.round(r_sq_ob,3)), fontsize =10)
plt.show()

######## Regression plot for crevasse splay facies

cs =pp.loc[:,"Facies"] == "crevasse splay"
cs = pp[cs]
por_cs = cs.loc[:, "Porosity (%)"].to_numpy().reshape(-1,1)
per_cs = cs.loc[:, "Permeability (mD)"].to_numpy().reshape(-1,1)
plt.scatter(por_cs,per_cs)
plt.xlabel('porosity %')
plt.ylabel('permeability (mD)')
plt.title('crevasse splay')
model = LinearRegression()
model.fit(por_cs,per_cs)
r_sq_cs = model.score(por_cs, per_cs)
y_pred_cs = model.predict(por_cs)
plt.plot(por_cs, y_pred_cs, color = "red")
plt.text(27,100, "R^2 value: " + str(np.round(r_sq_cs,3)), fontsize =10)
plt.show()


