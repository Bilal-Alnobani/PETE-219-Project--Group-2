##### Well logs code goes here: #####
import lasio as lasio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import missingno as msno
import seaborn as sns

### Importing and showing the well info ###
las=lasio.read('D:/Desktop/1051661071.las')
well_df= las.df()
well_df.reset_index(inplace=True)
well_df.rename(columns={'DEPT':'DEPTH'}, inplace=True)

### Droping data that won't be used: ###
well_df=well_df[['DEPTH','CNPOR','RHOB','GR','RILD','RILM','DPOR']]

### Cleaning Data ###
n=9256-9170
well_df.drop(well_df.tail(n).index,inplace = True)

well_df['CNPOR'][well_df['CNPOR'] < 0] = np.nan
well_df['GR'][well_df['GR'] > 500] = np.nan
well_df['GR'][well_df['GR'] < 0] = np.nan
well_df['RILD'][well_df['RILD'] > 200] = np.nan
well_df['RILM'][well_df['RILM'] > 250] = np.nan
well_df['RHOB'][well_df['RHOB'] < 0] = np.nan
well_df['RHOB'][well_df['RHOB'] > 5] = np.nan

well_df=well_df.dropna()
msno.matrix(well_df)

### plotting well logs ###
ax1 = plt.subplot2grid((1,3), (0,0), rowspan=1, colspan = 1) 
ax2 = plt.subplot2grid((1,3), (0,1), rowspan=1, colspan = 1)
ax3 = plt.subplot2grid((1,3), (0,2), rowspan=1, colspan = 1)
ax4 = ax3.twiny()

ax1.plot("GR", "DEPTH", data = well_df, color = "green", lw=1)
ax1.set_xlabel("Gamma") 
ax1.set_xlim(0, 200) 
ax1.set_ylim(2900, 2000) 
ax1.grid() 

ax2.plot("RILD", "DEPTH", data = well_df, color = "black", lw=1)
ax2.set_xlabel("Deep Resistivity")
ax2.set_xlim(0.2, 2000)
ax2.semilogx()
ax2.set_ylim(2900, 2000)
ax2.grid()

ax3.plot("RHOB", "DEPTH", data = well_df, color = "red", lw=1)
ax3.set_xlabel("Density (red line)")
ax3.set_xlim(0.5, 2.95)
ax3.set_ylim(2900, 2000)
ax3.grid()

ax4.plot("CNPOR", "DEPTH", data = well_df, color = "blue", lw=1)
ax4.set_xlabel("Netron Porosity (blue line)")
ax4.set_xlim(45, -15)
ax4.set_ylim(2900, 2000)


### Plotting a Heat map ###
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(well_df.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Log Properties Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
plt.show()

#histograms 
### Plotting Histograms ###
## RHOB histogram
plt.hist(well_df['RHOB'], bins=20, color='red', alpha=0.5, edgecolor='black')
plt.ylabel("frequency", fontsize=14)
plt.xlabel("RHOB", fontsize= 14)
plt.xlim(0,4)

mean_RHOB = well_df['RHOB'].mean()
p5_RHOB= well_df['RHOB'].quantile(0.05)
p95_RHOB = well_df['RHOB'].quantile(0.95)
median_RHOB = well_df['RHOB'].median()
plt.axvline(mean_RHOB,color='blue', label='mean')
plt.axvline(p5_RHOB,color='green', label='5th Percentile')
plt.axvline(p95_RHOB,color='purple', label='95th Percentile')
plt.axvline(median_RHOB,color='red', label='Median')
plt.legend()
plt.show()

## GR histogram
plt.hist(well_df['GR'], bins=50, color='blue', alpha=0.5, edgecolor='black')
plt.ylabel("frequency", fontsize=14)
plt.xlabel("GR", fontsize= 14)
plt.xlim(0,200)

mean_GR = well_df['GR'].mean()
p5_GR= well_df['GR'].quantile(0.05)
p95_GR = well_df['GR'].quantile(0.95)
median_GR = well_df['GR'].median()
plt.axvline(mean_GR,color='blue', label='mean')
plt.axvline(p5_GR,color='green', label='5th Percentile')
plt.axvline(p95_GR,color='purple', label='95th Percentile')
plt.axvline(median_GR,color='red', label='Median')
plt.legend()
plt.show()

## CNPOR Histogram
plt.hist(well_df['CNPOR'], bins=70, color='green', alpha=0.5, edgecolor='black')
plt.ylabel("frequency", fontsize=14)
plt.xlabel("CNPOR", fontsize= 14)
plt.xlim(0,80)

mean_CNPOR = well_df['CNPOR'].mean()
p5_CNPOR= well_df['CNPOR'].quantile(0.05)
p95_CNPOR = well_df['CNPOR'].quantile(0.95)
median_CNPOR = well_df['CNPOR'].median()
plt.axvline(mean_CNPOR,color='blue', label='mean')
plt.axvline(p5_CNPOR,color='black', label='5th Percentile')
plt.axvline(p95_CNPOR,color='purple', label='95th Percentile')
plt.axvline(median_CNPOR,color='red', label='Median')
plt.legend()
plt.show()

## RILD Histogram
plt.hist(well_df['RILD'], bins=70, color='green', alpha=0.5, edgecolor='black')
plt.ylabel("frequency", fontsize=14)
plt.xlabel("RILD", fontsize= 14)
plt.xlim(0,100)

mean_RILD = well_df['RILD'].mean()
p5_RILD= well_df['RILD'].quantile(0.05)
p95_RILD = well_df['RILD'].quantile(0.95)
median_RILD = well_df['RILD'].median()
plt.axvline(mean_RILD,color='blue', label='mean')
plt.axvline(p5_RILD,color='black', label='5th Percentile')
plt.axvline(p95_RILD,color='purple', label='95th Percentile')
plt.axvline(median_RILD,color='red', label='Median')
plt.legend()
plt.show()

