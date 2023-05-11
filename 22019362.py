# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:13:20 2023

@author: 91798
"""
#Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from sklearn import cluster
import err_ranges as err


#Reading input data files
def read_data(file_name):
        """
        Function to read datafile
    
        """

        df = pd.read_excel(file_name)
    
        df_change = df.drop(columns=["Series Name","Country Name","Country Code"])
        df_change = df_change.replace(np.nan,0)
        df_transpose = np.transpose(df_change)
        #print(df_transposed)
        df_transpose = df_transpose.reset_index()
        df_transpose = df_transpose.rename(columns={"index":"year", 0:"UK", 1:"France"})
    
        df_transpose = df_transpose.iloc[1:]
        df_transpose = df_transpose.dropna()
        #print(df_transposed)
    
        df_transpose["year"] = df_transpose["year"].str[:4]
        df_transpose["year"] = pd.to_numeric(df_transpose["year"])
        df_transpose["France"] = pd.to_numeric(df_transpose["France"])
        df_transpose["UK"] = pd.to_numeric(df_transpose["UK"])
        print(df_transpose)
        return df_change, df_transpose

def curve_function(t, scale, growth):
    """
    
    Function to calculate curve fit values
    
    """

    c = scale * np.exp(growth * (t-1960))
    return c

#Calling the file read function
df_co2, df_co2t = read_data("co2_emission.xlsx")
df_gdp, df_gdpt = read_data("gdp.xlsx")
df_renew, df_renewt = read_data("Renewable.xlsx")


#Doing curve fit
param, cov = opt.curve_fit(curve_function,df_co2t["year"],df_co2t["France"],p0=[4e8, 0.1])
sigma = np.sqrt(np.diag(cov))

#Error
low,up = err.err_ranges(df_co2t["year"],curve_function,param,sigma)
df_co2t["fit_value"] = curve_function(df_co2t["year"], * param)

#Plotting the co2 emission values for France
plt.figure()
plt.title("CO2 emissions (metric tons per capita) - France")
plt.plot(df_co2t["year"],df_co2t["France"],label="data")
plt.plot(df_co2t["year"],df_co2t["fit_value"],c="red",label="fit")
plt.fill_between(df_co2t["year"],low,up,alpha=0.5)
plt.legend()
plt.xlim(1990,2019)
plt.xlabel("Year")
plt.ylabel("CO2")
plt.savefig("Co2_France.png", dpi = 500, bbox_inches='tight')
plt.show()

#Curve ft for UK
param, cov = opt.curve_fit(curve_function,df_co2t["year"],df_co2t["UK"],p0=[4e8, 0.1])
sigma = np.sqrt(np.diag(cov))
print(*param)
lower,upper = err.err_ranges(df_co2t["year"],curve_function,param,sigma)
df_co2t["fit_value"] = curve_function(df_co2t["year"], * param)

#Plotting
plt.figure()
plt.title("UK CO2 emission prediction For 2030")
predict_year = np.arange(1980,2030)
predict_france = curve_function(predict_year,*param)
plt.plot(df_co2t["year"],df_co2t["UK"],label="data")
plt.plot(predict_year,predict_france,label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("CO2")
plt.savefig("Co2_UK_Predicted.png", dpi = 500, bbox_inches='tight')
plt.show()

#Total energy use for France
param, cov = opt.curve_fit(curve_function,df_renewt["year"],df_renewt["France"],p0=[4e8, 0.1])
sigma = np.sqrt(np.diag(cov))
print(*param)
lower,upper = err.err_ranges(df_renewt["year"],curve_function,param,sigma)

#Plotting
df_renewt["fit_value"] = curve_function(df_renewt["year"], * param)
plt.figure()
plt.title("Renewable energy use as a percentage of total energy - France")
plt.plot(df_renewt["year"],df_renewt["France"],label="data")
plt.plot(df_renewt["year"],df_renewt["fit_value"],c="red",label="fit")
plt.fill_between(df_renewt["year"],lower,upper,alpha=0.5)
plt.legend()
plt.xlim(1990,2019)
plt.xlabel("Year")
plt.ylabel("Renewable energy(% of total energy use)")
plt.savefig("Renewable_France.png", dpi = 500, bbox_inches='tight')
plt.show()


#Plotting the predicted values for France co2
plt.figure()
plt.title("France CO2 emission prediction")
predict_year = np.arange(1980,2030)
predict_france = curve_function(predict_year,*param)
plt.plot(df_co2t["year"],df_co2t["France"],label="data")
plt.plot(predict_year,predict_france,label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("CO2")
plt.savefig("Co2_France_Predicted.png", dpi = 500, bbox_inches='tight')
plt.show()

#Predicted values for UK total energy use
param, cov = opt.curve_fit(curve_function,df_renewt["year"],df_renewt["UK"],p0=[4e8, 0.1])
sigma = np.sqrt(np.diag(cov))
print(*param)
lower,upper = err.err_ranges(df_renewt["year"],curve_function,param,sigma)

#Plotting the predicted values for UK total energy use
df_renewt["fit_value"] = curve_function(df_renewt["year"], * param)
plt.figure()
plt.title("Renewable energy prediction - UK")
predict_year = np.arange(1980,2030)
predict_france = curve_function(predict_year,*param)
plt.plot(df_renewt["year"],df_renewt["UK"],label="data")
plt.plot(predict_year,predict_france,label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Renewable energy(% of total energy use)")
plt.savefig("Renewable_Prediction_UK.png", dpi = 500, bbox_inches='tight')
plt.show()

#Clustering
france = pd.DataFrame()
france["co2_emission"] = df_co2t["France"]
france["renewable_energy"] = df_renewt["France"]
kmean = cluster.KMeans(n_clusters=2).fit(france)
label = kmean.labels_
plt.scatter(france["co2_emission"],france["renewable_energy"],c=label,cmap="jet")
plt.title("co2 emission vs renewable enery usage -France")
c = kmean.cluster_centers_

#Plotting Scatter CO2 vs Renewable France
for t in range(2):
    xc,yc = c[t,:]
    plt.plot(xc,yc,"ok",markersize=8)
    plt.savefig("Scatter_UK_France_CO2.png", dpi = 500, bbox_inches='tight')
plt.figure()

#Plotting Scatter UK and France CO2 Emission 
df_co2t= df_co2t.iloc[:,1:3]
kmean = cluster.KMeans(n_clusters=2).fit(df_co2t)
label = kmean.labels_
plt.scatter(df_co2t["UK"],df_co2t["France"],c=label,cmap="jet")
plt.title("UK and France - CO2 Emission")
c = kmean.cluster_centers_
plt.savefig("Scatter_UK_France_CO2.png", dpi = 500, bbox_inches='tight')
plt.show()


#Plotting GDP per capita
plt.figure()
plt.plot(df_gdpt["year"], df_gdpt["France"])
plt.plot(df_gdpt["year"], df_gdpt["UK"])
plt.xlim(1991,2020)
plt.xlabel("Year")
plt.ylabel("GDP Per Capita")
plt.legend(['FRANCE','UK'])
plt.title("GDP per capita")
plt.savefig("GDP.png", dpi = 500, bbox_inches='tight')
plt.show()



