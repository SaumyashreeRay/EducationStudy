#!/usr/bin/env python 
import numpy as np 
import pandas as pd 
test = pd.read_csv('test.csv', delimiter = ',') 
f1 = pd.read_csv('primaryenrollment.csv', delimiter = ',') 
f2 = pd.read_csv('secondaryenrollment.csv', delimiter = ',') 
f3 = pd.read_csv('tertiaryenrollment.csv', delimiter = ',') 
temp = f1.merge(f2, on=["Region/Country/Area", "Year"], how='outer') 
enrollment = temp.merge(f3, on=["Region/Country/Area", "Year"], how='outer') 
enrollment.to_csv("enrollment.csv",index=False) 
e1 = pd.read_csv('primary_expenditure.csv', delimiter = ',') 
e2 = pd.read_csv('secondary_expenditure.csv', delimiter = ',') 
e3 = pd.read_csv('tertiary_expenditure.csv', delimiter = ',') 
e4 = pd.read_csv('gdp_expenditure.csv', delimiter = ',') 
temp = e1.merge(e2, on=["Region/Country/Area", "Year"], how='outer') 
temp2 = temp.merge(e3, on=["Region/Country/Area", "Year"], how='outer') 
expenditures = temp2.merge(e4, on=["Region/Country/Area", "Year"], how='outer') 
expenditures.to_csv("expenditures.csv",index=False) 
a = pd.read_csv('gdp.csv', delimiter = ',',encoding='latin-1') 
b = pd.read_csv('gdp_percapita.csv', delimiter = ',', encoding='latin-1') 
c = pd.read_csv('population.csv', delimiter = ',', encoding='latin-1') 
m = expenditures.merge(a, on=["Region/Country/Area", "Year"], how='outer') 
m2 = m.merge(b, on=["Region/Country/Area", "Year"], how='outer') 
m3 = m2.merge(c, on=["Region/Country/Area", "Year"], how='outer') 
everything = m3.merge(enrollment, on=["Region/Country/Area", "Year"], how='outer') 
everything.to_csv("alldata.csv",index=False) 