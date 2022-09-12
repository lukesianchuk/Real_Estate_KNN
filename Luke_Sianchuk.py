####### KNN Paris Real Estate Model ###########
### DATA SOURCES:
# https://www.data.gouv.fr/en/datasets/r/90a98de0-f562-4328-aa16-fe0dd1dca60f
# https://cadastre.data.gouv.fr/data/etalab-cadastre/2021-04-01/shp/departements/75/cadastre-75-parcelles-shp.zip


import pandas as pd
import geopandas


# Importing sale data as pandas dataframe
saledata = pd.read_csv("valeursfoncieres-2020.txt",sep="|")
# Importing geographic data using geopandas
shp = geopandas.read_file("cadastre-75-parcelles-shp/parcelles.shp")

# Calculating centroid coordinates of each parcel for use in model
shp["x"] = shp.centroid.map(lambda p: p.x)
shp["y"] = shp.centroid.map(lambda p: p.y)

# Subsetting data to include only Paris property
mydata = saledata.loc[saledata['Voie'] == 'PARIS']

# Removing unnecessary columns
mydata = mydata.loc[:,["Valeur fonciere", "No voie", "Code voie", "Voie", "Code postal", "Commune","Code departement", "Code commune", "Section", "No plan"]]


# Merging geogrpahic data with sale data
# Saledata uses "Code commune" to reference cadastral data
# this matches with numero? field in geographic data

# Convert object type to integer
shp["numero_int"] = shp["numero"].astype(str).astype(int)
# Merge data
merged = pd.merge(mydata, shp, left_on="Code commune", right_on="numero_int", how="inner")

# Take out only the columns we will use in the model
training_data = merged.loc[:,["Valeur fonciere", "x", "y"]]
# Rename column to ENglish
training_data = training_data.rename(columns={'Valeur fonciere': 'Value'})
# Remove commas and convert to int
training_data['Value'] = training_data['Value'].str.replace(',00','')
training_data["Value"] = training_data["Value"].astype(str).astype(int)



##### Setting up the model ##########


# Create data matrices
y = training_data.loc[:,"Value"]
X = training_data.loc[:,["x","y"]]

# Separate into train/test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)

# Scale features to diminish influence of outlying values
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Fit only on X_train
scaler.fit(X_train)

# Scale both X_train and X_test
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Fitting multpiple models to determine best k value
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
errors = []
for i in range(1,30):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    mae = mean_absolute_error(y_test, pred_i)
    errors.append(mae)
    
# Plot to visualize mean aboslute errors
import matplotlib.pyplot as plt 
plt.plot(range(1, 30), errors)



############## Fitting our best model ###############


regressor = KNeighborsRegressor(n_neighbors=4)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)