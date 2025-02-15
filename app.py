from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load dataset
file_path = 'new_car_dataset.csv'
df = pd.read_csv(file_path)

# Encode categorical variables
le_BRAND = LabelEncoder()
df['CAR_BRAND'] = le_BRAND.fit_transform(df['CAR_BRAND'])

le_FUEL = LabelEncoder()
df['FUEL_TYPE'] = le_FUEL.fit_transform(df['FUEL_TYPE'])

le_TRANSMISSION = LabelEncoder()
df['TRANSMISSION'] = le_TRANSMISSION.fit_transform(df['TRANSMISSION'])

x = df[['CAR_BRAND', 'FUEL_TYPE', 'TRANSMISSION', 'LAUNCH_YEAR']]
y = df['EX_SHOWROOM_PRICE']

# Train KNN model
knn = NearestNeighbors(n_neighbors=4, metric='euclidean')
knn.fit(x)

# Mappings for decoding
brand_mapping = {0:'BMW', 1:'FORD', 2:'HONDA', 3:'HYUNDAI', 4:'KIA', 5:'MG', 6:'MAHINDRA', 7:'MARUTI SUZUKI', 8:'MERCEDEZ BENZ', 9:'NISSAN', 10:'RENAULT', 11:'SKODA', 12:'TATA', 13:'TOYOTA'}
fuel_type_mapping = {0:'CNG', 1:'DIESEL', 2:'ELECTRIC', 3:'HYBRID', 4:'PETROL'}
transmission_mapping = {0:'AUTO', 1:'MANUAL'}

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        car_brand = int(request.form['car_brand'])
        fuel_type = int(request.form['fuel_type'])
        transmission = int(request.form['transmission'])
        min_price = int(request.form['min_price'])
        max_price = int(request.form['max_price'])

        sample_input = [[car_brand, fuel_type, transmission, 2024]]
        distances, indices = knn.kneighbors(sample_input)
        matching_records = df.iloc[indices[0]]
        
        filtered_records = matching_records[
            (matching_records['EX_SHOWROOM_PRICE'] >= min_price) & 
            (matching_records['EX_SHOWROOM_PRICE'] <= max_price)
        ]
        
        #MAPPING FILTERED RECORDS
        filtered_records['CAR_BRAND'] = filtered_records['CAR_BRAND'].map(brand_mapping)
        filtered_records['FUEL_TYPE'] = filtered_records['FUEL_TYPE'].map(fuel_type_mapping)
        filtered_records['TRANSMISSION'] = filtered_records['TRANSMISSION'].map(transmission_mapping)

        #DISPLAY RECORDS TABLE
        results = filtered_records.to_dict(orient='records')

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
