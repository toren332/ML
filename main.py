import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import geopy.distance
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
import csv
import json
import random

c_coords = (55.75583333, 37.61777778)
STATIONS = []
# PARAMS = ('lat', 'lng', 'rooms', 'square', 'floor', 'center_distance', 'metro_dist', 'metro_quantity')
PARAMS = ('lat', 'lng', 'square')
NEW_ITEM = {
    'item_coords': (55.7470686, 37.85781589999999),
    'rooms': 3,
    'square': 40,
    'floor': 11,
}

with open('metro.json') as json_file:
    data = json.load(json_file)
    for i in data['lines']:
        for j in i['stations']:
            STATIONS.append(j)


def get_metro_quantity_and_dist(item_coords):
    q = 0
    distances = []
    for station in STATIONS:
        distance = geopy.distance.geodesic((station['lat'], station['lng']), item_coords).km
        distances.append(distance)
        if distance < 1.5:
            q += 1
    return q, min(distances)


def create_new(item, params=PARAMS):
    item_coords = item['item_coords']
    center_distance = geopy.distance.geodesic(c_coords, item_coords).km
    quantity, metro_dist = get_metro_quantity_and_dist(item_coords)
    lat, lng, rooms, square, floor, center_distance, metro_dist, metro_quantity = (item_coords[0], item_coords[1], item['rooms'], item['square'], item['floor'], center_distance, metro_dist, quantity)
    d = {
        'lat': lat,
        'lng': lng,
        'rooms': rooms,
        'square': square,
        'floor': floor,
        'center_distance': center_distance,
        'metro_dist': metro_dist,
        'metro_quantity': metro_quantity,
    }
    _ = []
    for i in params:
        _.append(d[i])
    return _


def play_with_params(example, params=PARAMS):
    new_example_0 = []
    new_example = []
    feature_names = list(params)
    for i in example:
        lat, lng, rooms, square, floor, center_distance, metro_dist, metro_quantity = i
        new_example_0.append({
            'lat': lat,
            'lng': lng,
            'rooms': rooms,
            'square': square,
            'floor': floor,
            'center_distance': center_distance,
            'metro_dist': metro_dist,
            'metro_quantity': metro_quantity,
        })
    for j in new_example_0:
        _ = []
        for i in params:
            _.append(j[i])
        new_example.append(_)
    return feature_names, np.array(new_example, dtype=np.float)

# with open('items.csv', newline='') as csvfile:
#     good_data = []
#     examples = []
#     target = []
#     target_names = []
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         item_coords = (row['lat'], row['lng'])
#         good_data.append({
#             'item_coords': item_coords,
#             'rooms': row['rooms'],
#             'square': row['square'],
#             'floor': row['floor'],
#             'target': int(float(row['total_price']) // 2000000)
#         })
#
# random.shuffle(good_data)
# for data in good_data[0:]:
#     item_coords = data['item_coords']
#     center_distance = geopy.distance.geodesic(c_coords, item_coords).km
#     quantity, metro_dist = get_metro_quantity_and_dist(item_coords)
#     if center_distance < 45 and float(data['square']) < 150 and float(data['floor']) < 60 and metro_dist<6:
#         examples.append([data['item_coords'][0], data['item_coords'][1], data['rooms'], data['square'], data['floor'], center_distance, metro_dist, quantity])
#         target.append(data['target'])
#
#
# with open('examples1.pickle', 'wb') as f:
#     pickle.dump(examples, f)
# with open('target1.pickle', 'wb') as f:
#     pickle.dump(target, f)


with open('examples1.pickle', 'rb') as f:
    examples = pickle.load(f)
with open('target1.pickle', 'rb') as f:
    target = pickle.load(f)

feature_names, examples = play_with_params(examples)
X_train, X_test, y_train, y_test = train_test_split(examples, target, random_state=0)
dataframe = pd.DataFrame(X_train, columns=feature_names)
grr = scatter_matrix(dataframe, c=y_train, figsize=(40, 40), marker='O', )
rfc = RandomForestClassifier(n_jobs=-1)


rfc.fit(X_train, y_train)

X_new = np.array([create_new(NEW_ITEM)])
prediction = rfc.predict(X_new)
print('Прогноз стоимости:')
print(str(prediction[0] * 2) + '-' + str(prediction[0] * 2 + 2) + ' млн руб.')
y_pred = rfc.predict(X_test)
print("Правильность на тестовом наборе: {:.2f}".format(np.mean(y_pred == y_test)))


plt.show()
