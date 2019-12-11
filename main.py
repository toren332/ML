import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestRegressor
import statistics


PARAMS = ('lat', 'lng', 'square')
NEW_ITEM = {
    'item_coords': (55.686083, 37.599891999999954),
    'square': 35,
}


def create_new(item):
    return [item['item_coords'][0], item['item_coords'][1], item['square']]


def get_example(example):
    new_example = []
    for i in example:
        lat, lng, rooms, square, floor, center_distance, metro_dist, metro_quantity = i
        new_example.append([lat, lng, square])
    return PARAMS, np.array(new_example, dtype=np.float)


with open('examples_regression.pickle', 'rb') as f:
    examples = pickle.load(f)
with open('target_regression.pickle', 'rb') as f:
    target = pickle.load(f)

feature_names, examples = get_example(examples)
all_prices = []
for j in range(5):
    med_prices = []

    for i in range(10):

        X_train, X_test, y_train, y_test = train_test_split(examples, target, random_state=0)
        dataframe = pd.DataFrame(X_train, columns=feature_names)
        # grr = scatter_matrix(dataframe, c=y_train, figsize=(30, 30), marker='O', )
        rfr = RandomForestRegressor(n_jobs=-1)
        rfr.fit(X_train, y_train)
        X_new = np.array([create_new(NEW_ITEM)])
        prediction = rfr.predict(X_new)
        med_prices.append(prediction[0])
        all_prices.append(prediction[0])
    # print(f'Среднее:\n{int(statistics.median(med_prices))} руб.')
print(f'Среднее:\n{int(statistics.median(all_prices))} руб.')
print(f'Колебания:\n{int(min(all_prices))} - {int(max(all_prices))} руб.')

# print(f"Правильность на тестовом наборе: {rfr.score(X_test, y_test):.2f}")
#
#
# plt.show()
