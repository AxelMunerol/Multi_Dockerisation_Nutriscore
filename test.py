# test/__init__.py
from flask import Blueprint, jsonify, request
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

test_api = Blueprint('test_api', __name__)

# Obtenir le chemin absolu du répertoire contenant le modèle
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model.pkl'))

# Charger le modèle
with open(model_dir, 'rb') as file:
    model = pickle.load(file)


def score_to_grade(score):
    if score <= -1:
        return 'A'
    elif 0 <= score <= 2:
        return 'B'
    elif 3 <= score <= 10:
        return 'C'
    elif 11 <= score <= 18:
        return 'D'
    else:
        return 'E'


@test_api.route('/test/predict', methods=['POST'])
def predict_test():
    try:
        # Récupérer les données JSON
        data = request.get_json()

        # Vérifier que toutes les données nécessaires sont présentes
        required_fields = [
            'energy_kcal', 'fat', 'saturated_fat', 'sugars',
            'fiber', 'proteins', 'salt',
            'fruits_vegetables_nuts_estimate', 'category'
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'status': 'error'
                }), 400

        # Initialiser le vecteur de caractéristiques
        feature_vector = np.zeros(47)

        # Remplir les caractéristiques nutritionnelles
        feature_vector[0] = float(data['energy_kcal'])
        feature_vector[1] = float(data['fat'])
        feature_vector[2] = float(data['saturated_fat'])
        feature_vector[3] = float(data['sugars'])
        feature_vector[4] = float(data['fiber'])
        feature_vector[5] = float(data['proteins'])
        feature_vector[6] = float(data['salt'])
        feature_vector[7] = float(data['fruits_vegetables_nuts_estimate'])

        # Mapping des catégories
        category_mapping = {
            "Appetizers": 8,
            "Artificially_sweetened_beverages": 9,
            "Biscuits_and_cakes": 10,
            "Bread": 11,
            "Breakfast_cereals": 12,
            "Cereals": 13,
            "Cheese": 14,
            "Chocolate_products": 15,
            "Dairy_desserts": 16,
            "Dressings_and_sauces": 17,
            "Dried_fruits": 18,
            "Eggs": 19,
            "Fats": 20,
            "Fish_and_seafood": 21,
            "Fruit_juices": 22,
            "Fruit_nectars": 23,
            "Fruits": 24,
            "Ice_cream": 25,
            "Legumes": 26,
            "Meat": 27,
            "Milk_and_yogurt": 28,
            "Nuts": 29,
            "Offals": 30,
            "One_dish_meals": 31,
            "Pastries": 32,
            "Pizza_pies_and_quiches": 33,
            "Plant_based_milk_substitutes": 34,
            "Potatoes": 35,
            "Processed_meat": 36,
            "Salty_and_fatty_products": 37,
            "Sandwiches": 38,
            "Soups": 39,
            "Sweetened_beverages": 40,
            "Sweets": 41,
            "Teas_and_herbal_teas_and_coffees": 42,
            "Unsweetened_beverages": 43,
            "Vegetables": 44,
            "Waters_and_flavored_waters": 45,
            "unknown": 46
        }

        # Définir la catégorie
        category = data['category']
        if category in category_mapping:
            feature_vector[category_mapping[category]] = 1
        else:
            return jsonify({
                'error': f'Invalid category: {category}',
                'status': 'error'
            }), 400

        # Reshape pour la prédiction
        feature_vector = feature_vector.reshape(1, -1)

        # Normaliser avec le scaler
        scaler = MinMaxScaler()
        feature_vector_scaled = scaler.fit_transform(feature_vector)

        # Faire la prédiction
        score_pred = model.predict(feature_vector_scaled)[0]
        grade_pred = score_to_grade(round(score_pred))

        # Retourner la prédiction
        return jsonify({
            'prediction': {
                'score': round(float(score_pred), 2),
                'grade': grade_pred
            },
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500