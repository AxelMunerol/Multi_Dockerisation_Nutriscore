from flask import Blueprint, render_template, request, jsonify
from graphs import create_graph
import numpy as np
import pandas as pd
import pickle
import json

# Chargement du modèle
with open('/model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Chargement du scaler
with open('/model/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

main = Blueprint('main', __name__)


# Fonction de conversion du score au grade NutriScore
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


# Route pour afficher le tableau de bord
@main.route('/')
def dashboard():
    graph = create_graph()
    return render_template('index.html', graph=graph)


# Route pour prédire depuis une API
@main.route('/predict_test', methods=['POST'])
def predict_test():
    data = request.get_json()
    required_fields = ['energy_kcal', 'fat', 'saturated_fat', 'sugars',
                       'fiber', 'proteins', 'salt',
                       'fruits_vegetables_nuts_estimate', 'category']

    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}', 'status': 'error'}), 400

    feature_vector = np.zeros(47)
    feature_vector[:8] = [float(data[field]) for field in required_fields[:-1]]

    category_mapping = {
        "pnns_groups_2_Appetizers": 8,
        "pnns_groups_2_Artificially_sweetened_beverages": 9,
        "pnns_groups_2_Biscuits_and_cakes": 10,
        "pnns_groups_2_Bread": 11,
        "pnns_groups_2_Breakfast_cereals": 12,
        "pnns_groups_2_Cereals": 13,
        "pnns_groups_2_Cheese": 14,
        "pnns_groups_2_Chocolate_products": 15,
        "pnns_groups_2_Dairy_desserts": 16,
        "pnns_groups_2_Dressings_and_sauces": 17,
        "pnns_groups_2_Dried_fruits": 18,
        "pnns_groups_2_Eggs": 19,
        "pnns_groups_2_Fats": 20,
        "pnns_groups_2_Fish_and_seafood": 21,
        "pnns_groups_2_Fruit_juices": 22,
        "pnns_groups_2_Fruit_nectars": 23,
        "pnns_groups_2_Fruits": 24,
        "pnns_groups_2_Ice_cream": 25,
        "pnns_groups_2_Legumes": 26,
        "pnns_groups_2_Meat": 27,
        "pnns_groups_2_Milk_and_yogurt": 28,
        "pnns_groups_2_Nuts": 29,
        "pnns_groups_2_Offals": 30,
        "pnns_groups_2_One_dish_meals": 31,
        "pnns_groups_2_Pastries": 32,
        "pnns_groups_2_Pizza_pies_and_quiches": 33,
        "pnns_groups_2_Plant_based_milk_substitutes": 34,
        "pnns_groups_2_Potatoes": 35,
        "pnns_groups_2_Processed_meat": 36,
        "pnns_groups_2_Salty_and_fatty_products": 37,
        "pnns_groups_2_Sandwiches": 38,
        "pnns_groups_2_Soups": 39,
        "pnns_groups_2_Sweetened_beverages": 40,
        "pnns_groups_2_Sweets": 41,
        "pnns_groups_2_Teas_and_herbal_teas_and_coffees": 42,
        "pnns_groups_2_Unsweetened_beverages": 43,
        "pnns_groups_2_Vegetables": 44,
        "pnns_groups_2_Waters_and_flavored_waters": 45, "unknown": 46
    }

    category = data['category']
    if category in category_mapping:
        feature_vector[category_mapping[category]] = 1
    else:
        return jsonify({'error': f'Invalid category: {category}', 'status': 'error'}), 400


    feature_vector_scaled = scaler.transform([feature_vector])
    score_pred = model.predict(feature_vector_scaled)[0]
    grade_pred = score_to_grade(round(score_pred))

    return jsonify({
        'prediction': {
            'score': round(float(score_pred), 2),
            'grade': grade_pred
        },
        'status': 'success'
    })


# Route pour afficher la page de prédiction
@main.route('/predict')
def predict():
    return render_template('predict.html')


# Route pour gérer les résultats de la prédiction via formulaire
@main.route('/results', methods=['POST'])
def results():


    def parse_input(value, num):
        mediane = [250, 6.4, 1.5, 3.7, 1.9, 6.4, 0.48, 2.1]
        return mediane[num] if value == "" else float(value)

        # Récupérer et traiter les valeurs des champs du formulaire
    energy_kcal = parse_input(request.form.get('energy-kcal'), 0)
    fat = parse_input(request.form.get('fat'), 1)
    saturated_fat = parse_input(request.form.get('saturated-fat'), 2)
    sugars = parse_input(request.form.get('sugars'), 3)
    fiber = parse_input(request.form.get('fiber'), 4)
    proteins = parse_input(request.form.get('proteins'), 5)
    salt = parse_input(request.form.get('salt'), 6)
    fruits_vegetables_nuts_estimate = parse_input(
    request.form.get('fruits-vegetables-nuts-estimate-from-ingredients'), 7)
    selected_name = request.form.get('selected_name')

    # Initialiser les données avec les valeurs de base
    new_data = np.array(
        [[energy_kcal, fat, saturated_fat, sugars, fiber, proteins, salt, fruits_vegetables_nuts_estimate] + [0] * 39])

    # Activer l’indice de la catégorie
    category_mapping = {
        "pnns_groups_2_Appetizers": 8,
        "pnns_groups_2_Artificially_sweetened_beverages": 9,
        "pnns_groups_2_Biscuits_and_cakes": 10,
        "pnns_groups_2_Bread": 11,
        "pnns_groups_2_Breakfast_cereals": 12,
        "pnns_groups_2_Cereals": 13,
        "pnns_groups_2_Cheese": 14,
        "pnns_groups_2_Chocolate_products": 15,
        "pnns_groups_2_Dairy_desserts": 16,
        "pnns_groups_2_Dressings_and_sauces": 17,
        "pnns_groups_2_Dried_fruits": 18,
        "pnns_groups_2_Eggs": 19,
        "pnns_groups_2_Fats": 20,
        "pnns_groups_2_Fish_and_seafood": 21,
        "pnns_groups_2_Fruit_juices": 22,
        "pnns_groups_2_Fruit_nectars": 23,
        "pnns_groups_2_Fruits": 24,
        "pnns_groups_2_Ice_cream": 25,
        "pnns_groups_2_Legumes": 26,
        "pnns_groups_2_Meat": 27,
        "pnns_groups_2_Milk_and_yogurt": 28,
        "pnns_groups_2_Nuts": 29,
        "pnns_groups_2_Offals": 30,
        "pnns_groups_2_One_dish_meals": 31,
        "pnns_groups_2_Pastries": 32,
        "pnns_groups_2_Pizza_pies_and_quiches": 33,
        "pnns_groups_2_Plant_based_milk_substitutes": 34,
        "pnns_groups_2_Potatoes": 35,
        "pnns_groups_2_Processed_meat": 36,
        "pnns_groups_2_Salty_and_fatty_products": 37,
        "pnns_groups_2_Sandwiches": 38,
        "pnns_groups_2_Soups": 39,
        "pnns_groups_2_Sweetened_beverages": 40,
        "pnns_groups_2_Sweets": 41,
        "pnns_groups_2_Teas_and_herbal_teas_and_coffees": 42,
        "pnns_groups_2_Unsweetened_beverages": 43,
        "pnns_groups_2_Vegetables": 44,
        "pnns_groups_2_Waters_and_flavored_waters": 45,
        "pnns_groups_2_unknown": 46
    }
    if selected_name in category_mapping:
        new_data[0, category_mapping[selected_name]] = 1


    # Normalisation
    new_data_scaled = scaler.transform(new_data)
    y_new_pred = model.predict(new_data_scaled)
    score_new_pred = round(y_new_pred[0])
    y_new_pred_grade = score_to_grade(score_new_pred)

    # Enregistrer la recherche et le résultat
    with open('/data/search_log.txt', 'a') as log_file:
        log_file.write(f"Requete :  {new_data} Résultat: {y_new_pred_grade}\n\n")

    return render_template('results.html', y_new_pred=y_new_pred_grade, score_new_pred=score_new_pred)
