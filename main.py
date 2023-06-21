import time
from problog.program import PrologString
from problog import get_evaluatable
from problog.logic import Term
from problog.learning import lfi
import pickle
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from patient import Patient
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
CORS(app)


def process_row(row, index, condition, term_string, evidence_list, final_model, is_last=False):
    value = float(row[index]) if row[index] != '' else 0
    is_normal = condition(value)
    evidence_list.append((Term(term_string), is_normal, None))
    if is_last:
        final_model += term_string + "." if is_normal else "\+" + term_string + "."
    else:
        final_model += term_string + "," if is_normal else "\+" + term_string + ","
    return final_model


def process_person_attribute(attribute, condition, term_string, evidence_list, cast_func=float, default_value=0):
    value = cast_func(attribute) if attribute != '' else default_value
    is_condition_met = condition(value)
    evidence_list.append((Term(term_string), is_condition_met, None))
    return evidence_list


def transformDatasetToModel(dataset):
    global set_liver
    global evidence
    global trained_model

    for index, row in dataset.iterrows():
        evidence_list = []
        if int(row[10]) == 1:
            final_model = "t(_)::liver_disease:-"
        else:
            final_model = "t(_)::healthy:-"

        # Gender column
        is_male = row[1].lower() == 'male'
        evidence_list.append((Term("male"), is_male, None))
        final_model += "male," if is_male else "\+male,"

        # Define the data
        conditions = [
            {"index": 0, "condition": lambda x: x < 50, "term_string": "young"},
            {"index": 2, "condition": lambda x: x < 1, "term_string": "normalBilirubin"},
            {"index": 3, "condition": lambda x: x < 0.3, "term_string": "normalDirectBilirubin"},
            {"index": 4, "condition": lambda x: x < 129, "term_string": "normalAlkalinePhosphotase"},
            {"index": 5, "condition": lambda x: x < 41, "term_string": "normalAlamine"},
            {"index": 6, "condition": lambda x: x < 40, "term_string": "normalAspartate"},
            {"index": 7, "condition": lambda x: 6 <= x <= 8.3, "term_string": "normalTotalProteins"},
            {"index": 8, "condition": lambda x: 3.5 <= x <= 5, "term_string": "normalAlbumin"},
            {"index": 9, "condition": lambda x: 0.8 <= x <= 2, "term_string": "normalAGRatio"}
        ]

        for i, cond in enumerate(conditions):
            is_last = i == len(conditions) - 1
            final_model = process_row(row, cond["index"], cond["condition"], cond["term_string"], evidence_list,
                                      final_model, is_last)

        final_model = final_model.replace(",.", ".")
        set_liver.add(final_model)
        evidence.append(evidence_list)


def submitPatient(person):
    global trained_model
    patient_evidence = []

    attributes = [
        {"attr": person.gender, "condition": lambda x: x.lower() == 'male', "term_string": "male", "cast_func": str,
         "default_value": False},
        {"attr": person.age, "condition": lambda x: x < 50 and x is not None, "term_string": "young", "cast_func": int},
        {"attr": person.total_bilirubin, "condition": lambda x: x < 1, "term_string": "normalBilirubin"},
        {"attr": person.direct_bilirubin, "condition": lambda x: x < 0.3, "term_string": "normalDirectBilirubin"},
        {"attr": person.alkaline_phosphotase, "condition": lambda x: x < 129,
         "term_string": "normalAlkalinePhosphotase"},
        {"attr": person.alamine_aminotransferase, "condition": lambda x: x < 41, "term_string": "normalAlamine"},
        {"attr": person.aspartate_aminotransferase, "condition": lambda x: x < 40, "term_string": "normalAspartate"},
        {"attr": person.total_proteins, "condition": lambda x: 6 <= x <= 8.3, "term_string": "normalTotalProteins"},
        {"attr": person.albumin, "condition": lambda x: 3.5 <= x <= 5, "term_string": "normalAlbumin"},
        {"attr": person.albumin_and_globulin_ratio, "condition": lambda x: 0.8 <= x <= 2,
         "term_string": "normalAGRatio"}
    ]

    for a in attributes:
        patient_evidence = process_person_attribute(a["attr"], a["condition"], a["term_string"], patient_evidence,
                                                    a.get("cast_func", float), a.get("default_value", 0))

    # Creating the problog model with evidences
    patient_data = trained_model
    for model_evidence in patient_evidence:
        patient_data += "\nevidence({}, {}).".format(model_evidence[0], model_evidence[1])
    patient_data += "\nquery(liver_disease).\nquery(healthy)."

    # Evaluate the new model with the evidences with Problog
    p_usermodel = PrologString(patient_data)
    result = get_evaluatable().create_from(p_usermodel, propagate_evidence=True).evaluate()

    counter = 0
    for query, value in result.items():
        if counter == 0:
            prob_message = "Probability of liver disease: " + format(value, ".4f") + "\n"
            counter = counter + 1
        else:
            prob_message = prob_message + "Probability to be healthy: " + format(value, ".4f") + "\n"
            counter = 0
    print(prob_message)
    return prob_message


def getProbabilities(person):
    prediction = submitPatient(person)
    # Now we extract the float values from the string
    lines = prediction.split("\n")
    liver_disease_prob = float(lines[0].split(":")[1].strip())
    healthy_prob = float(lines[1].split(":")[1].strip())

    return 1 if liver_disease_prob > healthy_prob else 2


def getDataClass(row):
    return int(row[10])


model_path = os.path.join(os.getcwd(), 'trained_model.pkl')

if os.path.exists(model_path):
    print("Found the trained model! Load in the system!")
    # Load the trained model from the file
    with open('trained_model.pkl', 'rb') as f:
        trained_model = pickle.load(f)

else:
    print("There is no trained model, training now!")
    # First I create a set and a list to save the csv and input user data
    set_liver = set()
    evidence = list()

    # Load the dataset
    data = pd.read_csv('indian_liver_patient.csv')
    # Split the dataset into training and testing
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    # Function that reads from csv and creates the training model
    transformDatasetToModel(train_data)

    term_list = list(set_liver)
    term_list.sort()

    # Creating the Learning Model
    terms = ['male', 'young', 'normalBilirubin', 'normalDirectBilirubin', 'normalAlkalinePhosphotase',
             'normalAlamine', 'normalAspartate', 'normalTotalProteins', 'normalAlbumin', 'normalAGRatio']
    model = "".join(f"t(_)::{term}.\n" for term in terms)

    for y in range(len(term_list)):
        if y != (len(term_list) - 1):
            model = model + term_list[y] + "\n"
        else:
            model = model + term_list[y]

    # Evaluate the learning model
    score, weights, atoms, iteration, lfi_problem = lfi.run_lfi(PrologString(model), evidence)
    trained_model = lfi_problem.get_model()

    # Save the untrained model to a file
    with open('untrained_model.pl', 'w') as f:
        f.write(model)
    print("Untrained Model created")

    # Save the trained model to a file
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(trained_model, f)
    print("Trained Model created")

    # Convert test_data into a list of Patient objects
    test_patients = [Patient(row[1], row[0], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]) for row
                     in
                     test_data.itertuples(index=False)]

    # Get the actual and predicted classifications
    actual_classifications = [getDataClass(row) for row in test_data.itertuples(index=False)]
    predicted_classifications = [getProbabilities(patient) for patient in test_patients]

    # Calculate the metrics
    acc = accuracy_score(actual_classifications, predicted_classifications)
    prec = precision_score(actual_classifications, predicted_classifications, average='macro')
    rec = recall_score(actual_classifications, predicted_classifications, average='macro')
    f1 = f1_score(actual_classifications, predicted_classifications, average='macro')
    cm = confusion_matrix(actual_classifications, predicted_classifications)
    # Write the metrics to a file
    with open('metrics.txt', 'w') as f:
        f.write(f'Accuracy: {acc}\n')
        f.write(f'Precision: {prec}\n')
        f.write(f'Recall: {rec}\n')
        f.write(f'F1 Score: {f1}\n')

        # Compute and write the confusion matrix
        f.write(f'Confusion Matrix: \n{cm}\n')

    # Metrics
    metrics = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1
    }

    # Bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'purple', 'orange', 'green'])
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.xlabel('Metrics')
    plt.ylim([0, 1])
    plt.show()

    # Confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

@app.route('/liver-disease-prediction', methods=['POST'])
def predict_liver_disease():
    jsonData = request.get_json(force=True)  # forcing the interpretation of request data as JSON

    # Create a patient object with the provided data
    patient = Patient(
        jsonData['gender'],
        int(jsonData['age']),
        float(jsonData['total_bilirubin']),
        float(jsonData['direct_bilirubin']),
        float(jsonData['alkaline_phosphotase']),
        float(jsonData['alamine_aminotransferase']),
        float(jsonData['aspartate_aminotransferase']),
        float(jsonData['total_proteins']),
        float(jsonData['albumin']),
        float(jsonData['albumin_and_globulin_ratio'])
    )

    # Call the function to get disease prediction
    prediction = submitPatient(patient)

    # Return the prediction result as JSON
    return jsonify({
        'prediction': prediction
    })


if __name__ == '__main__':
    app.run(port=5000, debug=True)
