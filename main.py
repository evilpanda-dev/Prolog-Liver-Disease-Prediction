import time
from problog.program import PrologString
from problog import get_evaluatable
from problog.logic import Term
from problog.learning import lfi
import pickle
import os
from flask import Flask, request, jsonify
# from flask_cors import CORS
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from patient import Patient


# app = Flask(__name__)
# CORS(app)

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

        # Age column
        age_user = int(row[0]) if row[0] != '' else 0
        is_young = age_user < 50
        evidence_list.append((Term("young"), is_young, None))
        final_model += "young," if is_young else "\+young,"

        # Total_Bilirubin
        total_bilirubin = float(row[2]) if row[2] != '' else 0
        is_normal_bilirubin = total_bilirubin < 1
        evidence_list.append((Term("normalBilirubin"), is_normal_bilirubin, None))
        final_model += "normalBilirubin," if is_normal_bilirubin else "\+normalBilirubin,"

        # Direct_Bilirubin
        direct_bilirubin = float(row[3]) if row[3] != '' else 0
        is_normal_direct_bilirubin = direct_bilirubin < 0.3
        evidence_list.append((Term("normalDirectBilirubin"), is_normal_direct_bilirubin, None))
        final_model += "normalDirectBilirubin," if is_normal_direct_bilirubin else "\+normalDirectBilirubin,"

        # Alkaline_Phosphotase
        alkaline_phosphotase = float(row[4]) if row[4] != '' else 0
        is_normal_alkaline_phosphotase = alkaline_phosphotase < 129
        evidence_list.append((Term("normalAlkalinePhosphotase"), is_normal_alkaline_phosphotase, None))
        final_model += "normalAlkalinePhosphotase," if is_normal_alkaline_phosphotase else "\+normalAlkalinePhosphotase,"

        # Alamine_Aminotransferase
        alamine_aminotransferase = float(row[5]) if row[5] != '' else 0
        is_normal_alamine = alamine_aminotransferase < 41
        evidence_list.append((Term("normalAlamine"), is_normal_alamine, None))
        final_model += "normalAlamine," if is_normal_alamine else "\+normalAlamine,"

        # Aspartate_Aminotransferase
        aspartate_aminotransferase = float(row[6]) if row[6] != '' else 0
        is_normal_aspartate = aspartate_aminotransferase < 40
        evidence_list.append((Term("normalAspartate"), is_normal_aspartate, None))
        final_model += "normalAspartate," if is_normal_aspartate else "\+normalAspartate,"

        # Total_Proteins
        total_proteins = float(row[7]) if row[7] != '' else 0
        is_normal_total_proteins = 6 <= total_proteins <= 8.3
        evidence_list.append((Term("normalTotalProteins"), is_normal_total_proteins, None))
        final_model += "normalTotalProteins," if is_normal_total_proteins else "\+normalTotalProteins,"

        # Albumin
        albumin = float(row[8]) if row[8] != '' else 0
        is_normal_albumin = 3.5 <= albumin <= 5
        evidence_list.append((Term("normalAlbumin"), is_normal_albumin, None))
        final_model += "normalAlbumin," if is_normal_albumin else "\+normalAlbumin,"

        # Albumin_and_Globulin_Ratio
        albumin_and_globulin_ratio = float(row[9]) if row[9] != '' else 0
        is_normal_ag_ratio = 0.8 <= albumin_and_globulin_ratio <= 2
        evidence_list.append((Term("normalAGRatio"), is_normal_ag_ratio, None))
        final_model += "normalAGRatio." if is_normal_ag_ratio else "\+normalAGRatio."

        final_model = final_model.replace(",.", ".")
        set_liver.add(final_model)
        evidence.append(evidence_list)


def submitPatient(person):
    global trained_model
    patient_evidence = []

    # Gender input
    gender_user = person.gender
    is_male = gender_user == 1 if gender_user is not None else False
    patient_evidence.append((Term("male"), is_male, None))

    # Age input
    age_user = int(person.age)
    is_young = age_user is not None and age_user < 50
    patient_evidence.append((Term("young"), is_young, None))

    # Total_Bilirubin
    total_bilirubin = float(person.total_bilirubin) if person.total_bilirubin != '' else 0
    is_normal_bilirubin = total_bilirubin < 1
    patient_evidence.append((Term("normalBilirubin"), is_normal_bilirubin, None))

    # Direct_Bilirubin
    direct_bilirubin = float(person.direct_bilirubin) if person.direct_bilirubin != '' else 0
    is_normal_direct_bilirubin = direct_bilirubin < 0.3
    patient_evidence.append((Term("normalDirectBilirubin"), is_normal_direct_bilirubin, None))

    # Alkaline_Phosphotase
    alkaline_phosphotase = float(person.alkaline_phosphotase) if person.alkaline_phosphotase != '' else 0
    is_normal_alkaline_phosphotase = alkaline_phosphotase < 129
    patient_evidence.append((Term("normalAlkalinePhosphotase"), is_normal_alkaline_phosphotase, None))

    # Alamine_Aminotransferase
    alamine_aminotransferase = float(person.alamine_aminotransferase) if person.alamine_aminotransferase != '' else 0
    is_normal_alamine = alamine_aminotransferase < 41
    patient_evidence.append((Term("normalAlamine"), is_normal_alamine, None))

    # Aspartate_Aminotransferase
    aspartate_aminotransferase = float(
        person.aspartate_aminotransferase) if person.aspartate_aminotransferase != '' else 0
    is_normal_aspartate = aspartate_aminotransferase < 40
    patient_evidence.append((Term("normalAspartate"), is_normal_aspartate, None))

    # Total_Proteins
    total_proteins = float(person.total_proteins) if person.total_proteins != '' else 0
    is_normal_total_proteins = 6 <= total_proteins <= 8.3
    patient_evidence.append((Term("normalTotalProteins"), is_normal_total_proteins, None))

    # Albumin
    albumin = float(person.albumin) if person.albumin != '' else 0
    is_normal_albumin = 3.5 <= albumin <= 5
    patient_evidence.append((Term("normalAlbumin"), is_normal_albumin, None))

    # Albumin_and_Globulin_Ratio
    albumin_and_globulin_ratio = float(
        person.albumin_and_globulin_ratio) if person.albumin_and_globulin_ratio != '' else 0
    is_normal_ag_ratio = 0.8 <= albumin_and_globulin_ratio <= 2
    patient_evidence.append((Term("normalAGRatio"), is_normal_ag_ratio, None))

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
    print("Trained model found, loading...")
    # Load the trained model from the file
    with open('trained_model.pkl', 'rb') as f:
        trained_model = pickle.load(f)

else:
    print("No trained model found, creating a new one...")
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
    model = """"""
    model = model + "t(_)::male.\n"
    model = model + "t(_)::young.\n"
    model = model + "t(_)::normalBilirubin.\n"
    model = model + "t(_)::normalDirectBilirubin.\n"
    model = model + "t(_)::normalAlkalinePhosphotase.\n"
    model = model + "t(_)::normalAlamine.\n"
    model = model + "t(_)::normalAspartate.\n"
    model = model + "t(_)::normalTotalProteins.\n"
    model = model + "t(_)::normalAlbumin.\n"
    model = model + "t(_)::normalAGRatio.\n"

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
    test_patients = [Patient(row[1], row[0], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]) for row in
                     test_data.itertuples(index=False)]

    # Get the actual and predicted classifications
    actual_classifications = [getDataClass(row) for row in test_data.itertuples(index=False)]
    predicted_classifications = [getProbabilities(patient) for patient in test_patients]

    # Calculate the metrics
    acc = accuracy_score(actual_classifications, predicted_classifications)
    prec = precision_score(actual_classifications, predicted_classifications, average='macro')
    rec = recall_score(actual_classifications, predicted_classifications, average='macro')
    f1 = f1_score(actual_classifications, predicted_classifications, average='macro')

    # Write the metrics to a file
    with open('metrics.txt', 'w') as f:
        f.write(f'Accuracy: {acc}\n')
        f.write(f'Precision: {prec}\n')
        f.write(f'Recall: {rec}\n')
        f.write(f'F1 Score: {f1}\n')

        # Compute and write the confusion matrix
        cm = confusion_matrix(actual_classifications, predicted_classifications)
        f.write(f'Confusion Matrix: \n{cm}\n')
