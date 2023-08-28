import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from scipy.stats import mode
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from keras.models import load_model
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load your models and data
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load your disease prediction models
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)

data = pd.read_csv('/content/Training (2).csv')
X = data.drop('prognosis', axis=1)  # Symptoms
y = data['prognosis']


# Encode disease labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

x_encoded = label_encoder.fit_transform(X)

final_svm_model.fit(X, y_encoded)
final_nb_model.fit(X, y_encoded)



# Define your functions
lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):  # Define the bow function
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

#Returns list with possible intents and its probability
def predict_class(sentence, model):
    p = bow(sentence, words)  # Use the bow function
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

#Response is chosen based on the predicted and intent present is matching
def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            result = np.random.choice(intent['responses'])
            break
    return result


#Convert symptoms into binary list marks 1 if a symptom is present
#the predictDisease function takes a list of symptoms as input, 
#converts them into a binary representation, predicts diseases using three different machine learning models, and combines the predictions to make a final prediction. 
#The final prediction is then converted back into the original disease label using the label encoder.
def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    input_data = [0] * len(X.columns)
    for symptom in symptoms:
        if symptom in X.columns:
            index = X.columns.get_loc(symptom)
            input_data[index] = 1
    input_data = np.array(input_data).reshape(1, -1)

    rf_prediction = final_rf_model.predict(input_data)[0]
    nb_prediction = final_nb_model.predict(input_data)[0]
    svm_prediction = final_svm_model.predict(input_data)[0]

    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]

    return label_encoder.inverse_transform([final_prediction])[0]

def chatbot_response(msg):
    if msg.startswith("predict_disease:"):
        symptoms = msg.replace("predict_disease:", "").strip()
        disease_prediction = predictDisease(symptoms)
        response = f"Based on the symptoms provided, the predicted disease is: {disease_prediction}"
    else:
        ints = predict_class(msg, model)
        res = get_response(ints, intents)
        response = res

    return response

print("DT: Hi there! I'm your chatbot. You can start chatting with me. Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("DT: Goodbye!")
        break

    response = chatbot_response(user_input)
    print("DT:", response)
