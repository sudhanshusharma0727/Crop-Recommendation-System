from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle
import requests

# importing model
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

# creating flask app
app = Flask(__name__)

# render home page
@app.route('/')
def home():
    title = 'Crop - Home'
    return render_template('index.html',title=title)

@ app.route('/crop_recommend')
def crop_recommend():
    title = 'Crop - Crop Recommendation'
    return render_template('crop.html', title=title)

@app.route("/crop-predict", methods=['POST'])
def crop_prediction():
    title = 'Crop - Crop predicted'
    try:
        if request.method == 'POST':
            N = int(request.form['nitrogen'])
            P = int(request.form['phosphorous'])
            K = int(request.form['pottasium'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            temp = float(request.form['temperature'])
            humidity = float(request.form['humidity'])

            if(N > 150 or P > 150 or K > 200 or temp > 50 or humidity > 100 or ph > 14 or rainfall > 300):
                result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
                image_url = 'img.jpg'
                return render_template('try_again.html', title=title)
                
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = model.predict(final_features)

        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            image_url = f'static\{crop}.jpg'
            print("crop",crop)
            return render_template('crop-result.html',image_url=image_url,prediction=crop, title=title)

        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
            image_url = 'img.jpg'
            return render_template('try_again.html', title=title)

    except Exception as e:
        # Log the exception or print it for debugging purposes
        print(e)
        return "An error occurred while processing the request."

# python main
if __name__ == "__main__":
    app.run()