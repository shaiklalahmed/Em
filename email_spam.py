from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)


with open('Email_spam_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('Email_spam_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('email_spam.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        email_text = request.form['text']
        
       
        vectorized_text = vectorizer.transform([email_text])
        
        
        prediction = model.predict(vectorized_text)
        
        
        result = 'spam' if prediction[0] else 'not spam'
        return render_template('email_spam.html', prediction_text=f'This email is {result}.')
    
    except Exception as e:
        return render_template('email_spam.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
