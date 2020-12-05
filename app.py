from flask import Flask, render_template, request, jsonify, url_for, redirect, flash
from utils import *
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.secret_key = "secret"

mail = ''
feedback_array = []

model = None
tfidf_vect = None
email_df = None
filename = None
train = None

@app.route('/', methods = ['GET', 'POST'])
def index():
    fb = request.form.get('feedback')
    print(fb)
    if fb == 'Clean': 
        feedback_array.append([mail,0])
    if fb == 'Spam': 
        feedback_array.append([mail,1])
    if model == None:
        return render_template('index.html', valid=False)
    else:
        # print(type(model['report']))
        report = model['report'].split(' ')
        report = [s.replace("\n","") for s in report if s]
        ctr = 0
        print ('Pre-report:', report)
        for i in report:
            try:
                if float(i) <= 1:
                    report[ctr] = float(i) * 100
                ctr += 1
            except:
                ctr += 1
                pass
        print('Post-report:',report)
        return render_template('index.html', size=email_df.shape[0], train=int((1-train) * 100), acc=round(float(model['accuracy'][1]), 2), report=report, filename = filename)

@app.route('/results', methods = ['POST'])
def results():
    global mail
    message = request.form.get('message')
    mail = message
    if len(message) == 0:
        flash(u'Enter email text', 'error')
        return redirect('/')
    else:
        df = convert_text([mail], [0])
        predict_X = tfidf_vect.transform(df['clean_text'])

        result = int(model['model'].predict(predict_X)[0])
        prob = model['model'].predict_proba(predict_X)[0][result]

        
        return render_template('results.html', content=message, result=result, prob=round(prob*100))

@app.route('/feedback')
def feedback():
    return render_template('feedback.html', fb_array=feedback_array)

@app.route('/retrain')
def retrain():
    global feedback_array
    global email_df
    global model
    global filename
    global tfidf_vect
    global train
    if request.args.get('size') is None:
        return redirect('/')
    else:
        if len(feedback_array) == 0:
            # print('empty')
            flash(u'List not populated yet!', 'error')
            return redirect('/feedback')
        else:
            df = convert_text([text[0] for text in feedback_array], [text[1] for text in feedback_array])
            email_df = email_df.append(df)
            file_n = retrain_model(email_df, float(request.args.get('size'))/10)
            feedback_array = []
            with open(file_n, 'rb') as f:
                model = pickle.load(f)
                email_df = model['dataset']
                tfidf_vect = model['vector']
                train = model['training']
                filename = file_n
            flash("Retrained model (" + file_n + ") saved and loaded!")
            return redirect('/')

@app.route('/upload', methods=['POST'])
def upload_file():
    global model
    global tfidf_vect
    global email_df
    global filename
    global train
    file = request.files['file']
    filename = file.filename
    if file:
        if file.filename.split('.')[-1] == 'pkl':
            model = pickle.load(file)
            email_df = model['dataset']
            tfidf_vect = model['vector']
            train = model['training']
    return redirect("/")

if __name__ == '__main__':
	app.run(host='0.0.0.0', port = 5000, debug = True)