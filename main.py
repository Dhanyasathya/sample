import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
from datetime import datetime as dt
import pandas_gbq
import pandas as pd

# %%bigquery graph
# select * from arvind_dataset.arvind_datasets

# global graph
# project_id = "arvind-machinelearning-dhanya"
# graph = pandas_gbq.read_gbq('SELECT * FROM predicted_value.df', project_id=project_id
#                           )

#from regression import final_value

#global final_value
app = Flask(__name__)
model = pickle.load(open('classifier.pkl','rb'))


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    store_id = final_features[0][0]
    date_value = str(final_features[0][1])
    df_foo = pd.DataFrame([date_value],columns=['string_val'])
    date_final = pd.to_datetime(df_foo["string_val"])
    df_foo["Year"] = date_final.dt.year
    df_foo["Month"] = date_final.dt.month
    df_foo["Day"] = date_final.dt.day
    df_foo["Week"] = date_final.dt.week
    print(df_foo.head())
    pred = [x for x in df_foo.columns if x  in ['Year' , 'Month' , 'Day' , 'Week']]

    prediction = model.predict(df_foo[pred])
    output_val = np.exp(prediction)
    print(prediction[0])

    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="Predicted Sales values is {}".format(output_val[0]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

@app.route('/plot',methods=['GET','POST'])

def plot():
        
        img = BytesIO()
        dtf = pd.DataFrame.from_records(graph,columns=['BUSINESS_DATE','predict_sales_real', 'act_sales_real'])
        dtf.plot(x='BUSINESS_DATE', y=['predict_sales_real', 'act_sales_real'], figsize=(10,5), grid=True)
        fig = plt.gcf()
        fig.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template('plot.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)