
import numpy as np
import pandas as pd
df = pd.read_csv('c:/Users/Joachim/python/MagDane.csv')
df.sample(5)
X = df.drop(columns=['outcome'])
y = df['outcome']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
#train the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train.values ,y_train)
y_pred = rf.predict(X_test.values)
print(accuracy_score(y_test,y_pred))
#save the model in pickle format
import pickle 
pickle.dump(rf,open('model.pkl','wb'))
from flask import Flask,request,jsonify
import numpy as np
import pickle
model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict',methods=['POST'])
def predict():
    plec = request.form.get('plec')
    wiek = request.form.get('wiek')
    tygodnie = request.form.get('tygodnie')
    input_query = np.array([[plec,wiek,tygodnie]])
    result = model.predict(input_query)[0]
    return jsonify({'outcome':str(result)})
    result = ""
if __name__ == '__main__':
    app.run(debug=True)