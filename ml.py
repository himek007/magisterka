import numpy as np
import pandas as pd
import time
df = pd.read_csv('MagDane.csv')
df.sample(5)
X = df.drop(columns=['outcome'])
y = df['outcome']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=2)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
start = time.time()
rf.fit(X_train.values ,y_train)
stop = time.time()
print(f"Czas treningu losowego lasu: {stop - start}s")
y_pred = rf.predict(X_test.values)
print(accuracy_score(y_test,y_pred))
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
start = time.time()
dt.fit(X_train.values, y_train)
stop = time.time()
print(f"Czas treningu drzewo decyzyjne: {stop - start}s")
y_pred2 = dt.predict(X_test.values)
print(accuracy_score(y_test, y_pred2))
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
neu = make_pipeline(StandardScaler(),  MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,20), random_state=1)) 
start = time.time()
neu.fit(X_train.values, y_train.values)
stop = time.time()
print(f"Czas treningu sieci neuronowych: {stop - start}s")
y_pred3 = neu.predict(X_test.values)
print(accuracy_score(y_test, y_pred3))
import pickle 
pickle.dump(rf,open('model.pkl','wb'))
pickle.dump(dt,open('model2.pkl','wb'))
pickle.dump(neu,open('model3.pkl','wb'))
from flask import Flask,request,jsonify
import numpy as np
import pickle
model = pickle.load(open('model.pkl','rb'))
model2 =pickle.load(open('model2.pkl','rb'))
model3 =pickle.load(open('model3.pkl','rb'))
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
    result2 = model2.predict(input_query)[0]
    result3 = model3.predict(input_query)[0]
    return jsonify({'outcome':str(result), 'outcome2':str(result2), 'outcome3':str(result3)})
if __name__ == '__main__':
    app.run(debug=True)