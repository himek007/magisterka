import numpy as np
import pandas as pd
import time
import customtkinter
global plec
global tygodnie
global wiek
df = pd.read_csv('MagDaneLicz.csv')
df.sample(5)
X = df.drop(columns=['outcome'])
y = df['outcome']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=2)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_leaf=1, min_samples_split=2)
start = time.time()
rf.fit(X_train.values, y_train)
stop = time.time()
czasLas = f"Czas treningu losowego lasu: {stop - start}s"
print(f"Czas treningu losowego lasu: {stop - start}s")
y_pred = rf.predict(X_test.values)
celLas=accuracy_score(y_test,y_pred)
print(accuracy_score(y_test,y_pred))
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=20, min_samples_leaf=4, min_samples_split=6, criterion="gini")
start = time.time()
dt.fit(X_train.values, y_train)
stop = time.time()
czasDrzewo=f"Czas treningu drzewo decyzyjne: {stop - start}s"
print(f"Czas treningu drzewo decyzyjne: {stop - start}s")
y_pred2 = dt.predict(X_test.values)
celDrzewo = accuracy_score(y_test, y_pred2)
print(accuracy_score(y_test, y_pred2))
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
neu = make_pipeline(StandardScaler(),  MLPClassifier(hidden_layer_sizes=(22, 49), solver='lbfgs', max_iter=148, activation='tanh', random_state=1))
start = time.time()
neu.fit(X_train.values, y_train.values)
stop = time.time()
czasNeuro=f"Czas treningu sieci neuronowych: {stop - start}s"
print(f"Czas treningu sieci neuronowych: {stop - start}s")
y_pred3 = neu.predict(X_test.values)
celNeuro = accuracy_score(y_test, y_pred3)
print(accuracy_score(y_test, y_pred3))
import pickle 

pickle.dump(rf,open('model.pkl','wb'))
pickle.dump(dt,open('model2.pkl','wb'))
pickle.dump(neu,open('model3.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
model2 =pickle.load(open('model2.pkl','rb'))
model3 =pickle.load(open('model3.pkl','rb'))
def number_to_text(number):
  mapping = {
    1: "Brak",
    2: "Wozek-kule",
    3: "Wozek-laska-foot up",
    4: "Balkonik",
    5: "Wozek-parapodium",
    6: "Wozek-Luska",
    7: "Wozek",
    8: "kule",
    9: "Stabilizator przedramienia",
    10: "Rurka traheo CA",
    11: "Stabilizator ST. Skokowego-foot-up-czwornog",
    12: "Stabilizator ST. Skokowego-foot-up",
    13: "Luska",
    14: "ST. ST kolanowgo L-stab. Sta kol P-Ĺuska na golen i stope L",
    15: "Proteza-kule",
    16: "Kule-balkonik",
    17: "kule-temblak",
    18: "Wozek-poduszka przeciwodl",
    19: "Wozek-podnosnik z funkcja pionizacji",
    20: "Wozek-czwornog-foot-up-stab stawu skokowego",
    21: "Wozek-balkonik",
    22: "Sznurowka",
    23: "Wozek-laska",
    24: "Kamizelka ortopedyczna (zamiast temblaka)",
    25: "Kamizelka-wozek",
    26: "Wozek-stabilizator ST biodrowego-kule",
    27: "Wozek-foot up-laska",
    28: "Wozek-hemiflex-poduszka",
    29: "Proteza tymczasowa",
    30: "Balkonik-Wozek-Luska aktywna na stope opaajaca-stab przeciw przeprosom ST kolanowych li p- stab ST skokwegi lip",
    31: "Laska jednopunktowa- b.z",
    32: "Wozek-Laska",
    33: "Kule-czwornog",
    34: "Wozek-czwornog",
    35: "Wozek-foot up",
    36: "Wozek-ortezy ST kol-balkonik",
    37: "Temblak-wozek-poduszka-hemiflex-podnosnik",
    38: "Wozek-foot-up-temblak",
    39: "Wozek-kule - laska jednopunktowa",
    40: "Wozek-stab ST skok-czwornog",
    41: "Stab ST skokowego-foot up",
    42: "Wozek-stab ST skokowych",
    43: "Wozek-poduszka-13kg",
    44: "Wozek wysoki-pionizator-poduszka-stab ST lokciowego",
    45: "Laska jednopunktowa-sznurowka",
    46: "Wozek-poduszka-balkonik-stab sty kolanowego l i p",
    47: "Wozek-poduszka-temblak-czwornog",
    48: "Wozek wysoki-wozek niski-czwornog",
    49: "Parapodium-poduszka-balkonik-Luska na kkd- b.z"
  }
  return mapping.get(number, "other")
def predict():
    plec = 0
    wiek = 0
    tygodnie = 0
    plec = set_plec()
    wiek = set_wiek()
    tygodnie = set_tygodnie()
    input_query = np.array([[plec,wiek,tygodnie]])
    result = model.predict(input_query)[0]
    result2 = model2.predict(input_query)[0]
    result3 = model3.predict(input_query)[0]
    result = number_to_text(result)
    result2 = number_to_text(result2)
    result3 = number_to_text(result3)
    czasLas1 = float(czasLas.split(":")[1].strip()[:-1])
    label_las_czas.configure(text = "Czas treningu: {:.5f}".format(czasLas1)+"s")
    czasDrzewo1 = float(czasDrzewo.split(":")[1].strip()[:-1])
    label_drzewo_czas.configure(text = "Czas treningu: {:.5f}".format(czasDrzewo1)+"s")
    czasNeuro1 = float(czasNeuro.split(":")[1].strip()[:-1])
    label_neuro_czas.configure(text = "Czas treningu: {:.5f}".format(czasNeuro1)+"s")
    label_las_cel.configure(text = "Celność: {:.2%}".format(celLas))
    label_drzewo_cel.configure(text = "Celność: {:.2%}".format(celDrzewo))
    label_neuro_cel.configure(text = "Celność: {:.2%}".format(celNeuro))
    label_las_wynik.configure(text ="Przewidziany wynik: " + result )
    label_drzewo_wynik.configure(text = "Przewidziany wynik: " + result2)
    label_neuro_wynik.configure(text ="Przewidziany wynik: " + result3)
customtkinter.set_appearance_mode("light")
customtkinter.set_default_color_theme("dark-blue")
root = customtkinter.CTk()
root.geometry("600x800")
frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=15, padx=15, fill="both", expand=True)
label = customtkinter.CTkLabel(master=frame, text="System wyboru przedmiotu zaopatrzenia medycznego po udarze", font=("Arial", 17))
label.pack(pady=8, padx=10)
plec1 = customtkinter.CTkOptionMenu(frame,values=["Mezczyzna", "Kobieta"], font=("Arial", 14))
plec1.pack(pady=8, padx=10)
plec1.set("Wybierz plec")
def set_plec():
    selected_option = plec1.get()
    if selected_option == "Mezczyzna":
        plec = 1
        return plec
    elif selected_option == "Kobieta":
        plec = 0
        return plec

wiek1 = customtkinter.CTkEntry(master= frame, placeholder_text="Wpisz wiek", font=("Arial", 14))
wiek1.pack(pady=8, padx=10)
def set_wiek():
    wiek = wiek1.get()
    return wiek

tygodnie1 = customtkinter.CTkEntry(master= frame, placeholder_text="Ile tygodni po udarze", font=("Arial", 13.25))
tygodnie1.pack(pady=8, padx=10)
def set_tygodnie():
    tygodnie=tygodnie1.get()
    return tygodnie
button = customtkinter.CTkButton(master=frame, text="Przewidywanie", command=predict, font=("Arial", 14))
button.pack(pady=8, padx=10)

label_las = customtkinter.CTkLabel(master=frame, text="Losowy Las", font=("Arial", 14))
label_las.pack(pady=8, padx=10)
label_las_czas = customtkinter.CTkLabel(master=frame, text="Czas treningu:", font=("Arial", 14))
label_las_czas.pack(pady=8, padx=10)
label_las_cel = customtkinter.CTkLabel(master=frame, text="Celność:", font=("Arial", 14))
label_las_cel.pack(pady=8, padx=10)
label_las_wynik = customtkinter.CTkLabel(master=frame, text="Wynik przewidywania:", font=("Arial", 14))
label_las_wynik.pack(pady=8, padx=10)

label_drzewo = customtkinter.CTkLabel(master=frame, text="Drzewo Decyzyjne(C&RT)", font=("Arial", 14))
label_drzewo.pack(pady=8, padx=10)
label_drzewo_czas = customtkinter.CTkLabel(master=frame, text="Czas treningu:", font=("Arial", 14))
label_drzewo_czas.pack(pady=8, padx=10)
label_drzewo_cel = customtkinter.CTkLabel(master=frame, text="Celność:", font=("Arial", 14))
label_drzewo_cel.pack(pady=8, padx=10)
label_drzewo_wynik = customtkinter.CTkLabel(master=frame, text="Wynik przewidywania:", font=("Arial", 14))
label_drzewo_wynik.pack(pady=8, padx=10)

label_neuro = customtkinter.CTkLabel(master=frame, text="Sieć neuronowa", font=("Arial", 14))
label_neuro.pack(pady=8, padx=10)
label_neuro_czas = customtkinter.CTkLabel(master=frame, text="Czas treningu:", font=("Arial", 14))
label_neuro_czas.pack(pady=8, padx=10)
label_neuro_cel = customtkinter.CTkLabel(master=frame, text="Celność:", font=("Arial", 14))
label_neuro_cel.pack(pady=8, padx=10)
label_neuro_wynik = customtkinter.CTkLabel(master=frame, text="Wynik przewidywania:", font=("Arial", 14))
label_neuro_wynik.pack(pady=8, padx=10)

root.mainloop()
