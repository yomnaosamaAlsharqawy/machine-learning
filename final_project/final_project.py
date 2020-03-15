from tkinter import *

import pandas as pds
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


# machine learning part
def calculate_accuracy(data: pds, result_text: Text, par: int) -> NONE:
    x_train, x_test, y_train, y_test = data
    if par == 1:
        clf = GaussianNB()
    elif par == 2:
        clf = SVC(kernel='sigmoid', gamma='auto')
    elif par == 3:
        clf = DecisionTreeClassifier(random_state=0)
        plot_tree(clf.fit(x_train, y_train))
    else:
        clf = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=2000)

    if par != 3:
        clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    result_text.configure(state='normal')
    result_text.delete(1.0, END)
    result_text.insert(END, accuracy_score(y_test, y_pred))
    result_text.configure(state='disabled')


# import excel and separate it
def import_excel_data(file_name: str, target_name: str) -> pds:
    data = pds.read_excel(file_name)
    df = pds.DataFrame(data)
    y = df[target_name]
    x = df.drop([target_name], axis=1)
    return x, y


# Splits Excel data to be trained and tested
def split_data(data: pds):
    x_train, x_test, y_train, y_test = train_test_split(data[0], data[1])
    global data_split
    data_split = x_train, x_test, y_train, y_test
    return x_train, x_test, y_train, y_test


# Displays Excel data
def display_excel(file_name: str):
    data = pds.read_excel(file_name)
    df = pds.DataFrame(data)
    pds.set_option('display.max_columns', 20)
    print(df)


# Excel Part
excel_name = 'HTRU_2.xlsx'
excel_data = import_excel_data(excel_name, 'class')
data_split = split_data(excel_data)

# GUI part
wind = Tk(className="Machine learning dataset")
l1 = Label(wind, text="Machine Learning")
l1.grid(row=0, column=3)

nbResult = []
for i in range(0, 4):
    nbResult.append(Text(wind, height=1, width=20))
    nbResult[i].grid(row=i + 1, column=4)
    nbResult[i].configure(state='disabled')

nb = Button(wind, text="Naive Bayes", command=lambda: calculate_accuracy(data_split, nbResult[0], 1))
nb.grid(row=1, column=1)

svm = Button(wind, text="Support Vector Machine", command=lambda: calculate_accuracy(data_split, nbResult[1], 2))
svm.grid(row=2, column=1)

DT = Button(wind, text="DecisionTree", command=lambda: calculate_accuracy(data_split, nbResult[2], 3))
DT.grid(row=3, column=1)

NN = Button(wind, text="Neural Network", command=lambda: calculate_accuracy(data_split, nbResult[3], 4))
NN.grid(row=4, column=1)

DS = Button(wind, text="Split data again", command=lambda: split_data(excel_data))
DS.grid(row=5, column=2)

SE = Button(wind, text="Show excel data", command=lambda: display_excel(excel_name))
SE.grid(row=5, column=3)

wind.mainloop()
