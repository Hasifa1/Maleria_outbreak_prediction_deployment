from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
#model.pkl is trained ml model
import pickle
#deserialize -read the binary file of ml model
clf= pickle.load(open('model.pkl','rb'))
#################################################################
#for getting range decided onxtrain- repeat the steps till normalization
import pandas as pd
df=pd.read_csv("outbreak_detect.csv")

#removing null values
import math
mean_mxt=df['maxTemp'].mean()
mean_mxt=math.floor(mean_mxt)

df['maxTemp']=df['maxTemp'].fillna(mean_mxt)
mean_mnt=df['minTemp'].mean()
mean_mnt=math.floor(mean_mnt)

df['minTemp']=df['minTemp'].fillna(mean_mnt)
#data preprocessing
from sklearn import preprocessing
LE=preprocessing.LabelEncoder()
#fitting the technique to dataset
df.Outbreak=LE.fit_transform(df.Outbreak)

df=df.drop(25,axis=0)
df=df.drop(['Positive','pf'],axis=1)

#splitting the data
X=df.iloc[:,:-1]
Y=df.iloc[:,-1:]
#splitting
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#######################################################################
#normalize

app=Flask(__name__)

@app.route("/")#annotation that triggers the methods.---->default annotation that renders the 1st web page to the browser
def hello():
    return render_template('index.html')
#jinja 2  ->template engine which would be going to templates folder and selecting the web page hence folder name should be template

@app.route('/predict',methods=['POST','GET'])
def predict_class():
    print([x for x in request.form.values()]) #for checking the post method
    features=[int(x) for x in request.form.values()]
    print(features)
    sst=StandardScaler().fit(X_train)
    output=clf.predict(sst.transform([features]))
    print(output)
    if output[0]==0:
        return render_template('index.html',pred=f'There is no chance of maleria outbreak')
    else:
        return render_template('index.html',pred=f'The person a chance of maleria outbreak')


if __name__=="__main__":
    app.run(debug=True)