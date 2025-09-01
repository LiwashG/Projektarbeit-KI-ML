import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

# Ordner für Plots erstellen
if not os.path.exists("plots"):
    os.makedirs("plots")

class base_class:
    def __init__(self, df, Y_class):
        self.target_names = ['class 0', 'class 1', 'class 2']
        self.Y_class = Y_class
        self.df = df
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df, self.Y_class, test_size=0.2, random_state=42
        )

    def predicten(self):
        self.y_pred = self.model.predict(self.X_test)
        print(f"The Predicted Outputs are as follows: \n{self.y_pred}")

    def Confusion_mat(self, save=True, name=None):
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        if save:
            filename = f"plots/confusion_matrix_{self.Name}.png" if name is None else f"plots/{name}.png"
            plt.savefig(filename, bbox_inches='tight')


    def get_metrics_score(self):
        print(f"The calculated Accuracy Score of the {self.Name} Model is {accuracy_score(self.y_test, self.y_pred)}")
        print(f"The Classification Report of the {self.Name} is \n{classification_report(self.y_test, self.y_pred, target_names=self.target_names)}")


class RFC(base_class):
    def __init__(self, df, Y_class, name=None):
        super().__init__(df, Y_class)
        self.model = RandomForestClassifier(random_state=42, n_jobs=-1)
        self.Name = "Random Forest Classifier" if name is None else name
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.model.feature_importances = self.feature_importance()

    def params_opt(self):
        parameters = {'n_estimators':[1000, 2500, 5000, 7500, 10000], 
                      'max_depth':[30,35,45,50], 
                      "ccp_alpha":[0.01,0.1,1,10]}
        self.opt_model = GridSearchCV(self.model, parameters, n_jobs=-1, verbose=0)
        self.opt_model.fit(self.X_train, self.y_train)
        self.y_pred_clf = self.opt_model.predict(self.X_test)

    def feature_importance(self, save=True):
        self.feature_importance = self.model.feature_importances_
        for feature, importance in zip(self.X_train.columns, self.feature_importance):
            print(f"Feature: {feature} and the Importance: {importance}")

        sorted_indices = self.feature_importance.argsort()[::-1]
        sorted_importances = self.feature_importance[sorted_indices]

        plt.figure(figsize=(10,6))
        plt.bar(range(self.X_train.shape[1]), sorted_importances)
        plt.xticks(range(self.X_train.shape[1]), self.X_train.columns[sorted_indices], rotation=90)
        plt.title("Feature Importances")
        if save:
            plt.savefig("plots/feature_importances_RFC.png", bbox_inches='tight')


    def random_filter(self):
        Ind = np.where(self.feature_importance < self.feature_importance[-1])
        Colname = self.df.columns[Ind]

        df1 = self.df.drop(columns=Colname)
        df1 = df1.drop(columns="Random_Variable")
        self.df1 = df1
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df1, self.Y_class, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(random_state=42, n_jobs=-1)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

        self.Confusion_mat()
        self.get_metrics_score()


class Dec_tree(base_class):
    def __init__(self, df, Y_class, feature_importances_=None):
        super().__init__(df, Y_class)
        self.model = DecisionTreeClassifier(random_state=42)
        self.Name = "Decision Tree"
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.feature_importance = feature_importances_

    def params_opt(self):
        parameters = {'max_depth':[10,50,100,500], "ccp_alpha":[0.01,0.1,1,10]}
        self.opt_model = GridSearchCV(self.model, parameters, n_jobs=-1, verbose=0)
        self.opt_model.fit(self.X_train, self.y_train)
        self.y_pred_clf = self.opt_model.predict(self.X_test)


class SVM_clf(base_class):
    def __init__(self, df, Y_class, feature_importances_=None, name=None, dec=None):
        super().__init__(df, Y_class)
        if dec is None:
            self.model = make_pipeline(StandardScaler(), SVC(random_state=42))
        elif dec=="n":
            self.model = make_pipeline(Normalizer(), SVC(random_state=42))
        elif dec=="r":
            self.model = make_pipeline(RobustScaler(), SVC(random_state=42))
        self.Name = "Support Vector Classifier" if name is None else name
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.feature_importance = feature_importances_

    def params_opt(self):
        parameters = {'svc__kernel': ["linear", "poly", "rbf", "sigmoid"],
                      'svc__C': [0.1, 1, 5, 10]}
        self.opt_model = GridSearchCV(self.model, parameters, cv=5, n_jobs=-1, verbose=0)
        self.opt_model.fit(self.X_train, self.y_train)
        self.y_pred_clf = self.opt_model.predict(self.X_test)

if __name__ == "__main__":
    ## Load Data
    df = pd.read_excel(r"Projektarbeit-KI-ML/Supervised/chiefs_knife_dataset.xlsx")
    Y = df['Ra']

    ind_0 = np.where(Y < 0.13)
    ind_1 = np.where((Y >= 0.13) & (Y <= 0.21))
    ind_2 = np.where(Y > 0.21)

    Y_class = Y.copy()
    Y_class[ind_0] = 0
    Y_class[ind_1] = 1
    Y_class[ind_2] = 2

    df = df.iloc[:, 2:-17].drop(columns="Linie")
    df["Random_Variable"] = np.random.rand(len(df),1) * 100

    ## Train Models
    rfc = RFC(df, Y_class)
    rfc.Confusion_mat()
    rfc.get_metrics_score()
    rfc.random_filter()

    df = rfc.df1
    dt = Dec_tree(df, Y_class, rfc.model.feature_importances_)
    svc_clf_s = SVM_clf(df, Y_class, rfc.model.feature_importances_, "SVC Standardized")
    svc_clf_n = SVM_clf(df, Y_class, rfc.model.feature_importances_, "SVC Normalized", "n")
    svc_clf_r = SVM_clf(df, Y_class, rfc.model.feature_importances_, "SVC Robust", "r")

    # Confusion Matrices
    rfc.Confusion_mat()
    dt.Confusion_mat()
    svc_clf_s.Confusion_mat()
    svc_clf_n.Confusion_mat()
    svc_clf_r.Confusion_mat()

    # Metric Scores
    dt.get_metrics_score()
    svc_clf_s.get_metrics_score()
    svc_clf_n.get_metrics_score()
    svc_clf_r.get_metrics_score()

    # Hyperparameter Optimization
    rfc.params_opt()
    dt.params_opt()
    svc_clf_s.params_opt()
    svc_clf_n.params_opt()
    svc_clf_r.params_opt()
