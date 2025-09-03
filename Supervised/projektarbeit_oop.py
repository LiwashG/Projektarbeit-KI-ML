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

# Globale Liste für Ergebnisse
results_list = []

# ------------------- Funktion zum Excel-Speichern -------------------
def save_results_to_excel(results_list, filename="model_results.xlsx", folder="results"):
    if not results_list:
        print("Warning: results_list is empty. No Excel file will be saved.")
        return

    if not os.path.exists(folder):
        os.makedirs(folder)

    results_df = pd.DataFrame(results_list)

    # Metrics-Blatt
    metrics_cols = [col for col in results_df.columns if col != "Best Params"]
    metrics_df = results_df[metrics_cols]

    # Hyperparameters-Blatt
    hyperparams_df = results_df[["Model", "Best Params"]]

    filepath = os.path.join(folder, filename)
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        hyperparams_df.to_excel(writer, sheet_name="Hyperparameters", index=False)

    print(f"Ergebnisse wurden in '{os.path.abspath(filepath)}' gespeichert!")

# ------------------- Basisklasse -------------------
class base_class:
    def __init__(self, df, Y_class):
        self.target_names = ['class 0', 'class 1', 'class 2']
        self.Y_class = Y_class
        self.df = df
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df, self.Y_class, test_size=0.2, random_state=42
        )
        self.best_params = None

    def Confusion_mat(self, save=True, name=None):
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        if save:
            filename = f"plots/confusion_matrix_{self.Name}.png" if name is None else f"plots/{name}.png"
            plt.savefig(filename, bbox_inches='tight')

    def get_metrics_score(self):
        acc = accuracy_score(self.y_test, self.y_pred)
        clf_report = classification_report(self.y_test, self.y_pred, target_names=self.target_names, output_dict=True)

        print(f"The calculated Accuracy Score of the {self.Name} Model is {acc}")
        print(f"The Classification Report of the {self.Name} is \n{classification_report(self.y_test, self.y_pred, target_names=self.target_names)}")

        results_list.append({
            "Model": self.Name,
            "Accuracy": acc,
            "Precision_class0": clf_report["class 0"]["precision"],
            "Recall_class0": clf_report["class 0"]["recall"],
            "F1_class0": clf_report["class 0"]["f1-score"],
            "Precision_class1": clf_report["class 1"]["precision"],
            "Recall_class1": clf_report["class 1"]["recall"],
            "F1_class1": clf_report["class 1"]["f1-score"],
            "Precision_class2": clf_report["class 2"]["precision"],
            "Recall_class2": clf_report["class 2"]["recall"],
            "F1_class2": clf_report["class 2"]["f1-score"],
            "Macro_F1": clf_report["macro avg"]["f1-score"],
            "Weighted_F1": clf_report["weighted avg"]["f1-score"],
            "Best Params": str(self.best_params) if self.best_params else ""
        })

# ------------------- Random Forest Classifier -------------------
class RFC(base_class):
    def __init__(self, df, Y_class, name=None):
        super().__init__(df, Y_class)
        self.model = RandomForestClassifier(random_state=42, n_jobs=-1)
        self.Name = "Random Forest Classifier" if name is None else name
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.feature_importance = self.feature_importance_func()

    def params_opt(self):
        parameters = {'n_estimators': [1000, 2500, 5000, 7500, 10000],
                      'max_depth': [30, 35, 45, 50],
                      "ccp_alpha": [0.01, 0.1, 1, 10]}
        opt_model = GridSearchCV(self.model, parameters, n_jobs=-1, verbose=0)
        opt_model.fit(self.X_train, self.y_train)
        self.y_pred = opt_model.predict(self.X_test)
        self.best_params = opt_model.best_params_

    def feature_importance_func(self, save=True):
        self.feature_importance = self.model.feature_importances_
        sorted_indices = self.feature_importance.argsort()[::-1]
        sorted_importances = self.feature_importance[sorted_indices]

        plt.figure(figsize=(10, 6))
        plt.bar(range(self.X_train.shape[1]), sorted_importances)
        plt.xticks(range(self.X_train.shape[1]), self.X_train.columns[sorted_indices], rotation=90)
        plt.title("Feature Importances")
        if save:
            plt.savefig("plots/feature_importances_RFC.png", bbox_inches='tight')
        return self.feature_importance

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

# ------------------- Decision Tree -------------------
class Dec_tree(base_class):
    def __init__(self, df, Y_class, feature_importances_=None):
        super().__init__(df, Y_class)
        self.model = DecisionTreeClassifier(random_state=42)
        self.Name = "Decision Tree"
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.feature_importance = feature_importances_

    def params_opt(self):
        parameters = {'max_depth': [10, 50, 100, 500], "ccp_alpha": [0.01, 0.1, 1, 10]}
        opt_model = GridSearchCV(self.model, parameters, n_jobs=-1, verbose=0)
        opt_model.fit(self.X_train, self.y_train)
        self.y_pred = opt_model.predict(self.X_test)
        self.best_params = opt_model.best_params_

# ------------------- Support Vector Classifier -------------------
class SVM_clf(base_class):
    def __init__(self, df, Y_class, feature_importances_=None, name=None, dec=None):
        super().__init__(df, Y_class)
        if dec is None:
            self.model = make_pipeline(StandardScaler(), SVC(random_state=42))
        elif dec == "n":
            self.model = make_pipeline(Normalizer(), SVC(random_state=42))
        elif dec == "r":
            self.model = make_pipeline(RobustScaler(), SVC(random_state=42))
        self.Name = "Support Vector Classifier" if name is None else name
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

    def params_opt(self):
        parameters = {'svc__kernel': ["linear", "poly", "rbf", "sigmoid"],
                      'svc__C': [0.1, 1, 5, 10]}
        opt_model = GridSearchCV(self.model, parameters, cv=5, n_jobs=-1, verbose=0)
        opt_model.fit(self.X_train, self.y_train)
        self.y_pred = opt_model.predict(self.X_test)
        self.best_params = opt_model.best_params_

# ------------------- Main -------------------
if __name__ == "__main__":
    # Load Data
    df = pd.read_excel(r"chiefs_knife_dataset.xlsx")
    Y = df['Ra']

    ind_0 = np.where(Y < 0.13)
    ind_1 = np.where((Y >= 0.13) & (Y <= 0.21))
    ind_2 = np.where(Y > 0.21)

    Y_class = Y.copy()
    Y_class[ind_0] = 0
    Y_class[ind_1] = 1
    Y_class[ind_2] = 2

    df = df.iloc[:, 2:-17].drop(columns="Linie")
    df["Random_Variable"] = np.random.rand(len(df), 1) * 100

    # Train Models
    rfc = RFC(df, Y_class)
    df = rfc.df1 if hasattr(rfc, 'df1') else df
    dt = Dec_tree(df, Y_class, rfc.feature_importance)
    svc_clf_s = SVM_clf(df, Y_class, rfc.feature_importance, "SVC Standardized")
    svc_clf_n = SVM_clf(df, Y_class, rfc.feature_importance, "SVC Normalized", "n")
    svc_clf_r = SVM_clf(df, Y_class, rfc.feature_importance, "SVC Robust", "r")

    # Hyperparameter Optimization zuerst
    rfc.params_opt()
    dt.params_opt()
    svc_clf_s.params_opt()
    svc_clf_n.params_opt()
    svc_clf_r.params_opt()

    # Metric Scores nach Optimierung speichern (inkl. Best Params)
    rfc.get_metrics_score()
    dt.get_metrics_score()
    svc_clf_s.get_metrics_score()
    svc_clf_n.get_metrics_score()
    svc_clf_r.get_metrics_score()

    # Confusion Matrices
    rfc.Confusion_mat()
    dt.Confusion_mat()
    svc_clf_s.Confusion_mat()
    svc_clf_n.Confusion_mat()
    svc_clf_r.Confusion_mat()

    # Ergebnisse in Excel speichern
    save_results_to_excel(results_list)
