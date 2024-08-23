# You Should Run data_feature First
try:
    import numpy as np
    from Models.MLPClassifier_Model import *
    from Models.Knn_Model import *
    from Models.RandomForestClassifier_Model import *
    from Models.LogisticRegression_Model import *
    from Models.SVC_Model import *
    from Models.DecisionTreeClassifier_Model import *
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    from Models.Final_Knn_Model import *


    # Load data from CSV
    data = pd.read_csv("final_dataframe.csv", index_col="Date")
    if not data.empty:

        n = len(data)

        # Define features and target
        X = data[['Open', 'High', 'Low', 'Close', "Next_Hour_Open", 'Volume', "histogram", "ema7", "ema14", "ema21",
                  'sma', 'squeeze', 'upper_band', 'lower_band', 'macd',
                  'day_of_week']].iloc[29:n - 1]
        y = data["Benefit"].iloc[29:n - 1]

        # Split data
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.0005, shuffle=False, random_state=17)

        # Stacking predictions
        predictions_stacking = np.vstack([logistic_pred(x_train, y_train, x_test),
            mlp_pred(x_train, y_train, x_test),
            knn_pred(x_train, y_train, x_test),
            svm_pred(x_train, y_train, x_test),
            dt_pred(x_train, y_train, x_test),
            rf_pred(x_train, y_train, x_test)]).T

        # Meta model prediction
        predictions_final = knn_pred(predictions_stacking, y_test, predictions_stacking)
        accuracy = np.mean(predictions_final == y_test)
        print("دقت مدل Stacking:", accuracy * 100)
        print("________________________________________________________________")
        # Classification reports
        print("MLPClassifier :")
        print(classification_report(y_test, mlp_pred(x_train, y_train, x_test), zero_division=1))
        print("________________________________________________________________")
        print("Logistic :")
        print(classification_report(y_test, logistic_pred(x_train, y_train, x_test),zero_division = 1))
        print("________________________________________________________________")
        print("KNN :")
        print(classification_report(y_test, knn_pred(x_train, y_train, x_test), zero_division=1))
        print("________________________________________________________________")
        print("SVM :")
        print(classification_report(y_test, svm_pred(x_train, y_train, x_test), zero_division=1))
        print("________________________________________________________________")
        print("DecisionTree :")
        print(classification_report(y_test, dt_pred(x_train, y_train, x_test), zero_division=1))
        print("________________________________________________________________")
        print("RandomForest :")
        print(classification_report(y_test, rf_pred(x_train, y_train, x_test), zero_division=1))
        print("________________________________________________________________")
        print("Final :")
        print(classification_report(y_test, predictions_final))
        print("________________________________________________________________")

        # Create a heatmap for the confusion matrix

        print("Done")
        conf_matrix = confusion_matrix(y_test, predictions_final)
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig("final_predict.jpg")
        plt.show()

        # Bar Plot for Correct and Incorrect Predictions
        correct_predictions = np.sum(predictions_final == y_test)
        incorrect_predictions = np.sum(predictions_final != y_test)

        plt.bar(['Correct', 'Incorrect'], [correct_predictions, incorrect_predictions], color=['green', 'red'])
        plt.title("Correct vs Incorrect Predictions")
        plt.xlabel("Prediction Type")
        plt.ylabel("Number of Predictions")
        plt.savefig("final_predict_bar_chart.jpg")
        plt.show()


    else:
        print("Couldn't Load The Data")
        print("Try Again")
except:
    pass
