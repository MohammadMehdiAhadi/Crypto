# You Should Run data_feature First
try:
    from Models.Final_Model import *
    from models_fit_predict import *

    # Stacking predictions
    predictions_stacking = np.vstack([logistic_predict,
                                      mlp_predict,
                                      knn_predict,
                                      svm_predict,
                                      decision_tree_predict,
                                      random_forest_predict
                                      ]).T

    # Meta model prediction
    predictions_final = final_pred(predictions_stacking, y_test, predictions_stacking)
    accuracy = np.mean(predictions_final == y_test)
    print("دقت مدل Stacking:", accuracy * 100)
    print("________________________________________________________________")
    # Classification reports
    print("Logistic :")
    print(classification_report(y_test, logistic_predict, zero_division=1))
    print("________________________________________________________________")
    print("MLPClassifier :")
    print(classification_report(y_test, mlp_predict, zero_division=1))
    print("________________________________________________________________")
    print("KNN :")
    print(classification_report(y_test, knn_predict, zero_division=1))
    print("________________________________________________________________")
    print("SVM :")
    print(classification_report(y_test, svm_predict, zero_division=1))
    print("________________________________________________________________")
    print("DecisionTree :")
    print(classification_report(y_test, decision_tree_predict, zero_division=1))
    print("________________________________________________________________")
    print("RandomForest :")
    print(classification_report(y_test, random_forest_predict, zero_division=1))
    print("________________________________________________________________")
    print("Final :")
    print(classification_report(y_test, predictions_final))
    print("________________________________________________________________")

    # Create a heatmap for the confusion matrix

    print("visualizing . . .")
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

    # dfgvtbh
    fpr, tpr, thresholds = roc_curve(y_test, predictions_final)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.3f)' % roc_auc)
    plt.title('ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.grid(True)
    plt.savefig("roc_auc.jpg")
    plt.show()
    print('The End . ')
except Exception as e:
    print("Something Went Wrong")
    print(e)
