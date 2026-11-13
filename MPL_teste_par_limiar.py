from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     FixedThresholdClassifier, cross_val_predict)
from sklearn.metrics import (make_scorer, fbeta_score, recall_score,
                             precision_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)   # 0 = maligno, 1 = benigno

# Pipeline base: padronização + MLP
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(max_iter=2000, random_state=42)),
])

# Envolve o pipeline: o limiar passa a valer para a probabilidade de "maligno"
modelo = FixedThresholdClassifier(pipe, pos_label=0, response_method="predict_proba")

# Hiperparâmetros da MLP (prefixo estimator__mlp__) + limiar (threshold)
param_grid = {
    "estimator__mlp__hidden_layer_sizes": [(50,), (100,), (64, 32)],
    "estimator__mlp__activation": ["relu", "tanh"],
    "estimator__mlp__solver": ["lbfgs", "adam"],
    "estimator__mlp__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
    "threshold": [0.5, 0.4, 0.3, 0.2],
}

# Métricas com "maligno" (0) como classe positiva
scoring = {
    "f2": make_scorer(fbeta_score, beta=2, pos_label=0),
    "recall": make_scorer(recall_score, pos_label=0),
    "precisao": make_scorer(precision_score, pos_label=0),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

busca = GridSearchCV(modelo, param_grid, cv=cv, scoring=scoring,
                     refit="f2",   # escolhe o melhor pelo F2
                     n_jobs=-1)
busca.fit(X, y)

i = busca.best_index_
res = busca.cv_results_
print("Melhores hiperparâmetros:", busca.best_params_)
print(f"F2={res['mean_test_f2'][i]:.3f} | "
      f"Recall={res['mean_test_recall'][i]:.3f} | "
      f"Precisão={res['mean_test_precisao'][i]:.3f}")

# Matriz de confusão 5-fold com a melhor configuração
pred = cross_val_predict(busca.best_estimator_, X, y, cv=cv)
ConfusionMatrixDisplay.from_predictions(
    y, pred, display_labels=["Maligno", "Benigno"], cmap="Blues")
plt.title("Matriz de Confusão - MLP otimizada (5-Fold)")
plt.show()