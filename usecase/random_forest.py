import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def select_random_forest(
    df: pd.DataFrame,
    top_n: int,  # 選択するpeakの数
    n_estimators: int,
    random_state: int,
    criterion: str,
    max_depth: int,
    min_samples_leaf: int,
    bootstrap: bool,
) -> tuple[list[int], pd.Series]:
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
    )
    X = df.to_numpy()
    y = df.index.to_numpy()

    clf = clf.fit(X, y)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    score = clf.score(X, y)
    print(f"Random forest classifier score: {score * 100}%")

    # predicted = clf.predict(X)
    # print("predicted")
    # print(f"{predicted}")

    result_indices = []
    print("Feature ranking:")
    for f in range(top_n):
        print(f"{f + 1}. feature {indices[f]} ({importances[indices[f]]})")
        result_indices.append(int(indices[f]))

    importances = pd.Series(importances, index=df.columns)
    return (result_indices, importances)
