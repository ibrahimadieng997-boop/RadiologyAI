"""A coller a la FIN de votre notebook (apres l'evaluation) :

    from export_artifacts import export
    export(model, history, x_test, X_test_vgg, y_test)

Cree le dossier artifacts/ lu par l'application (modele, metriques, courbes, exemples).
"""
import json, os
import numpy as np
from PIL import Image
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             precision_recall_fscore_support, confusion_matrix)

def export(model, history, x_test, X_test_vgg, y_test, out="artifacts"):
    os.makedirs(f"{out}/samples", exist_ok=True)
    model.save(f"{out}/model.keras")
    p = model.predict(X_test_vgg).ravel()
    yp = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, yp)
    fpr, tpr, _ = roc_curve(y_test, p)
    pp, rr, _ = precision_recall_curve(y_test, p)
    loss, acc = model.evaluate(X_test_vgg, y_test, verbose=0)
    m = {"accuracy": float(acc), "loss": float(loss), "n_test": int(len(y_test)),
         "precision": pr.tolist(), "recall": rc.tolist(), "f1": f1.tolist(),
         "auc": float(auc(fpr, tpr)), "ap": float(average_precision_score(y_test, p)),
         "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
         "pr": {"precision": pp.tolist(), "recall": rr.tolist()},
         "cm": confusion_matrix(y_test, yp).tolist()}
    json.dump(m, open(f"{out}/metrics.json", "w"))
    json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()},
              open(f"{out}/history.json", "w"))
    samples = []
    for cls in (0, 1):
        for i in np.where(y_test == cls)[0][:8]:
            fn = f"samples/{'normal' if cls == 0 else 'pneumonia'}_{i}.png"
            Image.fromarray((x_test[i] * 255).astype("uint8")).save(f"{out}/{fn}")
            samples.append({"file": fn, "true": int(cls), "prob": float(p[i])})
    json.dump(samples, open(f"{out}/samples.json", "w"))
    print("Export termine ->", out)
