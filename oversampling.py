from imblearn.over_sampling import SMOTE
import pandas as pd

def apply_smote(X_train, y_train):
    print(f"\n{'='*20} Aplicare SMOTE pe setul de antrenament {'='*20}")
    
    print("Distributia initiala a claselor in Training:")
    print(y_train.value_counts())

    smote = SMOTE(random_state=42, k_neighbors=3) # k_neighbors=3 pentru a evita crearea de exemple zgomotoase
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print("Distributia claselor dupa SMOTE:")
    print(y_train_res.value_counts())
    
    return X_train_res, y_train_res
