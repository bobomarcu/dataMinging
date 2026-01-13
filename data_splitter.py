from sklearn.model_selection import train_test_split

def split_dataset(X, y, train_ratio=0.6, test_ratio=0.1, pred_ratio=0.3, random_state=42):

    if abs((train_ratio + test_ratio + pred_ratio) - 1.0) > 1e-9:
        raise ValueError("Proportiile trebuie sa insumeze 1.0 (100%)")

    temp_size = test_ratio + pred_ratio
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=temp_size, 
        random_state=random_state, 
        stratify=y  # Pastram distributia claselor
    )

    test_size_relative = test_ratio / temp_size
    
    X_test, X_pred, y_test, y_pred = train_test_split(
        X_temp, y_temp, 
        train_size=test_size_relative, 
        random_state=random_state, 
        stratify=y_temp
    )

    print("-" * 30)
    print("Split Distributie:")
    print(f"Training:   {X_train.shape[0]} ({train_ratio*100:.0f}%)")
    print(f"Testing:    {X_test.shape[0]} ({test_ratio*100:.0f}%)")
    print(f"Prediction: {X_pred.shape[0]} ({pred_ratio*100:.0f}%)")
    print("-" * 30)

    return X_train, X_test, X_pred, y_train, y_test, y_pred
