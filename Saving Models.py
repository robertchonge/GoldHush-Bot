import joblib

# Saving the trained models

joblib.dump(gb_model, 'gradient_boosting_model.pkl')

joblib.dump(rf_model, 'random_forest_model.pkl')


joblib.dump(linear_model, 'linear_model.pkl')