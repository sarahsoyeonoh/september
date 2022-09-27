import xgboost
import shap
import pandas as pd

data = pd.read_csv('/Users/sarahoh/Desktop/py_scripts/hello/220901/dataset.csv')

X = data[['ethnicity', 'race','maternal_age','complications', 'type', 'prenatal_care', 'first_frequency', 'first_amount', 'second_frequency', 'second_amount', 'third_frequency', 'third_amount']]
y = data['fas_f']

model = xgboost.XGBClassifier().fit(X, y)

# compute SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

shap.plots.bar(shap_values)
shap.plots.bar(shap_values, max_display=20)


sex = ["Women" if shap_values[i,"BSPT_SEX_1M2F"].data == 2 else "Men" for i in range(shap_values.shape[0])]
shap.plots.bar(shap_values.cohorts(sex).abs.mean(0), max_display=20)

