from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from utils import load_standard_data, load_biased_data
from preprocess import preprocess_data

df = load_standard_data()

df['Total_Score'] = (df['Midterm_Score'] + df['Final_Score'] + df['Projects_Score']) / 3

y = df['Total_Score']

preprocessor, X = preprocess_data(df, target_cols=['Total_Score'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

prediction = model.predict(X_test)
mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")