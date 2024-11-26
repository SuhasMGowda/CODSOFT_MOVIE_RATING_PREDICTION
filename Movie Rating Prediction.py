import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

file_path = r'C:\Users\ACER\Desktop\Suhas\Internship\CodeSoft\IMDb_Movies_India.csv'
data = pd.read_csv(file_path, encoding='latin1')
data = data.dropna(subset=['Genre', 'Director', 'Actor 1', 'Rating'])
features = data[['Genre', 'Director', 'Actor 1']]
target = data['Rating']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
categorical_features = ['Genre', 'Director', 'Actor 1']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('regressor', RandomForestRegressor(random_state=42))
])

model.fit(X_train, y_train)

genre = input("Enter the genre of the movie: ")
director = input("Enter the director of the movie: ")
actor = input("Enter the lead actor of the movie: ")

sample_movie = pd.DataFrame({'Genre': [genre], 'Director': [director], 'Actor 1': [actor]})
predicted_rating = model.predict(sample_movie)
print(f"Predicted Rating: {predicted_rating[0]:.2f}")
