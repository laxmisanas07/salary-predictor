import pandas as pd
from sklearn.linear_model import LinearRegression

# Load your CSV file
data = pd.read_csv('salary_data.csv')

# Create model (magic happens here)
X = data[['experience', 'age']]
y = data['income']
model = LinearRegression().fit(X, y)

# YOUR prediction
exp = float(input("Enter years of experience: "))
age = float(input("Enter your age: "))

salary = model.predict([[exp, age]])[0]
print(f"YOUR PREDICTED SALARY: ${salary:,.0f}")
