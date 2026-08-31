import nflreadpy as nfl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

ff_opp = nfl.load_ff_opportunity(
    seasons=range(2018, 2026)
).to_pandas()

FEATURES = [
    "total_fantasy_points_exp",
    "total_yards_gained_exp",
    "total_touchdown_exp",
    "total_first_down_exp"
]

# Remove missing values
data = ff_opp[FEATURES + ["total_fantasy_points"]].dropna()

X = data[FEATURES]
y = data["total_fantasy_points"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("R²:", r2_score(y_test, predictions))