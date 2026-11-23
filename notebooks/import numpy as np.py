import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

csv_path = Path(__file__).resolve().parent.parent / 'data' / 'house_prices.csv'
df = pd.read_csv(csv_path)


X = df[['area_m2']]
y = df['price_eur']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

results = pd.DataFrame({
    'area_m2': X_test['area_m2'].values,
    'actual_price': y_test.values,
    'predicted_price': y_pred.round(0).astype(int)
})

results.head(10)

x_line = pd.DataFrame({'area_m2': np.linspace(df['area_m2'].min(), df['area_m2'].max(), 200)})
y_line = model.predict(x_line)

results['residual'] = y_test.values - y_pred
results['residual'] = results['residual'].round(0).astype(int)
results.head(10)

train_predictions = model.predict(X_train)
train_residuals = y_train.values - train_predictions
train_sse = np.sum(train_residuals ** 2)

def regression_metrics(y_true, y_hat):
    mse = mean_squared_error(y_true, y_hat)
    return {
        'MAE (€)': mean_absolute_error(y_true, y_hat),
        'MSE (€²)': mse,
        'RMSE (€)': np.sqrt(mse),
        'R²': r2_score(y_true, y_hat),
    }

metrics_1f = pd.DataFrame({
    'Training': regression_metrics(y_train, model.predict(X_train)),
    'Test': regression_metrics(y_test, y_pred),
})
metrics_1f.style.format('{:,.3f}', subset=pd.IndexSlice[['R²'], :]) \
                .format('{:,.0f}', subset=pd.IndexSlice[['MAE (€)', 'MSE (€²)', 'RMSE (€)'], :])

features = ['area_m2', 'bedrooms', 'distance_to_city_km', 'age_years']

X_multi = df[features]
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred_m = model_multi.predict(X_test_m)

comparison = pd.DataFrame({
    'Area only (test)': regression_metrics(y_test, y_pred),
    'Four features (test)': regression_metrics(y_test_m, y_pred_m),
})
comparison.drop(index='MSE (€²)').round(3)

def regression_metrics(y_true, y_hat):
    mse = mean_squared_error(y_true, y_hat)
    return {
        'MAE (€)': mean_absolute_error(y_true, y_hat),
        'MSE (€²)': mse,
        'RMSE (€)': np.sqrt(mse),
        'R²': r2_score(y_true, y_hat),
    }

metrics_1f = pd.DataFrame({
    'Training': regression_metrics(y_train, model.predict(X_train)),
    'Test': regression_metrics(y_test, y_pred),
})
metrics_1f.style.format('{:,.3f}', subset=pd.IndexSlice[['R²'], :]) \
                .format('{:,.0f}', subset=pd.IndexSlice[['MAE (€)', 'MSE (€²)', 'RMSE (€)'], :])

TRUE = {
    'intercept': 120_000,
    'area_m2': 3_800,
    'bedrooms': 15_000,
    'distance_to_city_km': -9_500,
    'age_years': -1_200,
}

learned = pd.Series(
    [model_multi.intercept_, *model_multi.coef_],
    index=['intercept', *features]
)
pd.DataFrame({'True': pd.Series(TRUE), 'Learned': learned.round(0)})

noise_rng = np.random.default_rng(0)
n_noise = 10

X_noisy = X_multi.copy()
for k in range(n_noise):
    X_noisy[f'noise_{k}'] = noise_rng.normal(size=len(X_noisy))

X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
    X_noisy, y, test_size=0.2, random_state=42
)
model_noisy = LinearRegression().fit(X_train_n, y_train_n)

pd.DataFrame({
    'Training R²': [model_multi.score(X_train_m, y_train_m), model_noisy.score(X_train_n, y_train_n)],
    'Test R²':     [model_multi.score(X_test_m, y_test_m),   model_noisy.score(X_test_n, y_test_n)],
}, index=['4 real features', f'4 real + {n_noise} noise features']).round(3)

toy_rng = np.random.default_rng(39)
true_curve = lambda x: np.sin(2 * np.pi * x)

x_toy_train = np.sort(np.clip(np.linspace(0, 1, 18) + toy_rng.normal(0, 0.02, 18), 0, 1))
y_toy_train = true_curve(x_toy_train) + toy_rng.normal(0, 0.25, 18)
x_toy_test = np.sort(toy_rng.uniform(x_toy_train.min(), x_toy_train.max(), 18))
y_toy_test = true_curve(x_toy_test) + toy_rng.normal(0, 0.25, 18)

def poly_model(degree):
    return make_pipeline(
        PolynomialFeatures(degree, include_bias=False),
        StandardScaler(),
        LinearRegression()
    ).fit(x_toy_train[:, None], y_toy_train)

def rmse(m, x, y_true):
    return np.sqrt(mean_squared_error(y_true, m.predict(x[:, None])))

x_grid = np.linspace(x_toy_train.min(), x_toy_train.max(), 500)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for ax, (degree, label) in zip(axes, [(1, 'Underfitting'), (3, 'Good fit'), (15, 'Overfitting')]):
    m = poly_model(degree)
    ax.plot(x_grid, m.predict(x_grid[:, None]), color='C2', lw=2, label='Model')
    ax.scatter(x_toy_train, y_toy_train, color='C3', label='Training data')
    ax.scatter(x_toy_test, y_toy_test, facecolors='none', edgecolors='k', label='Test data')
    ax.set_ylim(-1.9, 1.9)
    ax.set_title(f'{label} (degree {degree})\n'
                 f'train RMSE {rmse(m, x_toy_train, y_toy_train):.2f} · '
                 f'test RMSE {rmse(m, x_toy_test, y_toy_test):.2f}')
    ax.set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].legend(loc='lower left')
plt.tight_layout()
plt.show()

degrees = range(1, 16)
train_err = [rmse(poly_model(d), x_toy_train, y_toy_train) for d in degrees]
test_err  = [rmse(poly_model(d), x_toy_test,  y_toy_test)  for d in degrees]

plt.figure(figsize=(9, 5))
plt.plot(degrees, train_err, 'o-', label='Training error')
plt.plot(degrees, test_err,  'o-', label='Test error')
plt.yscale('log')
plt.xticks(list(degrees))
plt.xlabel('Model complexity (polynomial degree)')
plt.ylabel('RMSE (log scale)')
plt.title('Training vs. test error')
plt.legend()
plt.show()