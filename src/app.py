import os
from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import io
import base64
from datetime import date
import math

app = Flask(__name__)

@app.route('/')
def index():
    # Path to the CSV file
    df = pd.read_csv('/workspaces/Web_app_using_Flask/Aquifer_Petrignano (1).csv')

    # Data processing
    df = df[df.Rainfall_Bastia_Umbra.notna()].reset_index(drop=True)
    df = df.drop(['Depth_to_Groundwater_P24', 'Temperature_Petrignano'], axis=1)
    df.columns = ['date', 'rainfall', 'depth_to_groundwater', 'temperature', 'drainage_volume', 'river_hydrometry']
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df = df.sort_values(by='date')

    # Fill NaN values
    df['depth_to_groundwater'] = df['depth_to_groundwater'].fillna(method='ffill')
    df['depth_to_groundwater'] = df['depth_to_groundwater'].fillna(method='bfill')

    # Ensure there are no remaining NaN values
    df = df.dropna()

    # Prepare data for Prophet
    univariate_df = df[['date', 'depth_to_groundwater']].copy()
    univariate_df.columns = ['ds', 'y']
    train_size = int(0.85 * len(df))
    train = univariate_df.iloc[:train_size, :]
    x_valid, y_valid = univariate_df.iloc[train_size:, 0], univariate_df.iloc[train_size:, 1]

    # Train and predict with Prophet
    model = Prophet()
    model.fit(train)
    y_pred = model.predict(pd.DataFrame(x_valid))

    # Calculate metrics
    score_mae = mean_absolute_error(y_valid, y_pred.tail(len(y_valid))['yhat'])
    score_rmse = math.sqrt(mean_squared_error(y_valid, y_pred.tail(len(y_valid))['yhat']))

    # Plot results
    f, ax = plt.subplots(1, figsize=(15, 6))
    model.plot(y_pred, ax=ax)
    sns.lineplot(x=x_valid, y=y_valid, ax=ax, color='orange', label='Ground truth')

    ax.set_title(f'Prediction \n MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Depth to Groundwater', fontsize=14)

    # Save plot to a string buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return render_template('results.html', plot_data=plot_data, mae=score_mae, rmse=score_rmse)

if __name__ == '__main__':
    app.run(debug=True)
