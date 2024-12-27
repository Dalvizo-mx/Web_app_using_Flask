import os
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import io
import base64
from datetime import date

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return redirect(url_for('process_file', filename=file.filename))
    return redirect(request.url)

@app.route('/process/<filename>')
def process_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(file_path)

    # Data processing
    df = df[df.Rainfall_Bastia_Umbra.notna()].reset_index(drop=True)
    df = df.drop(['Depth_to_Groundwater_P24', 'Temperature_Petrignano'], axis=1)
    df.columns = ['date', 'rainfall', 'depth_to_groundwater', 'temperature', 'drainage_volume', 'river_hydrometry']
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df = df.sort_values(by='date')

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
    score_rmse = mean_squared_error(y_valid, y_pred.tail(len(y_valid))['yhat'], squared=False)

    # Plot results
    f, ax = plt.subplots(1, figsize=(15, 6))
    model.plot(y_pred, ax=ax)
    sns.lineplot(x=x_valid, y=y_valid, ax=ax, color='orange', label='Ground truth')

    ax.set_title(f'Prediction \n MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Depth to Groundwater', fontsize=14)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return render_template('results.html', plot_data=plot_data, mae=score_mae, rmse=score_rmse)

if __name__ == '__main__':
    app.run(debug=True)
