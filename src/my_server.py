from flask import Flask, render_template, request, send_file, send_from_directory
import pandas as pd
import ensembles
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from joblib import load
from joblib import dump

app = Flask(__name__, template_folder="templates")

user_data = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/choose_model', methods=['POST'])
def choose_model():
    selected_model = request.form.get('model')
    if selected_model not in ['Random Forest', 'Gradient Boosting']:
        return "Invalid model selected"
    user_data['selected_model'] = selected_model
    return render_template('trees.html')


@app.route('/set_parameters', methods=['POST'])
def set_parameters():
    try:
        num_trees = int(request.form['num_trees'])
        max_depth = int(request.form['max_depth'])
        subsample_size = int(request.form['subsample_size'])
        if num_trees <= 0 or max_depth <= 0 or subsample_size <= 0:
            raise ValueError("Parameters must be positive integers.")
        user_data['num_trees'] = num_trees
        user_data['max_depth'] = max_depth
        user_data['subsample_size'] = subsample_size
        return render_template('upload_data.html')
    except ValueError as e:
        return f"Invalid parameter values: {str(e)}"


@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    try:
        file = request.files.get('training_file')
        if not file or file.filename == '' or not file.filename.lower().endswith(('.csv')):
            raise ValueError("Please upload a valid CSV file.")
        filename = os.path.join('uploads', 'training_data.csv')
        file.save(filename)
        user_data['training_data_path'] = filename
        return render_template('upload_validation_data.html')
    except ValueError as e:
        return f"Error uploading training data: {str(e)}"


@app.route('/upload_validation_data', methods=['POST'])
def upload_validation_data():
    file = request.files['validation_file']
    if file:
        filename = os.path.join('uploads', 'validation_data.csv')
        file.save(filename)
        user_data['validation_data_path'] = filename
    return render_template('enter_target_variable.html')


@app.route('/upload_test_data', methods=['POST'])
def upload_test_data():
    file = request.files['test_file']
    filename = os.path.join('uploads', 'test_data.csv')
    file.save(filename)
    user_data['test_data_path'] = filename
    return render_template('enter_target_variable.html')


@app.route('/train_model', methods=['POST'])
def train_model():
    target_variable = request.form.get('target_variable')
    val = True
    try:
        # Загрузка данных из CSV-файлов
        training_data = pd.read_csv(user_data['training_data_path'])
        if 'validation_data_path' in user_data:
            validation_data = pd.read_csv(user_data['validation_data_path'])
        else:
            validation_data = None
        if 'validation_data_path' not in user_data:
            val = False
        if 'date' in training_data.columns:
            training_data = training_data.drop(columns=['date'])
        if val and 'date' in validation_data.columns:
            validation_data = validation_data.drop(columns=['date'])
        # One-Hot Encoding для категориальных признаков
        categorical_cols = training_data.select_dtypes(include=['object']).columns
        training_data = pd.get_dummies(training_data, columns=categorical_cols, drop_first=True)
        if val:
            validation_data = pd.get_dummies(validation_data, columns=categorical_cols, drop_first=True)
        # Разделение данных на обучающую и валидационную выборки
        X_train = training_data.drop(columns=[target_variable])
        y_train = training_data[target_variable]
        X_val = None
        y_val = None
        if val:
            X_val = validation_data.drop(columns=[target_variable])
            y_val = validation_data[target_variable]
        # Выбор модели в зависимости от выбора пользователя
        algorithm_name = ''
        if user_data['selected_model'] == 'Random Forest':
            model = ensembles.RandomForestMSE(n_estimators=user_data['num_trees'],
                                              max_depth=user_data['max_depth'],
                                              feature_subsample_size=user_data['subsample_size'])
            algorithm_name = 'Случайный лес'
        elif user_data['selected_model'] == 'Gradient Boosting':
            model = ensembles.GradientBoostingMSE(n_estimators=user_data['num_trees'],
                                                  max_depth=user_data['max_depth'],
                                                  feature_subsample_size=user_data['subsample_size'])
            algorithm_name = 'Градиентный бустинг'
        else:
            return "Invalid model selected"
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        if val:
            X_val = X_val.to_numpy()
            y_val = y_val.to_numpy()
        # Количество признаков в обучающей и валидационной выборке
        num_features_train = X_train.shape[1]
        num_features_val = X_val.shape[1] if val else 0
        # Обучение модели
        ret = model.fit(X_train, y_train, val, X_val, y_val)
        model_filename = 'uploads/mod.joblib'
        dump(model, model_filename)
        # Вывод параметров пользователя
        params_text = 'Алгоритм: '
        params_text += str(algorithm_name)
        params_text += ', Количество деревьев: '
        params_text += str(user_data['num_trees'])
        params_text += ', Максимальная глубина: '
        params_text += str(user_data['max_depth'])
        params_text += ', Размер подвыборки признаков: '
        params_text += str(user_data['subsample_size'])
        data_info_text = 'Признаков в обучающей выборке: '
        data_info_text += str(num_features_train)
        data_info_text += ', Признаков в валидационной выборке: '
        data_info_text += str(num_features_val)
        user_data['al_n'] = algorithm_name
        user_data['n_1'] = num_features_train
        user_data['n_2'] = num_features_val
        # Создание графика значений функций потерь с использованием Plotly
        fig = make_subplots(rows=1, cols=1, subplot_titles=["Training and Validation Loss"])
        fig.add_trace(go.Scatter(x=list(range(len(ret[0]))), y=ret[0], mode='lines', name='Training Loss'))
        if ret[1] is not None:
            fig.add_trace(go.Scatter(x=list(range(len(ret[1]))), y=ret[1], mode='lines', name='Validation Loss'))
        fig.update_layout(title_text='Training and Validation Loss', xaxis_title='Iteration', yaxis_title='Loss')
        plot_html = fig.to_html(full_html=False)
        user_data['loss_plot'] = plot_html
        return render_template('model_trained.html', params=params_text, data_info=data_info_text)
    except Exception as e:
        return f"Error during model training: {str(e)}"


@app.route('/download_file/<filename>')
def download_file(filename):
    return send_file(f'uploads/{filename}', as_attachment=True)


@app.route('/view_loss_plot')
def view_loss_plot():
    # Отображение графика на веб-странице
    plot_html = user_data.get('loss_plot')
    if plot_html:
        return plot_html
    else:
        return "Loss plot not available"


@app.route('/download_loss_plot')
def download_loss_plot():
    plot_html = user_data.get('loss_plot')
    if plot_html:
        return plot_html
    else:
        return "Loss plot not available"
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = load_model()
        file = request.files['predict_file']
        filename = os.path.join('uploads', 'predict_data.csv')
        file.save(filename)
        predict_data = pd.read_csv(filename)
        if 'date' in predict_data.columns:
            predict_data = predict_data.drop(columns=['date'])
        # One-Hot Encoding для категориальных признаков
        categorical_cols = predict_data.select_dtypes(include=['object']).columns
        predict_data = pd.get_dummies(predict_data, columns=categorical_cols, drop_first=True)
        predictions = model.predict(predict_data.to_numpy())
        predictions_df = pd.DataFrame(predictions, columns=['predictions'])
        predictions_filename = os.path.join('uploads', 'predictions.csv')
        predictions_df.to_csv(predictions_filename, index=False)
        return render_template('model_trained.html',
                               params="Модель успешно обучена",
                               predictions_file=predictions_filename,
                               loss_plot=user_data.get('loss_plot'))
    except Exception as e:
        return f"Ошибка во время предсказания: {str(e)}"


def load_model():
    model = load('uploads/mod.joblib')
    return model

@app.route('/reset')
def reset():
    user_data.clear()
    # Удаление обучающих данных
    training_data_path = 'uploads/training_data.csv'
    if os.path.exists(training_data_path):
        os.remove(training_data_path)
    # Перенаправление на главную страницу
    return render_template('index.html')


@app.route('/make_prediction', methods=['GET', 'POST'])
def make_prediction():
    if request.method == 'GET':
        return render_template('make_prediction.html')
    elif request.method == 'POST':
        try:
            model = load_model()
            file = request.files['predict_file']
            filename = os.path.join('uploads', 'predict_data.csv')
            file.save(filename)
            predict_data = pd.read_csv(filename)
            predict_data = predict_data.drop(columns=['date'])
            # One-Hot Encoding для категориальных признаков
            categorical_cols = predict_data.select_dtypes(include=['object']).columns
            predict_data = pd.get_dummies(predict_data, columns=categorical_cols, drop_first=True)
            predictions = model.predict(predict_data.to_numpy())
            predictions_df = pd.DataFrame(predictions, columns=['predictions'])
            predictions_filename = os.path.join('uploads', 'predictions.csv')
            predictions_df.to_csv(predictions_filename, index=False)
            return render_template('predictions.html', predictions_file=predictions_filename)
        except Exception as e:
            return f"Error during prediction: {str(e)}"


@app.route('/make_test_predictions', methods=['POST'])
def make_test_predictions():
    try:
        model = load_model()
        file = request.files['test_file']
        filename = os.path.join('uploads', 'test_data.csv')
        file.save(filename)
        test_data = pd.read_csv(filename)
        test_data = test_data.drop(columns=['date'])
        # One-Hot Encoding для категориальных признаков
        categorical_cols = test_data.select_dtypes(include=['object']).columns
        test_data = pd.get_dummies(test_data, columns=categorical_cols, drop_first=True)
        predictions = model.predict(test_data.to_numpy())
        predictions_df = pd.DataFrame(predictions, columns=['predictions'])
        predictions_filename = os.path.join('uploads', 'test_predictions.csv')
        predictions_df.to_csv(predictions_filename, index=False)
        return render_template('test_predictions.html', test_predictions_file=predictions_filename)
    except Exception as e:
        return f"Error during test predictions: {str(e)}"
    
@app.route('/download_predictions/<filename>')
def download_predictions(filename):
    return send_from_directory('uploads', filename, as_attachment=True)


@app.route('/model_trained')
def model_trained():
    params_text = 'Алгоритм: '
    params_text += str(user_data['al_n'])
    params_text += ', Количество деревьев: '
    params_text += str(user_data['num_trees'])
    params_text += ', Максимальная глубина: '
    params_text += str(user_data['max_depth'])
    params_text += ', Размер подвыборки признаков: '
    params_text += str(user_data['subsample_size'])
    data_info_text = 'Признаков в обучающей выборке: '
    data_info_text += str(user_data['num_trees'])
    data_info_text += ', Признаков в валидационной выборке: '
    data_info_text += str(user_data['n_2'])
    return render_template('model_trained.html', params=params_text, data_info=data_info_text)


@app.route('/download_training_data')
def download_training_data():
    filename = 'training_data.csv'
    return send_file(f'uploads/{filename}', as_attachment=True)
