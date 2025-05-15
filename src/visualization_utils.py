import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import plotly.express as px

def plot_feature_distribution(train, test, feature):
    train_data = train[feature].dropna()
    test_data = test[feature].dropna()

    kde_train = gaussian_kde(train_data)
    kde_test = gaussian_kde(test_data)

    x_min = min(train_data.min(), test_data.min())
    x_max = max(train_data.max(), test_data.max())
    x_vals = np.linspace(x_min, x_max, 500)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=kde_train(x_vals),
        mode='lines', name='train',
        fill='tozeroy'
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=kde_test(x_vals),
        mode='lines', name='test',
        fill='tozeroy'
    ))
    fig.update_layout(
        title=f"Distribution of {feature}",
        xaxis_title=feature,
        yaxis_title='density',
        template='plotly_white',
        width=800,
        height= 500,
    )
    fig.show()

def generate_hist(data, x_title):
    histogram = go.Histogram(x=data)
    layout = go.Layout(
        title=f"Histogram of {x_title}",
        xaxis=dict(title=x_title, range=[min(data), max(data)]),
        yaxis=dict(title='count'),
        width=800,
        height= 500,
        )
    fig = go.Figure(data=[histogram], layout=layout)
    fig.show()

def prepare_combined_data(train_before, train_after, test_before, test_after, column):
    train_before_copy = train_before[[column]].copy()
    train_before_copy['dataset'] = 'train'
    train_before_copy['imputation'] = 'before'

    train_after_copy = train_after[[column]].copy()
    train_after_copy['dataset'] = 'train'
    train_after_copy['imputation'] = 'after'

    test_before_copy = test_before[[column]].copy()
    test_before_copy['dataset'] = 'test'
    test_before_copy['imputation'] = 'before'

    test_after_copy = test_after[[column]].copy()
    test_after_copy['dataset'] = 'test'
    test_after_copy['imputation'] = 'after'

    combined = pd.concat([
        train_before_copy,
        train_after_copy,
        test_before_copy,
        test_after_copy], axis=0)

    return combined

def plot_before_after_imputation(train_before, train_after, test_before, test_after, column):
    combined = prepare_combined_data(train_before, train_after, test_before, test_after, column)

    fig = px.box(
        combined,
        x="dataset",
        y=column,
        color="imputation",
        title=f"Boxplot of {column} before and after imputation",
        labels={column: column},
        points="outliers"
    )

    fig.update_layout(boxmode="group")
    fig.show()

def plot_correlation(df, features):
    fig, axs = plt.subplots(figsize=(12, 10))
    sns.heatmap(df[features].corr(), annot=True, cmap=sns.cubehelix_palette(start=0.1, rot=-0.5, as_cmap=True))
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.title('Correlation between features')
    plt.show()

def plot_results_compiration(y_pred, y_val):
    percent_error = abs(y_pred - y_val)/(abs(y_val))*100
    df = pd.DataFrame({
        'y_true': y_val,
        'y_pred': y_pred,
        'percent_error': percent_error
    })

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=df['y_true'], nbinsx=50, name='true', opacity=0.6))
    fig_hist.add_trace(go.Histogram(x=df['y_pred'], nbinsx=50, name='predicted', opacity=0.6))

    fig_hist.update_layout(
        barmode='overlay',
        title='Histogram of true and predicted data',
        xaxis_title='ActualTOW [kg]',
        yaxis_title='count',
        width=800,
        height= 500,
    )

    fig_box = px.box(df, y='percent_error')
    fig_box.update_layout(
        title='Boxplot of percent error',
        xaxis_title='ActualTOW [kg]',
        yaxis_title='error [%]',
        width=500,
        height=600,
    )
    fig_hist.show()
    fig_box.show()
