import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score, average_precision_score, roc_auc_score, precision_recall_curve, roc_curve
# Крок 1
file_name = 'KM-12-2.csv' 
df = pd.read_csv(file_name) 

# Крок 2
count_class = df['GT'].value_counts()
print(count_class)
# вибірка збалансована, оскільки кількість об'єктів в кожному класі однакова

# Крок 3
# k = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  
k = np.arange(0, 1.01, 0.01)

# Функція для визначення значень метрик 
def metrics(df, model):
    metrics_dict = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-score': [],
        'MCC': [],
        'Balanced Accuracy': [],
        'AUC for Precision-Recall Curve': [],
        'AUC for ROC Curve': [],
        "Youden's J statistic": []
    }

    for threshold in k:
        model_pred = (df[model] >= threshold).astype(int)
        accuracy = accuracy_score(df['GT'], model_pred)
        precision = precision_score(df['GT'], model_pred, zero_division=1)
        recall = recall_score(df['GT'], model_pred, zero_division=1)
        f1 = f1_score(df['GT'], model_pred)
        mcc = matthews_corrcoef(df['GT'], model_pred)
        balanced_accuracy = balanced_accuracy_score(df['GT'], model_pred)
        auc_prc = average_precision_score(df['GT'], model_pred)
        auc_roc = roc_auc_score(df['GT'], model_pred)
        j_statistic = 2 * balanced_accuracy - 1
        # print(f"Поріг: {threshold}, Модель: {model}, Accuracy: {accuracy}, Balanced Accuracy: {balanced_accuracy}, \nPrecision: {precision}, Recall: {recall}, F-score: {f1}, \nMCC: {mcc}, AUC for Precision-Recall Curve: {auc_prc}, AUC for ROC Curve: {auc_roc},  \nYouden's J statistic: {j_statistic}\n")            
        metrics_dict['Accuracy'].append(accuracy)
        metrics_dict['Precision'].append(precision)
        metrics_dict['Recall'].append(recall)
        metrics_dict['F1-score'].append(f1)
        metrics_dict['MCC'].append(mcc)
        metrics_dict['Balanced Accuracy'].append(balanced_accuracy)
        metrics_dict['AUC for Precision-Recall Curve'].append(auc_prc)
        metrics_dict['AUC for ROC Curve'].append(auc_roc)
        metrics_dict["Youden's J statistic"].append(j_statistic)

    return metrics_dict

# Функція для побудови графіка метрик відносно значень порогу    
def plot_metrics(metrics_dict):    
    plt.figure(figsize=(10, 8))

    for metric, values in metrics_dict.items():
        max_value = max(values)
        max_index = values.index(max_value)
        max_threshold = k[max_index]

        plt.plot(k, values, label=metric)
        plt.scatter(max_threshold, max_value, marker='*', s=100, label=f'Max {metric}')

    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Metrics vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pr_curve(df, model):
    precision, recall, thresholds = precision_recall_curve(df['GT'], df[model])
    x = np.linspace(0, 1, 100)
    y = x
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='PR Curve')
    plt.plot(x, y, linestyle='--', color='gray', label='Diagonal')  # Діагональ
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model}')
    plt.grid(True)
    plt.legend()
    
    # Інтерполяція для знаходження точки перетину
    recall_interp = np.interp(x, recall[::-1], precision[::-1])
    diff = np.abs(recall_interp - y)
    intersection_idx = np.argmin(diff)
    intersection_x = x[intersection_idx]
    intersection_y = y[intersection_idx]

    plt.scatter(intersection_x, intersection_y, color='red', marker='o')
    plt.show()

def plot_roc_curve(df, model):
    fpr, tpr, thresholds = roc_curve(df['GT'], df[model])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model}')
    plt.legend()
    plt.grid(True)
    
    # Оптимальний поріг (найближча точка до (0,1))
    distances = np.sqrt(fpr**2 + (1 - tpr)**2)
    optimal_idx = np.argmin(distances)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    
    plt.scatter(optimal_fpr, optimal_tpr, color='red', marker='o', label='Optimal Threshold')
    plt.legend()
    plt.show()

# 
def plot_hist_with_thresholds(df, model_prob, metrics_dict):
    plt.figure(figsize=(12, 6))

    # Побудова накладених гістограм для кожного класу
    for class_label in [0, 1]:
        plt.hist(df[df['GT'] == class_label][model_prob], bins=50, alpha=0.5, label=f'Class {class_label}', density=True)
    
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.title(f'{model_prob} - Histogram with Optimal Thresholds')
    plt.legend()
    plt.grid(True)

    # Додавання вертикальних ліній для оптимальних порогів
    for metric, values in metrics_dict.items():
        max_value = max(values)
        max_index = values.index(max_value)
        max_threshold = k[max_index]
        plt.axvline(x=max_threshold, linestyle='--', label=f'{metric} Optimal', alpha=0.7)

    plt.legend()
    plt.tight_layout()
    plt.show()


model_1_metrics = metrics(df, 'Model_1_1')
model_2_metrics = metrics(df, 'Model_2_1')

plot_metrics(model_1_metrics)
plot_metrics(model_2_metrics)

plot_pr_curve(df, 'Model_1_1')
plot_pr_curve(df, 'Model_2_1')

plot_roc_curve(df, 'Model_1_1')
plot_roc_curve(df, 'Model_2_1')

plot_hist_with_thresholds(df, 'Model_1_1', model_1_metrics)
plot_hist_with_thresholds(df, 'Model_2_1', model_2_metrics)

# Крок 5
birth_date = '14-12'  
day, month = map(int, birth_date.split('-'))

K = day % 9

class_1_count = df[df['GT'] == 1].shape[0]
remove_percentage = 50 + 5 * K
remove_count = int((remove_percentage / 100) * class_1_count)

df_class_1 = df[df['GT'] == 1].sample(frac=1, random_state=42)
df_class_1_to_remove = df_class_1.head(remove_count)
df_new = df.drop(df_class_1_to_remove.index)

# Крок 6
new_count_class_1 = df_new["GT"].value_counts()
print(f'Відсоток видалених об\'єктів класу 1: {remove_percentage}%')
print(f'Кількість об\'єктів кожного класу після видалення:\n{new_count_class_1}')

# Крок 7
model_1_metrics_new = metrics(df_new, 'Model_1_1')
model_2_metrics_new = metrics(df_new, 'Model_2_1')

plot_metrics(model_1_metrics_new)
plot_metrics(model_2_metrics_new)

plot_pr_curve(df_new, 'Model_1_1')
plot_pr_curve(df_new, 'Model_2_1')

plot_roc_curve(df_new, 'Model_1_1')
plot_roc_curve(df_new, 'Model_2_1')

plot_hist_with_thresholds(df_new, 'Model_1_1', model_1_metrics_new)
plot_hist_with_thresholds(df_new, 'Model_2_1', model_2_metrics_new)