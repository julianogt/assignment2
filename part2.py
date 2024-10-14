import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
import mlrose_hiive as mlrose
from timeit import default_timer as timer

shopping_data = pd.read_csv('shopping_behavior.csv')

frequency_mapping = {
    'Annually': 1,
    'Monthly': 12,
    'Weekly': 52,
    'Fortnightly': 26,
    'Quarterly': 4,
    'Every 3 Months': 4,
    'Bi-Weekly': 26
}

shopping_data['Frequency of Purchases'] = shopping_data['Frequency of Purchases'].map(frequency_mapping)
y_shopping = shopping_data['Frequency of Purchases']

# Calculate the average of 'Frequency of Purchases'
average_frequency = y_shopping.mean()

# Convert 'Frequency of Purchases' to binary labels
y_shopping_binary = np.where(y_shopping > average_frequency, 1, 0)

# Select features and preprocess them
X_shopping = shopping_data[['Customer ID', 'Item Purchased', 'Age', 'Purchase Amount (USD)', 
                             'Review Rating', 'Gender', 'Category', 'Season', 'Shipping Type']]
preprocessor_shopping = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'Purchase Amount (USD)', 'Review Rating']),
        ('cat', OneHotEncoder(drop='first'), ['Gender', 'Category', 'Season', 'Shipping Type'])
    ]
)

X_shopping = preprocessor_shopping.fit_transform(X_shopping)

# Split the dataset into training, validation, and test sets
X_shop_train, X_temp, y_shop_train, y_temp = train_test_split(X_shopping, y_shopping_binary, test_size=0.3, random_state=42)
X_val, X_shop_test, y_val, y_shop_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

=print("Training data size:", X_shop_train.shape[0])

rhc_nn = mlrose.NeuralNetwork(
    hidden_nodes=[5],
    activation='tanh',
    algorithm='random_hill_climb',
    max_iters=1000,
    bias=True,
    is_classifier=True,
    learning_rate=0.01,
    early_stopping=True,
    clip_max=5,
    max_attempts=100,
    random_state=1,
    curve=True
)

# Train the RHC neural network
rhc_nn.fit(X_shop_train, y_shop_train)

# Make Predictions
y_pred_rhc = rhc_nn.predict(X_shop_test)

# Evaluate
print("Accuracy Test Score: ", accuracy_score(y_shop_test, y_pred_rhc))
print("Precision Score: ", precision_score(y_shop_test, y_pred_rhc, average='weighted'))
print("Recall Score: ", recall_score(y_shop_test, y_pred_rhc, average='weighted'))
print("F1 Score: ", f1_score(y_shop_test, y_pred_rhc, average='weighted'))
print('Confusion Matrix:\n', confusion_matrix(y_shop_test, y_pred_rhc))

cm = confusion_matrix(y_shop_test, y_pred_rhc)

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plot_confusion_matrix(cm, ['Below Average', 'Above Average'],
                      title='Confusion Matrix: RHC Neural Network')

# Plot fitness curve for RHC
plt.figure(figsize=(10, 6))
plt.plot(rhc_nn.fitness_curve, label='Fitness Curve - RHC', color='blue')
plt.title('Fitness Curve for RHC Neural Network')
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.grid()
plt.legend()
plt.show()

# Plot learning curves
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.close()
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot training scores
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="b")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.grid()

# Plot learning curve RHC
title = "Learning Curves - NN - RHC - Tuned NN"
cv = ShuffleSplit(n_splits=10, test_size=0.4, random_state=1)
plot_learning_curve(rhc_nn, title, X_shop_train, y_shop_train, ylim=(0.5, 1.01), cv=cv, n_jobs=-1)
plt.show()


# *********************************************** SA ************************************************************

# Define Simulated Annealing neural network
ann = mlrose.NeuralNetwork(
    hidden_nodes=[1],
    activation='tanh',
    algorithm='simulated_annealing',
    is_classifier=True,
    early_stopping=True,
    random_state=1,
    max_attempts=200,
    max_iters=5000,
    bias=True,
    learning_rate=.1,
    curve=True,
    schedule=mlrose.GeomDecay(init_temp=100.0, decay=0.95, min_temp=0.00001)
)

# Train
start = timer()
ann.fit(X_shop_train, y_shop_train)
end = timer()
print("Time for fit (SA): ", end - start)

# Make Predictions
y_pred_sa = ann.predict(X_shop_test)
y_pred_train_sa = ann.predict(X_shop_train)

# Evaluate
ann_acc = accuracy_score(y_shop_test, y_pred_sa)
ann_acc_train = accuracy_score(y_shop_train, y_pred_train_sa)
scores = cross_val_score(ann, X_shopping, y_shopping_binary, cv=5)

print("Accuracy Test Score (SA): ", ann_acc)
print("Accuracy Train Score (SA):", ann_acc_train)
print("Cross Value (SA): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("----")
print('Precision Score (SA): ' + str(precision_score(y_shop_test, y_pred_sa, average='weighted')))
print('Recall Score (SA): ' + str(recall_score(y_shop_test, y_pred_sa, average='weighted')))
print('F1 Score (SA): ' + str(f1_score(y_shop_test, y_pred_sa, average='weighted')))
print("----")
print('RMSE (SA): ' + str(mean_squared_error(y_shop_test, y_pred_sa)))
print("----")
print('Confusion Matrix (SA): \n' + str(confusion_matrix(y_shop_test, y_pred_sa)))

c_matrix_sa = confusion_matrix(y_shop_test, y_pred_sa)
plot_confusion_matrix(c_matrix_sa, ['Below Average', 'Above Average'],
                      title='Confusion Matrix: SA Neural Network')

# Plot learning curve for SA
title = "Learning Curves - NN - SA"
cv = ShuffleSplit(n_splits=10, test_size=0.4, random_state=1)
plot_learning_curve(ann, title, X_shop_train, y_shop_train, ylim=(0.5, 1.01), cv=cv, n_jobs=-1)
plt.grid()
plt.show()

# Plot fitness curve for SA
plt.figure(figsize=(10, 6))
plt.plot(ann.fitness_curve, label='Fitness Curve - SA', color='orange')
plt.title('Fitness Curve for SA Neural Network')
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.grid()
plt.legend()
plt.show()

# *********************************************** GA ************************************************************

ga_ann = mlrose.NeuralNetwork(
    hidden_nodes=[1],
    activation='tanh',
    algorithm='genetic_alg',
    is_classifier=True,
    early_stopping=True,
    random_state=1,
    pop_size=200,
    max_attempts=100,
    max_iters=5000,
    bias=True,
    learning_rate=0.3,
    mutation_prob=0.2,
    curve=True
)

import warnings
warnings.filterwarnings('ignore')

# Train
start = timer()
ga_ann.fit(X_shop_train, y_shop_train)
end = timer()
print("Time for fit (GA): ", end - start)

# Make Predictions
y_pred_ga = ga_ann.predict(X_shop_test)
y_pred_train_ga = ga_ann.predict(X_shop_train)

# Evaluate
ga_ann_acc = accuracy_score(y_shop_test, y_pred_ga)
ga_ann_acc_train = accuracy_score(y_shop_train, y_pred_train_ga)
scores = cross_val_score(ga_ann, X_shopping, y_shopping_binary, cv=5)

print("Accuracy Test Score (GA): ", ga_ann_acc)
print("Accuracy Train Score (GA):", ga_ann_acc_train)
print("Cross Value (GA): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("----")
print('Precision Score (GA): ' + str(precision_score(y_shop_test, y_pred_ga, average='weighted')))
print('Recall Score (GA): ' + str(recall_score(y_shop_test, y_pred_ga, average='weighted')))
print('F1 Score (GA): ' + str(f1_score(y_shop_test, y_pred_ga, average='weighted')))
print("----")
print('RMSE (GA): ' + str(mean_squared_error(y_shop_test, y_pred_ga)))
print("----")
print('Confusion Matrix (GA): \n' + str(confusion_matrix(y_shop_test, y_pred_ga)))

c_matrix_ga = confusion_matrix(y_shop_test, y_pred_ga)
plot_confusion_matrix(c_matrix_ga, ['Below Average', 'Above Average'],
                      title='Confusion Matrix: GA Neural Network')

# Plot learning curve for GA
title = "Learning Curves - NN - GA"
cv = ShuffleSplit(n_splits=10, test_size=0.4, random_state=1)
plot_learning_curve(ga_ann, title, X_shop_train, y_shop_train, ylim=(0.5, 1.01), cv=cv, n_jobs=-1)
plt.grid()
plt.show()

# Plot fitness curve for GA
plt.figure(figsize=(10, 6))
plt.plot(ga_ann.fitness_curve, label='Fitness Curve - GA', color='green')
plt.title('Fitness Curve for GA Neural Network')
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.grid()
plt.legend()
plt.show()

# *********************************************** Comparison ************************************************************

my_dict = dict(
    x=np.arange(1, len(ga_ann.fitness_curve) + 1),  # Ensure the x-axis matches the length of the fitness curves
    RHC=[fitness[0] for fitness in rhc_nn.fitness_curve],  # Access only the fitness value
    SA=[fitness[0] for fitness in ann.fitness_curve],       # Access only the fitness value
    GA=[fitness[0] for fitness in ga_ann.fitness_curve]     # Access only the fitness value
)

# Convert the dictionary into a DataFrame
df = pd.DataFrame.from_dict(my_dict, orient='index').transpose()
print(df.head())

# Plotting the fitness curves
plt.figure(figsize=(10, 6))
plt.plot('x', 'RHC', data=df, marker='', color='blue', linewidth=4, linestyle='-', label="RHC")
plt.plot('x', 'SA', data=df, marker='', color='black', linewidth=4, linestyle=':', label="SA")
plt.plot('x', 'GA', data=df, marker='', color='red', linewidth=4, linestyle='-', label="GA")

plt.xlim(0, 400) 
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.title('Neural Network Optimal Weights Fitness/Iteration of Algorithms')
plt.legend()
plt.grid()
plt.show()

