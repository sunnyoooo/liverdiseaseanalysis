import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def process_csv(filepath):
    data = pd.read_csv(filepath)

    data.head()

    # Display data info (removing extra space before "info")
    data.info()

    # Record count calculations
    n_records = len(data)
    n_records_liv_pos = len(data[data['Dataset'] == 1])
    n_records_liv_neg = len(data[data['Dataset'] == 2])
    percent_liver_disease_pos = (n_records_liv_pos / n_records) * 100

    # Print the results
    print("Number of records: {}".format(n_records))
    print("Number of patients likely to have liver disease: {}".format(n_records_liv_pos))
    print("Number of patients unlikely to have liver disease: {}".format(n_records_liv_neg))
    print("Percentage of patients likely to have liver disease: {:.2f}%".format(percent_liver_disease_pos))

    # Visualizing features for each category of people (healthy/unhealthy)
    data1 = data[data['Dataset'] == 2].iloc[:, :-1]  # no disease
    data2 = data[data['Dataset'] == 1].iloc[:, :-1]  # with disease

    # Figure setup
    fig = plt.figure(figsize=(10, 15))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax1.grid()
    ax2.grid()

    # Titles and labels
    ax1.set_title('Features vs Mean Values', fontsize=13, weight='bold')
    ax1.text(0.5, 0.8, 'NO DISEASE', fontsize=20, ha='center', color='green', weight='bold', transform=ax1.transAxes)
    ax2.set_title('Features vs Mean Values', fontsize=13, weight='bold')
    ax2.text(0.5, 0.8, 'DISEASE', fontsize=20, ha='center', color='red', weight='bold', transform=ax2.transAxes)

    # Axes formatting for ax1 and ax2
    plt.sca(ax1)
    plt.xticks(rotation=0, weight='bold', family='monospace', size='large')
    plt.yticks(weight='bold', family='monospace', size='large')

    plt.sca(ax2)
    plt.xticks(rotation=0, weight='bold', family='monospace', size='large')
    plt.yticks(weight='bold', family='monospace', size='large')

    # Plotting bar plots (correcting palette parameter name)
    sns.barplot(data=data1, ax=ax1, orient='h', palette='bright')  # no disease
    sns.barplot(data=data2, ax=ax2, orient='h', palette='bright', saturation=0.8)  # with disease
    plt.show()

    # Visualizing differences in chemicals in Healthy/Unhealthy people
    with_disease = data[data['Dataset'] == 1].drop(columns=['Gender', 'Age', 'Dataset'])
    names1 = with_disease.columns.unique()
    mean_of_features1 = with_disease.mean(axis=0, skipna=True)

    without_disease = data[data['Dataset'] == 2].drop(columns=['Gender', 'Age', 'Dataset'])
    names2 = without_disease.columns.unique()
    mean_of_features2 = without_disease.mean(axis=0, skipna=True)

    # Creating DataFrame for plotting
    people = []
    for x, y in zip(names1, mean_of_features1):
        people.append([x, y, 'Diseased'])
    for x, y in zip(names2, mean_of_features2):
        people.append([x, y, 'Healthy'])

    new_data = pd.DataFrame(people, columns=['Chemicals', 'Mean_Values', 'Status'])

    # Plotting Comparison - Diseased vs Healthy
    fig = plt.figure(figsize=(20, 8))
    plt.title('Comparison - Diseased vs Healthy', size=20, loc='center')
    plt.xticks(rotation=30, weight='bold', family='monospace', size='large')
    plt.yticks(weight='bold', family='monospace', size='large')
    g1 = sns.barplot(x='Chemicals', y='Mean_Values', hue='Status', data=new_data, palette="RdPu_r")
    plt.legend(prop={'size': 20})
    plt.xlabel('Chemicals', size=19)
    plt.ylabel('Mean Values', size=19)
    plt.show()

    # Missing command to display data
    new_data.head()

    # Percentage of Chemicals in Unhealthy People
    with_disease = data[data['Dataset'] == 1].drop(columns=['Dataset', 'Gender', 'Age'])
    names = with_disease.columns.unique()
    mean_of_features = with_disease.mean(axis=0, skipna=True)

    # Example list of chemicals and means for pie chart
    list_names = ['Total_Bilirubin', 'Alkaline_Phosphotase', 'Direct_Bilirubin', 'Albumin',
                  'Alamine_Aminotransferase', 'Total_Protiens', 'Aspartate_Aminotransferase',
                  'Albumin_and_Globulin_Ratio']
    list_means = [4.164, 319.01, 1.924, 3.06, 99.61, 6.46, 137.7, 0.914]


    # Dictionary for chemicals and means
    mydict = {}
    l_names = []
    l_means = []
    for x, y in zip(names, mean_of_features):
        mydict[x] = y
        l_names.append(x)
        l_means.append(y)


    # Other statistics of dataset
    fig = plt.figure(figsize=(7, 5))
    plt.title('Percentage of Chemicals in Unhealthy People', size=14, color='#016450')

    # Create a pie chart
    plt.axis('equal')
    explode = (0.09,) * len(list_means)
    color_pink = ['#7a0177', '#ae017e', '#dd3497', '#f768a1', '#fa9fb5', '#fcc5c0', '#fde0dd', '#fff7f3']

    wedges, texts, autotexts = plt.pie(
        list_means,
        explode=explode,
        labels=list_names,
        labeldistance=1,
        textprops=dict(color='k'),
        radius=1.5,
        autopct="%1.1f%%",
        pctdistance=0.7,
        wedgeprops={'linewidth': 3, 'edgecolor': 'white'}
    )

    plt.setp(autotexts, size=12)
    plt.setp(texts, size=9)

    # Add a circle at the center
    my_circle = plt.Circle((0, 0), 1, color='white')
    p = plt.gcf()  # get current figure reference
    p.gca().add_artist(my_circle)  # get current axes
    plt.show()

    # Male vs Female statistics

    fig = plt.figure(figsize=(15, 6), frameon=False)
    plt.title("Total Data", loc='center', weight='bold', size=15)
    plt.xticks([])  # to disable xticks
    plt.yticks([])  # to disable yticks

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    only_gender = data['Gender']
    male_tot = only_gender[only_gender == 'Male']
    no_of_male = len(male_tot)
    no_of_female = len(data) - len(male_tot)
    m_vs_f = [no_of_male, no_of_female]

    # Diseased and Not Diseased statistics
    with_disease = data[data['Dataset'] == 1]
    not_with_disease = data[data['Dataset'] == 2]
    no_of_diseased = len(with_disease)
    no_of_not_diseased = len(not_with_disease)
    d_vs_healthy = [no_of_diseased, no_of_not_diseased]

    ax1.axis('equal')
    wedges, texts, autotexts = ax1.pie(
        m_vs_f,
        labels=('Male', 'Female'),
        radius=1,
        textprops=dict(color='k'),
        colors=['xkcd:ocean blue', 'xkcd:dark pink'],
        autopct="%1.1f%%"
    )

    ax2.axis('equal')
    wedges2, texts2, autotexts2 = ax2.pie(
        d_vs_healthy,
        labels=('Diseased', 'Not Diseased'),
        radius=1,
        textprops=dict(color='k'),
        colors=['#d95f02', '#1b9e77'],
        autopct="%1.1f%%"
    )
    plt.setp(autotexts, size=20)
    plt.setp(texts, size=20)
    plt.setp(autotexts2, size=20)
    plt.setp(texts2, size=20)
    plt.show()

    with_disease = data[data['Dataset'] == 1]
    not_with_disease = data[data['Dataset'] == 2]

    print("With Disease Data:")
    print(with_disease.head())
    print("Not With Disease Data:")
    print(not_with_disease.head())

    with_disease_m = with_disease[with_disease['Gender'] == 'Male']
    not_with_disease_m = not_with_disease[not_with_disease['Gender'] == 'Male']
    with_disease_f = with_disease[with_disease['Gender'] == 'Female']
    not_with_disease_f = not_with_disease[not_with_disease['Gender'] == 'Female']

    print("With Disease Male Data:")
    print(with_disease_m.head())
    print("Not With Disease Male Data:")
    print(not_with_disease_m.head())
    print("With Disease Female Data:")
    print(with_disease_f.head())
    print("Not With Disease Female Data:")
    print(not_with_disease_f.head())

    # Count the number of diseased and non-diseased individuals in each group
    no_of_diseased_m = len(with_disease_m)
    no_of_not_diseased_m = len(not_with_disease_m)
    no_of_diseased_f = len(with_disease_f)
    no_of_not_diseased_f = len(not_with_disease_f)

    print(f"Male Diseased: {no_of_diseased_m}, Male Not Diseased: {no_of_not_diseased_m}")
    print(f"Female Diseased: {no_of_diseased_f}, Female Not Diseased: {no_of_not_diseased_f}")

    # Pie chart data for males and females
    d_vs_healthy_m = [no_of_diseased_m, no_of_not_diseased_m]
    d_vs_healthy_f = [no_of_diseased_f, no_of_not_diseased_f]

    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Ensure equal aspect ratio for pie charts
    ax1.axis('equal')
    ax2.axis('equal')

    # Plot pie charts for Male and Female disease status
    wedges, texts, autotexts = ax1.pie(
        d_vs_healthy_m,
        labels=('Diseased', 'Not Diseased'),
        radius=1,
        textprops=dict(color='k'),
        colors=['#f46d43', '#4575b4'],
        autopct="%1.1f%%"
    )

    wedges2, texts2, autotexts2 = ax2.pie(
        d_vs_healthy_f,
        labels=('Diseased', 'Not Diseased'),
        radius=1,
        textprops=dict(color='k'),
        colors=['#f46d43', '#4575b4'],
        autopct="%1.1f%%"
    )

    # Adjust text size
    plt.setp(autotexts, size=20)
    plt.setp(texts, size=20)
    plt.setp(autotexts2, size=20)
    plt.setp(texts2, size=20)

    # Add labels for the charts
    ax1.text(0, 0.04, 'Male', size=20, color='#f7fcfd', horizontalalignment='center', weight='bold')
    ax2.text(0, 0.04, 'Female', size=20, color='#f7fcfd', horizontalalignment='center', weight='bold')

    # Show the plot
    plt.show()

    # Machine Learning
    # Separate the target values from the rest of the table
    X = data.iloc[:, :-1].values
    t = data.iloc[:, -1].values

    # Label encoding for target values
    for u in range(len(t)):
        if t[u] == 2:
            t[u] = 0

    # Encoding 'Gender' column into numerical values
    from sklearn.preprocessing import LabelEncoder

    lbl = LabelEncoder()
    X[:, 1] = lbl.fit_transform(X[:, 1])

    # Fill missing values in 'Albumin_and_Globulin_Ratio' column with mean
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X[:, 9:10] = imputer.fit_transform(X[:, 9:10])

    # Training and Testing data
    from sklearn.model_selection import train_test_split

    X_train, X_test, t_train, t_test = train_test_split(X, t, random_state=99, test_size=0.05)

    # Feature Scaling (excluding 'Age' and 'Gender')
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train[:, 2:] = sc.fit_transform(X_train[:, 2:])
    X_test[:, 2:] = sc.transform(X_test[:, 2:])

    # Model Evaluation metrics
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # Logistic Regression
    from sklearn.linear_model import LogisticRegression

    classifier_logis = LogisticRegression(random_state=100)
    classifier_logis.fit(X_train, t_train)
    y_pred_logis = classifier_logis.predict(X_test)
    cm_logis = confusion_matrix(t_test, y_pred_logis)
    print(cm_logis)
    accuracy_logis = accuracy_score(t_test, y_pred_logis)
    print(f'The accuracy of LogisticRegression is: {accuracy_logis * 100:.2f}%')
    logreg_score = round(classifier_logis.score(X_train, t_train) * 100, 2)
    print(f'Logistic Regression Training Score is: {logreg_score}')
    print(classification_report(t_test, y_pred_logis))

    # Support Vector Machine
    from sklearn.svm import SVC

    classifier_svc = SVC(kernel='rbf', random_state=1234, gamma='auto')
    classifier_svc.fit(X_train, t_train)
    y_pred_svc = classifier_svc.predict(X_test)
    cm_svc = confusion_matrix(t_test, y_pred_svc)
    print(cm_svc)
    accuracy_svc = accuracy_score(t_test, y_pred_svc)
    print(f'The accuracy of Support Vector Classification is: {accuracy_svc * 100:.2f}%')
    svc_score = round(classifier_svc.score(X_train, t_train) * 100, 2)
    print(f'Support Vector Machine Training Score is: {svc_score}')
    print(classification_report(t_test, y_pred_svc))

    # Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier

    classifier_rfc = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=100)
    classifier_rfc.fit(X_train, t_train)
    y_pred_rfc = classifier_rfc.predict(X_test)
    cm_rfc = confusion_matrix(t_test, y_pred_rfc)
    print(cm_rfc)
    accuracy_rfc = accuracy_score(t_test, y_pred_rfc)
    print(f'The accuracy of RandomForestClassifier is: {accuracy_rfc * 100:.2f}%')
    rfc_score = round(classifier_rfc.score(X_train, t_train) * 100, 2)
    print(f'Random Forest Training Score is: {rfc_score}')
    print(classification_report(t_test, y_pred_rfc))

    # Model Comparison
    # Model Comparison Data for Accuracy
    models_comparison = [
        ['Logistic Regression', accuracy_logis * 100],
        ['Support Vector Classification', accuracy_svc * 100],
        ['Random Forest Classification', accuracy_rfc * 100]
    ]

    # Convert to DataFrame for easier plotting
    models_comparison_df = pd.DataFrame(models_comparison, columns=['Model', '% Accuracy'])

    # Plotting the model comparison bar chart for Accuracy
    fig = plt.figure(figsize=(20, 8))
    sns.set()
    sns.barplot(x='Model', y='% Accuracy', data=models_comparison_df, palette='Dark2')

    # Customize labels and titles
    plt.xticks(size=18)
    plt.ylabel('% Accuracy', size=14)
    plt.xlabel('Model', size=14)
    plt.title('Model Accuracy Comparison', size=18)
    plt.show()

    # Model Comparison Data for Training Scores
    models_training_scores = [
        ['Logistic Regression', logreg_score],
        ['Support Vector Classification', svc_score],
        ['Random Forest Classification', rfc_score]
    ]

    # Convert to DataFrame for easier plotting
    models_training_scores_df = pd.DataFrame(models_training_scores, columns=['Model', 'Training Score'])

    # Plotting the model comparison bar chart for Training Scores
    fig = plt.figure(figsize=(20, 8))
    sns.set()
    sns.barplot(x='Model', y='Training Score', data=models_training_scores_df, palette='Set1')

    # Customize labels and titles
    plt.xticks(size=18)
    plt.ylabel('Training Score', size=14)
    plt.xlabel('Model', size=14)
    plt.title('Model Training Score Comparison', size=18)
    plt.show()

    # Analyzing effect of number of trees in Random Forest (training vs test accuracy)
    n_trees = [10, 50, 100, 200, 250, 400, 500, 1000, 1500, 2000, 2500]  # Define n_trees here

    def n_trees_acc(n):
        classifier_rfc = RandomForestClassifier(n_estimators=n, criterion='entropy', random_state=100)
        classifier_rfc.fit(X_train, t_train)
        train_acc = accuracy_score(t_train, classifier_rfc.predict(X_train)) * 100
        test_acc = accuracy_score(t_test, classifier_rfc.predict(X_test)) * 100
        return train_acc, test_acc

    train_accuracies = []
    test_accuracies = []

    for n in n_trees:
        train_acc, test_acc = n_trees_acc(n)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    d1 = pd.DataFrame({
        'no. of trees in forest': n_trees,
        'train accuracy': train_accuracies,
        'test accuracy': test_accuracies
    })

    # Plotting
    fig = plt.figure(figsize=(20, 6))
    sns.lineplot(x='no. of trees in forest', y='train accuracy', data=d1, label='Training Accuracy', color='blue')
    sns.lineplot(x='no. of trees in forest', y='test accuracy', data=d1, label='Test Accuracy', color='red')
    plt.title('Learning Curve: Trees in Forest vs Accuracy', size=18)
    plt.xlabel('no. of trees in forest', size=15)
    plt.ylabel('Accuracy', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()

def manual_entry_predict(classifier, scaler, label_encoder):
    """
    Collects manual details of a single person and predicts liver health.
    """
    age = float(input("Enter Age: "))
    gender = input("Enter Gender (Male/Female): ").strip().capitalize()
    while gender not in ["Male", "Female"]:
        print("Invalid input. Please enter 'Male' or 'Female'.")
        gender = input("Enter Gender (Male/Female): ").strip().capitalize()
    total_bilirubin = float(input("Enter Total Bilirubin: "))
    direct_bilirubin = float(input("Enter Direct Bilirubin: "))
    alkaline_phosphotase = float(input("Enter Alkaline Phosphotase: "))
    alamine_aminotransferase = float(input("Enter Alamine Aminotransferase: "))
    aspartate_aminotransferase = float(input("Enter Aspartate Aminotransferase: "))
    total_proteins = float(input("Enter Total Proteins: "))
    albumin = float(input("Enter Albumin: "))
    albumin_and_globulin_ratio = float(input("Enter Albumin and Globulin Ratio: "))

    # Encode and scale input
    gender_encoded = label_encoder.transform([gender])[0]
    user_input = np.array([[age, gender_encoded, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                            alamine_aminotransferase, aspartate_aminotransferase, total_proteins, albumin,
                            albumin_and_globulin_ratio]])
    user_input[:, 2:] = scaler.transform(user_input[:, 2:])

    # Predict and display result
    prediction = classifier.predict(user_input)

    # Display result
    if prediction[0] == 1:
        print("Prediction: The patient is likely to have liver disease.")
    else:
        print("Prediction: The patient is unlikely to have liver disease.")
def main():
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression

    # Preprocess data for model training
    filepath = "enter training data file path here"
    data = pd.read_csv(filepath)

    # Preprocessing
    X = data.iloc[:, :-1].values
    t = data.iloc[:, -1].values
    t = np.where(t == 2, 0, 1)
    lbl = LabelEncoder()
    X[:, 1] = lbl.fit_transform(X[:, 1])
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X[:, 9:10] = imputer.fit_transform(X[:, 9:10])
    X_train, X_test, t_train, t_test = train_test_split(X, t, random_state=99, test_size=0.05)
    sc = StandardScaler()
    X_train[:, 2:] = sc.fit_transform(X_train[:, 2:])
    X_test[:, 2:] = sc.transform(X_test[:, 2:])

    # Train model
    classifier = LogisticRegression(random_state=100)
    classifier.fit(X_train, t_train)

    # User choice loop
    while True:
        print("Choose an option:")
        print("1. Enter details manually")
        print("2. Upload a CSV file")
        choice = input("Enter your choice: ")

        if choice == '1':
            manual_entry_predict(classifier, sc, lbl)
        elif choice == '2':
            filepath = input("Enter the path to your CSV file: ")
            process_csv(filepath)
        else:
            print("Invalid choice. Please try again.")

        # Ask if the user wants to exit or continue
        continue_choice = input("Do you want to continue? (yes/no): ").strip().lower()
        if continue_choice != 'yes':
            break

if __name__ == "__main__":
    main()
