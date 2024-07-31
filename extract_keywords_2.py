from rake_nltk import Rake

# Define the list of sentences
sentences = [
    "Student retention and graduation prediction using machine learning",
    "Early identification of at-risk students",
    "Feature selection and dimensionality reduction techniques",
    "Predictive models for dropout analysis",
    "Predicting student retention and graduation status",
    "Factors affecting student retention, progression and graduation",
    "Predicting nursing student graduation outcomes",
    "Identifying struggling students early in the semester",
    "Predicting student dropout in Latin American universities",
    "Factors affecting undergraduate educational status",
    "Predicting student dropout using decision tree, random forest and gradient boosting",
    "Enrollment management and institutional research (EMIR)",
    "Predicting student semester dropout in Bangladesh",
    "Identifying students at risk of dropping out in UAE",
    "Dimensionality reduction using feature selection and extraction",
    "Overcoming imbalanced data using SMOTE",
    "Exam-taking behavior patterns and time management skills",
    "Predicting first-year failure at Czech Technical University",
    "Early diagnosis of high dropout risk students in Taiwan",
    "Variable selection for predicting student dropout",
    "Accuracy of decision tree, neural network and stacking models",
    "Extracting hidden patterns from student academic data",
    "Freshman student retention in higher education",
    "Predictive and cluster models for intervention programs",
    "Modified Mutated Firefly Algorithm (MMFA) for dimensionality reduction",
    "Predicting dropout of first-year undergraduate students",
    "Ensemble algorithm for retention and graduation prediction",
    "Factors predicting retention, progression and graduation in health majors",
    "Leveraging machine learning to identify variables affecting health students",
    "Stacked ensemble method to boost prediction accuracy",
    "Student Performance Prediction: The document discusses various methods to predict student performance, including using machine learning algorithms and analyzing student background factors.",
    "College Completion Intention (CCI): Thomas's study focused on understanding the relationship among factors that contribute to students' college completion intentions.",
    "Student Persistence: Factors such as pre-college experience, financial aid, and college academic performance were found to influence student persistence.",
    "Data Analysis: The document highlights the importance of data analysis in understanding student performance and dropout rates, including the use of various machine learning models.",
    "Machine Learning Models: Several machine learning models are discussed, including Multi-Layer Perceptron (MLP), Random Forest (RF), and Logistic Regression (LR).",
    "Data Imbalance: Techniques for handling class imbalance in datasets are discussed, including data sampling methods like Random Oversampling (ROS) and Synthetic Minority Over-sampling Technique (SMOTE).",
    "Causal Inference: The document emphasizes the use of causal inference methods to analyze the effects of various factors on student dropout and underperformance.",
    "Student Background Factors: Student background factors such as gender, age, and birthplace are considered in predicting student performance.",
    "Academic Performance: Previous academic performance and current academic factors are also included in the analysis.",
    "University Admittance: The document discusses the role of university admittance factors, including the year of admittance and achieved score.",
    "Student Demographics: Student demographics such as gender and age are considered in the analysis of student persistence and dropout rates.",
    "Data Reduction: Dimensional reduction techniques like Principal Component Analysis (PCA) are used to reduce the number of features in the dataset.",
    "Feature Extraction: Feature extraction methods are used to create new features from the original data.",
    "Feature Selection: Feature selection methods are used to select the most relevant features from the dataset.",
    "Classifiers: Various classifiers are discussed, including Logistic Regression (LR), K-Nearest Neighbor (KNN), C4.5, Naive Bayes (NB), and Support Vector Machines (SVM).",
    "Performance Measures: Standard measures of accuracy, recall, precision, and F-Measure are used to assess the performance of classifiers.",
    "Data Cleaning: Techniques for data cleaning, such as removing noisy features and handling missing values, are discussed.",
    "Data Sampling: Data sampling techniques like Random Oversampling (ROS) and Synthetic Minority Over-sampling Technique (SMOTE) are used to handle class imbalance.",
    "Cost-Sensitive Learning: Cost-sensitive learning techniques are used to minimize costs associated with the learning process.",
    "Artificial Neural Networks (ANNs): ANNs are used for classification problems due to their ability to handle nonlinear relationships.",
    "Multi-Layer Perceptron (MLP): MLP is used as a machine learning model for classification problems.",
    "Random Forest (RF): RF is used as a machine learning model for classification problems.",
    "Logistic Regression (LR): LR is used as a machine learning model for classification problems.",
    "Student Dropout: The document focuses on identifying students at risk of dropout and developing methods to predict and prevent dropout.",
    "Underperformance: The document discusses methods to identify students at risk of underperformance and develop strategies to improve their performance.",
    "Education Policy: The document highlights the need for educational policy makers to consider revising regulations that establish the minimum number of credits required for graduation.",
    "Higher Education: The document focuses on higher education, discussing issues related to student performance, dropout rates, and underperformance.",
    "Data Analysis in Education: The document emphasizes the importance of data analysis in education, highlighting the need for effective methods to analyze and predict student performance.",
    "Machine Learning in Education: The document discusses the application of machine learning in education, including the use of various machine learning models and techniques.",
    "Predictive Modeling: The document highlights the importance of predictive modeling in education, emphasizing the need for accurate predictions of student performance and dropout rates.",
    "Student Retention",
    "Student Dropout Prediction",
    "Educational Data Mining (EDM)",
    "Machine Learning in Education",
    "Prediction Models",
    "Logistic Regression",
    "Neural Networks (ANN)",
    "Support Vector Machines (SVM)",
    "Decision Trees",
    "Random Forests",
    "K-Nearest Neighbors (KNN)",
    "Naive Bayes",
    "Deep Learning",
    "Classification Algorithms",
    "Clustering Algorithms",
    "Data Preprocessing",
    "Feature Engineering",
    "Educational Analytics",
    "Predictive Modeling",
    "Academic Performance Prediction",
    "Student Behavior Analysis",
    "Early Warning Systems",
    "Intervention Strategies",
    "Attrition Rates",
    "Cognitive and Non-Cognitive Factors",
    "First-Year Student Performance",
    "Socioeconomic Factors",
    "Academic Self-Efficacy",
    "Enrollment Management",
    "Predictive Accuracy Metrics",
    "Machine Learning in Education: Methods and applications of machine learning to predict and enhance student performance.",
    "Student Dropout Prediction: Techniques for identifying students at risk of dropping out.",
    "Educational Data Mining (EDM): Use of data mining techniques to analyze educational data and improve learning outcomes.",
    "Learning Analytics (LA): Analysis of educational data to understand and optimize learning processes.",
    "Student Retention Strategies: Approaches to keep students enrolled and engaged in their studies.",
    "Performance Prediction: Methods for predicting student academic performance based on various indicators.",
    "Recommender Systems in Education: Systems designed to recommend learning resources to students.",
    "Self-Regulated Learning (SRL): Techniques to support students in managing their own learning.",
    "Artificial Intelligence in Education (AI-ED): Use of AI technologies to enhance educational processes.",
    "Clustering Algorithms: Grouping students based on similar characteristics to provide tailored support.",
    "Support Vector Machines (SVM): Application of SVM in educational settings for classification tasks.",
    "Neural Networks: Use of neural networks for predictive modeling in education.",
    "Logistic Regression: Application of logistic regression in predicting educational outcomes.",
    "Random Forests: Use of random forest algorithms in educational data analysis.",
    "Gradient Boosting: Techniques for boosting the performance of prediction models in education.",
    "Natural Language Processing (NLP): Use of NLP to analyze text data from educational contexts.",
    "Sentiment Analysis: Analyzing student feedback and emotions from their text data.",
    "Feature Engineering: Methods for creating and selecting relevant features for predictive models.",
    "Data Preprocessing: Techniques to prepare educational data for analysis.",
    "Evaluation Metrics: Metrics such as accuracy, precision, recall, and F1-score used to evaluate models.",
    "Ensemble Learning: Combining multiple models to improve prediction accuracy.",
    "Context-Aware Learning: Considering the context of learning environments in predictive models.",
    "Course Recommendation: Systems recommending courses based on student profiles and preferences.",
    "E-Learning Analytics: Analysis of data from online learning platforms to improve education.",
    "Predictive Modeling: Building models to predict various educational outcomes.",
    "Student Behavior Analysis: Studying patterns in student behavior to inform interventions.",
    "Dynamic Features: Using time-series data to capture changes in student performance over time.",
    "Data Collection Methods: Techniques for gathering educational data.",
    "Academic Performance Factors: Identifying key factors that influence student academic success.",
    "Predictive Systems for Early Intervention: Systems designed to provide early warnings and support for at-risk students."
]

# Join the sentences into a single text
text = " ".join(sentences)

# Initialize RAKE
rake = Rake()

# Extract keywords
rake.extract_keywords_from_text(text)
keywords = rake.get_ranked_phrases()

print(keywords)
