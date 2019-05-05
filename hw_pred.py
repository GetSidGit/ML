import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#import seaborn as sns

basedir = "/Users/Sid/PycharmProjects/NLP_ThePositiveIndia/data"
data = []
nltk.download('stopwords')

for dir in os.listdir(basedir):
    if not dir.startswith("."):
        for name in os.listdir(basedir + "/" + dir):
            if not name.startswith("."):
                dir_1 = basedir + "/" + dir + "/" + name + "/"
                for fname in os.listdir(dir_1):
                    fname_main = dir_1 + fname
                    with open(fname_main, encoding='utf-8', errors='ignore') as f:
                        tmp_line = f.readlines()
                        for line in tmp_line:
                            main_line_tmp = line[::-1].replace("|","", (line.count("|") - 1))[::-1]
                            main_line = name + "|" + main_line_tmp
                            data.append(main_line.split("|"))

data_df = pd.DataFrame(data, columns=['category', 'heading', 'content'])

# clean dataframe

data_df['heading'] = data_df['heading'].replace('[^a-zA-Z0-9 ]', '', regex=True)
data_df['content'] = data_df['content'].replace('[^a-zA-Z0-9 ]', '', regex=True)

data_df['category'].value_counts()


#india       3011
#politics    2405
#sports      2398
#business    1918

# not imbalanced dataset

tmp_df = pd.concat([data_df.loc[data_df['category'] == 'india'].head(2000), data_df.loc[data_df['category'] == 'sports'].head(1000),
                    data_df.loc[data_df['category'] == 'politics'].head(1000), data_df.loc[data_df['category'] == 'business'].head(1000),
                    data_df.loc[data_df['category'] == 'heartwarming']])





tmp_df['category'] = tmp_df['category'].replace('india', 'not_heartwarming')
tmp_df['category'] = tmp_df['category'].replace('sports', 'not_heartwarming')
tmp_df['category'] = tmp_df['category'].replace('politics', 'not_heartwarming')
tmp_df['category'] = tmp_df['category'].replace('business', 'not_heartwarming')


main_df = tmp_df

main_df['category'].value_counts()

# ignore heading for topic modelling

del main_df['heading']

null_list = main_df[main_df['content'].isnull()].index.tolist()
main_df_new = main_df.drop(null_list)

content_array = np.asarray(main_df_new['content'])
content_list = list(main_df_new['content'])


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(content_list).toarray()
labels = np.array(main_df_new['category'])

X_train, X_test, y_train, y_test = train_test_split(main_df_new['content'], main_df_new['category'], random_state = 0)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

# prediction is decent already

# print(clf.predict(count_vect.transform(["IPL 2019, SRH vs DC Live Cricket Match Score Online: Sunrisers Hyderabad have suffered defeats in their previous two games and would be desperate to get back to winning ways when they take on a confident Delhi Capitals at Rajiv Gandhi International Stadium on Sunday"])))
# print(clf.predict(count_vect.transform(["Claims totalling a little over Rs 1.42 lakh crore were admitted in 88 cases under the IBC till February 28, data collected by the Insolvency and Bankruptcy Board of India (IBBI) showed. "])))


print(clf.predict(count_vect.transform(["IPL 2019, SRH vs DC Live Cricket Match Score Online: Sunrisers Hyderabad have "
                                        "suffered defeats in their previous two games and would be desperate to get back to winning "
                                        "ways when they take on a confident Delhi Capitals at Rajiv Gandhi International Stadium "
                                        "on Sunday"])))



print(clf.predict(count_vect.transform(["Our pursuit of fulfilment and success demands long weeks of extremely busy work time table, "
                                        "with Saturday and Sunday being the only time of leisure, before the next ‘Monday Morning Blues’. "
                                        "Most of us never really want to waste a single moment in the weekend doing any job except for "
                                        "sleeping and ordering food. But, there is group of college students in Andhra Pradesh, who are "
                                        "utilizing their weekends to bring about a grin at the faces of young children, fostering "
                                        "creativity and nurturing desires. Rajesh Mummaneni, one of the foremost members of this group, "
                                        "observed that the manner of presenting education, which in formal terms is called pedagogy,  "
                                        "in government faculties isn’t enough for an overall improvement of the kids. And from there "
                                        "emanated the concept of ‘SAAKSHAR’. Established in December 2016, its pilot mission was launched "
                                        "in a mimimal number of five schools across Vijayanagaram and Visakhapatnam with a goal of "
                                        "enhancing the verbal exchange competencies (including spoken English and Telugu) of the "
                                        "students through methods of interactive studying."])))




models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])



#sns.boxplot(x='model_name', y='accuracy', data=cv_df)
#sns.stripplot(x='model_name', y='accuracy', data=cv_df,
#              size=8, jitter=True, edgecolor="gray", linewidth=2)
#plt.show()


cv_df.groupby('model_name').accuracy.mean()


#LinearSVC                 0.934753
#LogisticRegression        0.923451
#MultinomialNB             0.901053
#RandomForestClassifier    0.673213


model = LinearSVC()
model_lr = LogisticRegression()
model_nb = MultinomialNB()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, main_df_new.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
model_lr.fit(X_train, y_train)
model_nb.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_lr = model_lr.predict(X_test)
y_pred_nb = model_nb.predict(X_test)

conf_mat = accuracies(y_test, y_pred)
accuracy_score(y_test, y_pred)
# 0.94894146948941471
accuracy_score(y_test, y_pred_lr)
# 0.94427148194271482
accuracy_score(y_test, y_pred_nb)
# 0.9178082191780822




# predict new data

basedir = "/Users/Sid/PycharmProjects/NLP_ThePositiveIndia/data"
data_test = []
nltk.download('stopwords')

for dir in os.listdir(basedir):
    if not dir.startswith("."):
        for name in os.listdir(basedir + "/" + dir):
            if not name.startswith("."):
                dir_1 = basedir + "/" + dir + "/" + name + "/"
                for fname in os.listdir(dir_1):
                    fname_main = dir_1 + fname
                    with open(fname_main, encoding='utf-8', errors='ignore') as f:
                        tmp_line = f.readlines()
                        for line in tmp_line:
                            main_line_tmp = line[::-1].replace("|","", (line.count("|") - 1))[::-1]
                            main_line = name + "|" + main_line_tmp
                            data_test.append(main_line.split("|"))

data_test_df = pd.DataFrame(data_test, columns=['category', 'heading', 'content'])

# clean data_testframe

data_test_df['heading'] = data_test_df['heading'].replace('[^a-zA-Z0-9 ]', '', regex=True)
data_test_df['content'] = data_test_df['content'].replace('[^a-zA-Z0-9 ]', '', regex=True)

main_test_df = main_df


main_test_df_new = main_test_df

X_train, X_test, y_train, y_test = train_test_split(main_df_new['content'], main_df_new['category'], random_state = 0)

svc_test = model.fit(X_train_tfidf, y_train)


print(svc_test.predict(count_vect.transform(["IPL 2019, SRH vs DC Live Cricket Match Score Online: Sunrisers Hyderabad have "
                                        "suffered defeats in their previous two games and would be desperate to get back to winning "
                                        "ways when they take on a confident Delhi Capitals at Rajiv Gandhi International Stadium "
                                        "on Sunday"])))


print(svc_test.predict(count_vect.transform(["Compassionate Kozhikode, a platform to generate and nurture the spirit of compassion"
                                             " in individuals is founded on a firm faith in the innate goodness in each one of us. "
                                             "It believes that most of the human beings are  compassionate and capable of expressing "
                                             "it when opportunities are made visible. An initiative by the then district collector of "
                                             "Kozhikode, Prasanth Nair IAS, Operation Sulaimani is a tipping point for building a "
                                             "compassionate destination out of Kozhikode."])))