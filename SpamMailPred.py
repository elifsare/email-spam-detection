# Loading the data
import pandas as pd
rawData = pd.read_csv('D:/elif/calisma_alani/mail_data.csv')
print(rawData)

# replace the null values a null string
mailData = rawData.where((pd.notnull(rawData)), '') # çift tırnak boş verileri temslil ediyor
print(mailData.head())

#checking the number of rows and colums
print("satır, sütun --> " + str(mailData.shape))

# label spam mail as 0; ham mail as 1;
mailData.loc[mailData['Category'] == 'spam', 'Category',] = 0 # Category sütunundaki metin 'spam' ise bu değeri 0 ile değiştiriyorum
mailData.loc[mailData['Category'] == 'ham', 'Category',] = 1 # Category sütunundaki metin 'ham' ise bu değeri 1 ile değiştiriyorum

# verileri metin ve etiket olarak ayıracağız (separating the data as texts and label)
# x tüm mesajlar y tüm etiketler
X = mailData['Message']
y = mailData['Category']
print(X)
print(y)

# split the data into training and testing sets
from sklearn.model_selection import train_test_split
# X deki verileri 2 ye böleceğiz ki train ve test için 2 ayrı şey olsun
# y deki verileri 2 ye böleceğiz ki train ve test için 2 ayrı şey olsun
# toplam 4 bölüm aşağıda;
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2, #test için verinin %20'si kullanılacak train için %80
                                                    random_state = 3)  

# Feature Extraction (sabit değerleri sayısal değere dönüştürdük)
# transform the text data to feature vectors that can be used as input to the ligistic regression
# metni sayısal değere dönüştürüyoruz
# bu sayısal değerler modeli besleyecek yani giriş verisi olarak kullanılacak
from sklearn.feature_extraction.text import TfidfVectorizer #vektörleştirici
feature_extraction = TfidfVectorizer(min_df = 1,
                                     stop_words = 'english',
                                     lowercase = 'True')
# Fit the model
