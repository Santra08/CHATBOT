
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
word=[]
classes = []
doc = []
ignore_words = ['?', '!','(',')']
data_file = open(r"C:\Users\santr\Desktop\santra kujnni\CHATBOT\intents-1.json").read()
intents = json.loads(data_file)

for i in intents["intents"]:
    for p in i["patterns"]:
        w=nltk.word_tokenize(p)
        word.extend(w)
        doc.append((w,i["tag"]))
        if i['tag'] not in classes:
            classes.append(i['tag'])
#lower case, lemmatization

words=[]
for w in word:
    if w not in ignore_words:
        w=w.lower() #lower case : HI to hi
        w=lemmatizer.lemmatize(w)
        words.append(w)
words = sorted(list(set(words))) #words arranged in alphabetical order based on length
print(words)
'''classes = sorted(list(set(classes))) #sorted
#print(len(doc),"documents")
#print (len(classes), "classes", classes)
#print (len(words), "unique lemmatized words", words)
file1=open('words.pkl','wb')
file2=open('classes.pkl','wb')
pickle.dump(words,file1) # wb creates a file if it doesnot exist
pickle.dump(classes,file2)#dumbs information to the file and saves in pkl format
#pickle file is used for serializing, more flexible based on the type (int,str) in comparison to JSON

#CREATING THE TRAINING DATA

train = []
#[0]*2 = [0,0]
#[0]*4=[0,0,0,0]
output_empty = [0]*len(classes)
for d in doc:
    bag=[]
    pattern_words=d[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        if w in pattern_words:
            bag.append(1)
        else:
            bag.append(0)
    output_row=list(output_empty) #why give list() if it is already a list??
    output_row[classes.index(d[1])]=1
    train.append([bag,output_row])
random.shuffle(train)
train=np.array(train)
x_train=list(train[:,0]) #inside the train the list in 0th index
y_train=list(train[:,1])
print("Training data created")


#MODEL CREATION
# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(x_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h6', hist)
print("model created")
'''