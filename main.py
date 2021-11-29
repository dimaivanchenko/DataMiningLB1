import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

OUTPUT_FOLDER = "output/"

# 1) Вхідні дані
sample = pd.read_csv("sms-spam-corpus.csv", encoding="ISO-8859-1")
ham = sample[sample.v1 == "ham"]    # масив рядків з ham
spam = sample[sample.v1 == "spam"]  # масив рядків з spam
stop_words = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here"]

# 2) Обробка даних
haml = []           # масив рядків без цифр та символів з ham
spaml = []          # масив рядків без цифр та символів з spam
ham_filtered = []   # масив слів без стоп слів з ham
spam_filtered = []  # масив слів без стоп слів з spam

# прибираємо цифри та символи
for i in ham.v2:
    haml.append("".join(c for c in i if c.isalpha() or c == " ").lower())
for i in spam.v2:
    spaml.append("".join(c for c in i if c.isalpha() or c == " ").lower())

# прибираємо стопслова
for line in haml:
    l_ham = line.split()
    for word in l_ham:
        if word not in stop_words:
            ham_filtered.append(word)

for line in spaml:
    l_spam = line.split()
    for word in l_spam:
        if word not in stop_words:
            spam_filtered.append(word)

# 3) Cловник слів (підрахування кількості слів)
counts_ham = {}     # об'єкт ключ(слово): значення (кількість)  ham
counts_spam = {}    # об'єкт ключ(слово): значення (кількість)  spam
for word in ham_filtered:
    counts_ham[word] = counts_ham.get(word, 0) + 1

for word in spam_filtered:
    counts_spam[word] = counts_spam.get(word, 0) + 1

# запис до файлу
with open(OUTPUT_FOLDER + "quantity_ham.txt", "w", encoding="ISO-8859-1") as out:
    for key, val in counts_ham.items():
        out.write('{}:{}\n'.format(key, val))

with open(OUTPUT_FOLDER + "quantity_spam.txt", "w", encoding="ISO-8859-1") as out:
    for key, val in counts_spam.items():
        out.write('{}:{}\n'.format(key, val))

# 4) Графічне відображення
# графік розподілу по довжині слів
ham_word_length = []    # масив довжин слів ham
spam_word_length = []   # масив довжин слів spam

for word in ham_filtered:
    length1 = len(word)
    ham_word_length.append(length1)
for word in spam_filtered:
    length2 = len(word)
    spam_word_length.append(length2)

# середня довжина слова
average_words = sum(ham_word_length + spam_word_length) / len(ham_word_length + spam_word_length)

# унікальні довжини слів та кількість повторів цих слів
labels1, ham_word_count = zip(*Counter(ham_word_length).items())
ham_word_l = np.arange(len(labels1))
labels2, spam_word_count = zip(*Counter(spam_word_length).items())
spam_word_l = np.arange(len(labels2))

plt.plot(ham_word_l, ham_word_count, spam_word_l, spam_word_count, average_words, 0, "go")
plt.xlabel("Length of the word")
plt.ylabel("Periodicity")
plt.legend(["Ham", "Spam", "Average " + str(average_words)])
plt.savefig(OUTPUT_FOLDER + "LengthWord.png")
plt.show()


# графік розподілу по довжині рядків
ham_sentence_length = []    # масив довжин рядків ham
spam_sentence_length = []   # масив довжин рядків spam

for line in haml:
    length3 = len(line)
    ham_sentence_length.append(length3)
for line in spaml:
    length4 = len(line)
    spam_sentence_length.append(length4)

# середня довжина рядка
average_sentence = sum(ham_sentence_length + spam_sentence_length) / len(ham_sentence_length + spam_sentence_length)

# унікальні довжини рядків та кількість повторів цих рядків
labels3, ham_sentence_count = zip(*Counter(ham_sentence_length).items())
ham_sentence_l = np.arange(len(labels3))
labels4, spam_sentence_count = zip(*Counter(spam_sentence_length).items())
spam_sentence_l = np.arange(len(labels4))

plt.plot(ham_sentence_l, ham_sentence_count, spam_sentence_l, spam_sentence_count, average_sentence, 0, "go")
plt.xlabel("Length of the lines")
plt.ylabel("Periodicity")
plt.legend(["Ham", "Spam", "Average " + str(average_sentence)])
plt.savefig(OUTPUT_FOLDER + "LengthLines.png")
plt.show()


# графік топ 20
sort_ham = []    # масив ham для сортування за спаданням
sort_spam = []   # масив spam для сортування за спаданням

for (k, v) in counts_ham.items():
    sort_ham.append((v, k))
sort_ham = sorted(sort_ham, reverse=True)

for (k, v) in counts_spam.items():
    sort_spam.append((v, k))
sort_spam = sorted(sort_spam, reverse=True)

# берем перші 20 слів
sort_ham = sort_ham[:20]
sort_spam = sort_spam[:20]

# функція для створення графіку
def popularWords(top20, category, saving_name):
    x_labels = [value[1] for value in top20]
    y_labels = [value[0] for value in top20]
    plt.figure(figsize=(14, 7))
    ax = pd.Series(y_labels).plot(kind='bar')
    ax.set_xticklabels(x_labels)

    rects = ax.patches

    for rect, label in zip(rects, y_labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, label, ha='center', va='bottom')

    plt.xlabel("Word")
    plt.ylabel("Periodicity")
    plt.legend([category])
    plt.savefig(OUTPUT_FOLDER + saving_name)
    plt.show()

popularWords(sort_ham, "Ham", "PopularHam.png")
popularWords(sort_spam, "Spam", "PopularSpam.png")
