import nltk
import pymorphy2
from nltk.corpus import stopwords
import re
import ssl
import matplotlib.pyplot as plt
import numpy as np


tweets_amount = 10000

analyser = pymorphy2.MorphAnalyzer()


def main():

    tweets = clean_database()
    # Получаем словарь из даты отправки и самого очищенного твита
    frequency_words, lengths_tweets, normalized_tweets = words_frequency(tweets)
    # Получаем кортежи с частотностью слов и длиной твитов

    with open("frequency.txt", 'w') as f:
        for word, frequency in frequency_words:
            f.write("{} - {} - {}%\n".format(word, frequency, round((frequency/tweets_amount)*100, 2)))
    # Выводим в файл частотность слов

    with open("twits_length.txt", 'w') as f:
        for length_tweet, frequency in lengths_tweets:
            f.write("{} - {} - {}%\n".format(length_tweet, frequency, round((frequency/tweets_amount)*100, 2)))
    # Выводим в файл длины твитов

    with open("estimations.txt", 'r') as f:
        estimation_dict = {}
        for line in f.readlines():
            estimation_dict[line.split()[0]] = line.split()[1]
    # Создаем словарик частотности слов для удобства

    classifications = classification(normalized_tweets, estimation_dict)
    # Получаем данные наших классификаций по позитиву/нейтралу/негативу

    final_result = classifications[0]
    # Сохраним результаты 1 способа для 6 задания

    plot_classifier(classifications)
    # Строим график по данным

    with open("classifications.txt", 'w') as f:
        for index in range(len(classifications)):
            f.write("Rule {}\n".format(index + 1))
            for category in ("Positive", "Neutral", "Negative"):

                f.write("{} - {} - {}%\n"
                        .format(category,
                                classifications[index][category],
                                round(classifications[index][category]/tweets_amount*100, 2)))
            f.write("\n")
    # Записываем данные классификаций по правилам в файл

    top_five = get_top_five(frequency_words, estimation_dict)
    # Получаем данные топов позитивных и негативных прилагательных

    plot_top_five(top_five)
    # Строим график по данным

    with open("adjectives.txt", 'w') as f:
        f.write("Top-5 Positive:\n")
        for top_element in top_five[0]:
            f.write("{} - {} - {}%\n".
                    format(top_element[0], top_element[1], round(top_element[1]/tweets_amount*100, 2)))
        f.write("\n")
        f.write("Top-5 Negative:\n")
        for top_element in top_five[1]:
            f.write("{} - {} - {}%\n".
                    format(top_element[0], top_element[1], round(top_element[1]/tweets_amount*100, 2)))
    # Записываем в файл топы позитива и негатива

    time_utils(tuple(reversed(normalized_tweets)),
               tuple(reversed([timeline["time"] for timeline in tweets])),
               estimation_dict,
               final_result)

    with open("hours.txt", 'r') as f:
        hours_list = []
        for line in f.readlines():
            hours_list.append({str(line.split()[2]): [str(line.split()[4]), str(line.split()[5]).split("/")]})

    plot_timeline(hours_list)

    tops_estimated, matched_percent, tops_best_worth = estimation_check(estimation_dict, normalized_tweets)

    with open("estimation_check.txt", 'w') as f:
        f.write("Top-5 Closest:\n")
        for top_element in tops_estimated[0]:
            f.write("{} {} {}\n".
                    format(top_element[0], top_element[1], top_element[3]))
        f.write("\n")
        f.write("Top-5 Furthest:\n")
        for top_element in tops_estimated[1]:
            f.write("{} {} {}\n".
                    format(top_element[0], top_element[1], top_element[3]))
        f.write("\nEstimation accuracy: {}%".format(matched_percent))

    with open("best_worst.txt", 'w') as f:
        f.write("Top-5 Most Positive:\n")
        for top_element in tops_best_worth[1]:
            f.write("{} {}\n".
                    format(top_element[0], top_element[3]))
        f.write("\n")
        f.write("Top-5 Most Negative:\n")
        for top_element in tops_best_worth[0]:
            f.write("{} {}\n".
                    format(top_element[0], top_element[3]))

    plot_best_worth(tops_best_worth)


def clean_database():
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
    # Костыль для адекватного скачивания базы с nltk
    nltk.download('stopwords')
    # Загрузка базы ненужных слов
    with open("data.txt", 'r') as f:
        row_data_dict = \
            [{"time": str(line[0:16]), "raw_text": str(line[17:])} for line in f.readlines() if line != "\n"]
        row_data_dict[0]["time"] = "2018-07-11 01:27"
        # Костыль для уничтожения кракозябры в начале файла
    # Формируем словарь твитов с ключами Время и Сырой Текст

    for raw_record in row_data_dict:
        record = raw_record["raw_text"].lower()
        # Загоняем все в нижний регистр
        record = re.sub('#[ ]*[A-zA-яё0-9]*', '', record)
        # Удаление хэштегов
        record = re.sub("@[ ]*[^ \n]*", '', record)
        # Удаление юзернеймов
        record = re.sub("(?:pic.twitter.com/|https://|http://|.twitter.com/)[^ \n]*", '', record)
        # Удаление ссылок
        record = re.sub("[.,!?@$%^&*()_.-=+\"№;/`:<>{}]", ' ', record)
        # Меняем все спецсимволы на пробел, так как во многих твитах они срослись с двумя словами
        record = re.sub("\n", '', record)
        # Удаление символов переноса строки
        record = re.sub("[^А-я ]", '', record)
        # Удаление иностранных слов и цифр
        banned_words = stopwords.words("russian")
        raw_record["raw_text"] = " "\
            .join([word for word in record.split() if word != " " and word not in banned_words and len(word) > 3])
        # Избавляемся от лишних пробелов, которые появились при работе со спецсимволами ранее
        # Удаляем все ненужные слова (частицы, союзы и тд)
    return row_data_dict


def words_frequency(tweets):
    frequency_dict = {}
    # Инициализируем словарь для хранения частотности слов
    lenghts_tweets = {}
    # Словарь для хранения длин твитов
    normalized_tweets = []
    for tweet in tweets:
        temp_tweet = tweet["raw_text"].split()
        # Дробление твита на слова
        temp_tweet_length = len(temp_tweet)
        # Получаем длину твита
        if temp_tweet_length not in lenghts_tweets:
            lenghts_tweets[temp_tweet_length] = 1
            # Если твитов такой длины нет, то создаем ключ с такой длиной
        else:
            lenghts_tweets[temp_tweet_length] += 1
            # Если твит с такой длиной есть - то апаем счетчик на 1
        temp_temp_tweet = []
        # Временный список слов твита (чтобы избежать повторения вхождения слов в твиты)
        for word in temp_tweet:
            word = analyser.parse(word)[0]
            # Берем самое первое совпадение
            if str(word.tag.POS) in \
                    ('NOUN', 'ADJF', 'ADJS', 'COMP', 'VERB', 'INFN', "PRTF", "PRTS", "GRND", "ADVB"):
                # Оставляем только нужные нам части речи - существительные, прилагательные, глаголы и производные
                word = word.normal_form.lower()
                if word not in temp_temp_tweet and len(word) > 3:
                    # Отсекаем всякие местоимения, с которыми не справился PyMorph2
                    temp_temp_tweet.append(word)
                    if word not in frequency_dict:
                        frequency_dict[word] = 1
                        # Если слова в нормальной форме нет в словаре - добавляем
                    else:
                        frequency_dict[word] += 1
                        # Если слово есть в словаре - апаем его счетчик
        normalized_tweets.append(temp_temp_tweet)
    return tuple(reversed(sorted(frequency_dict.items(), key=lambda x: x[1]))),\
        tuple(reversed(sorted(lenghts_tweets.items(), key=lambda x: x[1]))), normalized_tweets


def classification(normalized_tweets, estimation_dict):
    # Правило 1
    first_rule_data = {"Positive": 0, "Neutral": 0, "Negative": 0}
    # Словарь для вхождений позитивных, нейтральных и негативных твитов
    for tweet in normalized_tweets:
        temp_sum = 0
        # Временная сумма каждого твита
        for word in tweet:
            temp_sum += int(estimation_dict[word])
        if temp_sum >= 1:
            first_rule_data["Positive"] += 1
        elif 1 > temp_sum > -1:
            first_rule_data["Neutral"] += 1
        elif -1 >= temp_sum:
            first_rule_data["Negative"] += 1

    # Правило 2
    second_rule_data = {"Positive": 0, "Neutral": 0, "Negative": 0}
    # Словарь для вхождений позитивных, нейтральных и негативных твитов
    for tweet in normalized_tweets:
        temp_positive = 0
        temp_neutral = 0
        temp_negative = 0
        for word in tweet:
            if int(estimation_dict[word]) == 1:
                temp_positive += 1
            elif int(estimation_dict[word]) == 0:
                temp_neutral += 1
            elif int(estimation_dict[word]) == -1:
                temp_negative += 1
        if temp_neutral >= temp_positive and temp_neutral >= temp_negative:
            second_rule_data["Neutral"] += 1
        elif temp_positive >= temp_neutral and temp_positive >= temp_negative:
            second_rule_data["Positive"] += 1
        elif temp_negative >= temp_neutral and temp_negative >= temp_positive:
            second_rule_data["Negative"] += 1

    # Правило 3
    # Умножаем длину каждого слова на оценку, складываем так все слова, а потом делим на количество слов в твите
    third_rule_data = {"Positive": 0, "Neutral": 0, "Negative": 0}
    # Словарь для вхождений позитивных, нейтральных и негативных твитов
    for tweet in normalized_tweets:
        temp_sum = 0
        # Временная сумма каждого твита
        for word in tweet:
            temp_sum += int(estimation_dict[word]) * len(word)
        try:
            temp_sum /= len(tweet)
        except ZeroDivisionError:
            temp_sum = 0

        if temp_sum >= 0.5:
            third_rule_data["Positive"] += 1
        elif 0.5 > temp_sum > -0.5:
            third_rule_data["Neutral"] += 1
        elif -0.5 >= temp_sum:
            third_rule_data["Negative"] += 1

    # Правило 4
    # Делаем вывод по первому слову твита (его оценке)
    fourth_rule_data = {"Positive": 0, "Neutral": 0, "Negative": 0}
    # Словарь для вхождений позитивных, нейтральных и негативных твитов
    for tweet in normalized_tweets:
        if len(tweet) > 0:
            if estimation_dict[tweet[0]] == "1":
                fourth_rule_data["Positive"] += 1
            elif estimation_dict[tweet[0]] == "0":
                fourth_rule_data["Neutral"] += 1
            elif estimation_dict[tweet[0]] == "-1":
                fourth_rule_data["Negative"] += 1
        else:
            fourth_rule_data["Neutral"] += 1

    return first_rule_data, second_rule_data, third_rule_data, fourth_rule_data


def plot_classifier(data):

    n_groups = 4
    positive = [temp_dict["Positive"] for temp_dict in data]
    neutral = [temp_dict["Neutral"] for temp_dict in data]
    negative = [temp_dict["Negative"] for temp_dict in data]

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.20
    opacity = 0.8

    rects1 = plt.bar(index, positive, bar_width,
                     alpha=opacity,
                     color='b',
                     label='positive')

    rects2 = plt.bar(index + bar_width, neutral, bar_width,
                     alpha=opacity,
                     color='g',
                     label='neutral')

    rects3 = plt.bar(index + 2*bar_width, negative, bar_width,
                     alpha=opacity,
                     color='r',
                     label='negative')

    plt.xlabel('Rule')
    plt.ylabel('Amount of tweets')
    plt.title('Tweets by rules')
    plt.xticks(index + bar_width, ('1', '2', '3', '4'))
    plt.legend()
    plt.savefig("classification.png")


def get_top_five(frequency_words, estimation_dict):
    top_five_positive = []
    top_five_negative = []
    # Два списка для топ-5 позитивных и негативных прилагательных
    for word in frequency_words:
        # Используем ранее полученный кортеж кортежей с частотностью слов, который уже отсортирован по убыванию
        if analyser.parse(word[0])[0].tag.POS == "ADJF":
            # Проверяем прилагательное ли наше слово
            if estimation_dict[word[0]] == "1" and len(top_five_positive) != 5:
                # Если оно имеет окраску 1 - пишем его в топ-5 позитива
                top_five_positive.append(word)
            elif estimation_dict[word[0]] == "-1" and len(top_five_negative) != 5:
                # Если оно имеет окраску -1 - пишем его в топ-5 негатива
                top_five_negative.append(word)
        if len(top_five_positive) == 5 and len(top_five_negative) == 5:
            break
            # Если у нас списки имеют по 5 элементов, сворачиваем наш for и выдаем списки
    return top_five_positive, top_five_negative


def plot_top_five(data):
    n_groups = 5
    positive = [amount[1] for amount in data[0]]
    negative = [amount[1] for amount in data[1]]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    index = np.arange(n_groups)
    bar_width = 0.20
    opacity = 0.8

    rect1 = ax1.bar(index, positive, bar_width,
                    alpha=opacity,
                    color='b',
                    label='positive')

    rect2 = ax2.bar(index, negative, bar_width,
                    alpha=opacity,
                    color='r',
                    label='negative')

    ax1.set_title('Positive adjectives')
    ax1.set_xticks(np.arange(n_groups))
    ax1.set_xticklabels([word[0] for word in data[0]])
    ax1.set_ylabel("Tweets amount")
    ax1.legend()

    ax2.set_title('Negative adjectives')
    ax2.set_xticks(np.arange(n_groups))
    ax2.set_xticklabels([word[0] for word in data[1]])
    ax2.set_ylabel("Tweets amount")
    ax2.legend()

    plt.savefig("adjectives.png")


def time_utils(tweets, timelines, estimation_dict, final_result):
    # Временное окно - 6 минут
    # Шаг - 4 часа

    days_list = ["8", "9", "0"]
    # Проверка 9 позиции

    hours_list = ["00", "04", "08", "12", "16", "20"]
    # Проверка 11-12 позиций

    result = []

    with open("hours.txt", "w") as f:
        for day_index in range(len(days_list)):
            for hours_index in range(len(hours_list)):
                temp_dict = {"Positive": 0, "Neutral": 0, "Negative": 0, "Total": 0}
                for tweet_index in range(tweets_amount):
                    if timelines[tweet_index][11:13] == hours_list[hours_index] \
                            and timelines[tweet_index][9] == days_list[day_index]:
                        result.append(temp_dict)
                        f.write("23:54 - {}:00 : {} {}/{}/{}\n"
                                .format(hours_list[hours_index],
                                        temp_dict["Total"],
                                        round(temp_dict["Positive"]/temp_dict["Total"], 2),
                                        round(temp_dict["Neutral"]/temp_dict["Total"], 2),
                                        round(temp_dict["Negative"]/temp_dict["Total"], 2)))
                        break
                    temp_sum = 0
                    # Временная сумма каждого твита
                    for word in tweets[tweet_index]:
                        temp_sum += int(estimation_dict[word])
                    temp_dict["Total"] += 1
                    if temp_sum >= 1:
                        temp_dict["Positive"] += 1
                    elif 1 > temp_sum > -1:
                        temp_dict["Neutral"] += 1
                    elif -1 >= temp_sum:
                        temp_dict["Negative"] += 1
        total_dict = {"Total": tweets_amount}
        final_result.update(total_dict)
        result.append(final_result)
        f.write("23:54 - 1:27 : {} {}/{}/{}\n"
                .format(final_result["Total"],
                        round(final_result["Positive"] / final_result["Total"], 2),
                        round(final_result["Neutral"] / final_result["Total"], 2),
                        round(final_result["Negative"] / final_result["Total"], 2)))


def plot_timeline(data):
    x_labels = [list(element.keys())[0] for element in data]
    y1_amount = [int(list(element.values())[0][0]) for element in data]
    y2_positive = [float(list(element.values())[0][1][0]) for element in data]
    y2_neutral = [float(list(element.values())[0][1][1]) for element in data]
    y2_negative = [float(list(element.values())[0][1][2]) for element in data]

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    n_groups = 19

    index = np.arange(n_groups)
    bar_width = 0.20
    opacity = 0.8

    ax1.plot(index, y2_positive, label="positive")
    ax1.plot(index, y2_neutral, "--", label="neutral")
    ax1.plot(index, y2_negative, ":", label="negative")
    ax1.set_xticks(np.arange(n_groups))
    ax1.set_xticklabels(x_labels)
    ax1.set_title('Distribution classes in time')
    ax1.set_ylabel("Fraction")
    ax1.legend()

    rect2 = ax2.bar(index, y1_amount, bar_width,
                    alpha=opacity,
                    color='r',
                    label='Tweets amount')
    ax2.set_title('Tweets amount in time')
    ax2.set_xticks(np.arange(n_groups))
    ax2.set_xticklabels(x_labels)
    ax2.set_ylabel("Tweets amount")
    ax2.legend()

    plt.savefig("timeline.png")


def estimation_check(estimations_dict, normalized_tweets):
    result = []
    full_amount = len(estimations_dict)
    full_matched = 0

    for word in estimations_dict:
        temp_sum_tweet = 0
        amount = 0
        for tweet in normalized_tweets:
            if word in tweet:
                temp_sum = 0
                amount += 1
                for tweet_word in tweet:
                    temp_sum += int(estimations_dict[tweet_word])
                if temp_sum >= 1:
                    temp_sum_tweet += 1
                elif 1 > temp_sum > -1:
                    pass
                elif -1 >= temp_sum:
                    temp_sum_tweet -= 1

        result.append([word,
                       estimations_dict[word],
                       amount,
                       round(temp_sum_tweet/amount, 2),
                       round(abs(int(estimations_dict[word]) - round(temp_sum_tweet/amount, 2)), 2)])

        if round(abs(int(estimations_dict[word]) - round(temp_sum_tweet/amount, 2)), 2) <= 0.5:
            full_matched += 1

    top_five_closest = list(sorted(result, key=lambda x: x[4]))[:5]
    top_five_furthest = list(reversed(sorted(result, key=lambda x: x[4])))[:5]

    top_five_best = list(sorted(result, key=lambda x: x[3]))[:5]
    top_five_worth = list(reversed(sorted(result, key=lambda x: x[3])))[:5]
    return (top_five_closest, top_five_furthest), \
        round(full_matched/full_amount*100, 2), \
        (top_five_best, top_five_worth)


def plot_best_worth(data):
    n_groups = 5
    best = [grade[3] for grade in data[1]]
    worth = [grade[3] for grade in data[0]]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    index = np.arange(n_groups)
    bar_width = 0.20
    opacity = 0.8

    rect1 = ax1.bar(index, best, bar_width,
                    alpha=opacity,
                    color='b',
                    label='best')

    rect2 = ax2.bar(index, worth, bar_width,
                    alpha=opacity,
                    color='r',
                    label='worth')

    ax1.set_title('Best words')
    ax1.set_xticks(np.arange(n_groups))
    ax1.set_xticklabels([word[0] for word in data[1]])
    ax1.set_ylabel("Grade")
    ax1.legend()

    ax2.set_title('Worth words')
    ax2.set_xticks(np.arange(n_groups))
    ax2.set_xticklabels([word[0] for word in data[0]])
    ax2.set_ylabel("Grade")
    ax2.legend()

    plt.savefig("best_worth.png")


if __name__ == "__main__":
    main()


