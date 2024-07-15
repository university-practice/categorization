# -*- coding: utf8 -*-

import pandas as pd
from main import find_closest_topic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
from spacy import load
from spacy.lang.ru.examples import sentences
from spacy.lang.ru import Russian

path = 'data.csv'

df = pd.read_csv(path)

nlp = Russian()
load_model = load("ru_core_news_sm")

def preprocess(text):
    doc = load_model(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)

    return " ".join(filtered_tokens)

def find_closest_topics(title, topics):
    # Создание модели TF-IDF и вычисление матрицы TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([preprocess(title)] + list(map(preprocess, topics)))

    # Вычисление косинусного сходства между заголовком и всеми темами
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Поиск индексов 5 тем с наибольшими значениями косинусного сходства
    closest_topic_indices = np.argsort(cosine_similarities)[-5:][::-1]

    # Возвращение ближайших тем и их значений косинусного сходства
    return [(topics[i], cosine_similarities[i]) for i in closest_topic_indices]

def calculate_accuracy(predictions, true_labels, top_n):
    correct_predictions = 0
    for pred, true_label in zip(predictions, true_labels):
        predicted_labels = [p[0] for p in pred[:top_n]]
        if true_label in predicted_labels:
            correct_predictions += 1
    return correct_predictions / len(true_labels)

themes = ['Научные основы охраны здоровья матери, женщины, плода и новорожденного.', 'Трудный диагноз в педиатрии: от практики к науке.', 'Возрастные особенности формирования здоровья в зависимости от медико-социальных факторов, современные технологии прогнозирования, диагностики, лечения и профилактики заболеваний у детей.', 'Трудный диагноз в клинике внутренних болезней.', 'Разработка новых методов профилактики, прогнозирования, диагностики и лечения хирургической и травматолого-ортопедической патологии у детей и взрослых.', 'Разработка новых методов профилактики, диагностики и лечения заболеваний нервной системы у детей и взрослых.', 'Актуальные вопросы клинической психологии и психиатрии в диагностике, лечении и профилактике заболеваний у взрослых и детей.', 'Диспластикоассоциированные заболевания и патологические состояния.', 'Структурно-функциональные и молекулярно-биологические аспекты межтканевых взаимоотношений у человека и животных в норме и патологии.', 'Актуальные проблемы молекулярной и клеточной биологии: структурно-функциональная организация цитоскелета (без публикации).', 'Актуальные вопросы формирования здорового образа жизни, развития оздоровительной, лечебной, адаптивной физической культуры и спорта.', 'Реабилитация пациентов с соматической, неврологической патологией и заболеваниями опорно-двигательного аппарата.', 'Клинические и лабораторно-инструментальные методы контроля эффективности лечения патологии внутренних органов (без публикации).', 'Секция реферативно-аналитических работ по естественно-научным дисциплинам (для студентов 1-2 курсов, без публикации).', 'Медико-социальные, организационно-правовые и организационные аспекты совершенствования оказания медицинской помощи населению.', 'Качество среды и здоровье человека.', 'Актуальные проблемы современной стоматологии.', 'Актуальные вопросы микробиологии.', 'История здравоохранения Ивановской области (без публикации).', 'Актуальные проблемы эндокринной патологии.', 'Актуальные подходы к оздоровлению детей в лечебно-профилактических и образовательных учреждениях.', 'Совершенствование методов профилактики, диагностики и лечения инфекционных заболеваний у взрослых и детей.', 'Современные аспекты деятельности медицинской сестры (для обучающихся учреждений среднего профессионального образования).', 'Онкологические заболевания: профилактика, ранняя диагностика и лечение.', 'Проблемы полиморбидности в клинике внутренних болезней: патогенез, диагностика, лечение и профилактика.', 'История ИГМИ-ИвГМА (без публикации).', 'Секция учащихся школ «Первые шаги в медицинской науке» (без публикации).']

# Прогнозы для всех элементов колонки
predictions = []
true_labels = df['target'].tolist()
for title in df['annotations']: # замените на имя колонки с заголовками
    predictions.append(find_closest_topics(title, themes))


# Вычисление точности для топ-1, топ-3 и топ-5 предсказаний
accuracy_top_1 = calculate_accuracy(predictions, true_labels, top_n=1)
accuracy_top_3 = calculate_accuracy(predictions, true_labels, top_n=3)
accuracy_top_5 = calculate_accuracy(predictions, true_labels, top_n=5)
print("TOP 1 /", len(themes), round(accuracy_top_1 * 100, 2), "%")
print("TOP 3 /", len(themes), round(accuracy_top_3 * 100, 2), "%")
print("TOP 5 /", len(themes), round(accuracy_top_5 * 100, 2), "%")
