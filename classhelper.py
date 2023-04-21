from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np
import pandas as pd


class TfidfEmbeddingVectorizer(object):
    def __init__(self, model_cbow):
        """根据语料,在Word2Vec model的基础上加上Tf-idf属性,让推荐更准确。
        Args:
            model_cbow : Word2Vec model
        """
        self.model_cbow = model_cbow
        self.word_idf_weight = None
        self.vector_size = model_cbow.wv.vector_size

    def fit(self, docs):
        """拟合

        Args:
            docs (list[list[str]]): 语料
        """
        text_docs = []
        for doc in docs:
            text_docs.append(" ".join(doc))
        # print(text_docs)
        tfidf = TfidfVectorizer()
        tfidf.fit(text_docs)
        # 如果一个单词从未见过，则给出已知idf值的最大值idf
        max_idf = max(tfidf.idf_)
        self.word_idf_weight = defaultdict(
            lambda: max_idf,
            [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()],
        )
        # print(self.word_idf_weight)
        # return self

    def transform(self, docs):
        """transform ingredients list to a vector
        Args:
            docs (list[str]): ingredients list
        Returns:
            vector
        """
        doc_word_vector = self.doc_average_list(docs)
        return doc_word_vector

    def doc_average(self, doc):
        """计算文档词嵌入的加权平均值
        Args:
            doc (list[str]): ingredients list

        Returns:
            vector
        """
        mean = []
        # print(doc)
        for word in doc:
            if word in self.model_cbow.wv.index_to_key:
                mean.append(
                    self.model_cbow.wv.get_vector(
                        word) * self.word_idf_weight[word]
                )
        if not mean:
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean

    def doc_average_list(self, docs):
        return np.vstack([self.doc_average(doc) for doc in docs])


class user:  # user类
    def __init__(self, need_nutrition=[], allergen=[]):
        self.allergen = allergen
        self.need_nutrition = need_nutrition


class family:
    def __init__(self, user_list=[]):
        self.user_list = user_list
        self.user_num = len(self.user_list)


class DayDietRec:
    def __init__(self, data_path, model_save_path):
        """init

        Args:
            data_path (str): the path of the csv file
            model_save_path (str): the path of the Word2Vec model
        """
        self.data = pd.read_csv(data_path)
        self.corpus = self.get_and_sort_corpus_ingredient(self.data)
        # self.train_ingredient_model(model_save_path)
        self.model = self.load_ingredient_model(model_save_path)

    def get_ingredient(self, s):
        """parser the str to get the ingredients list
        Args:
            s (str): a text of ingredients

        Returns:
            res (list) : the ingredients list
        """
        res = s.split(' ')
        if '' in res:
            res.remove('')
        return res

    def get_and_sort_corpus_ingredient(self, data):
        """get the corpus to

        Args:
            data (pd.dataframe): the recipes dataframe

        Returns:    
           corpus_sorted: corpus of csv file
        """
        corpus_sorted = []
        for doc in data.parsed:
            doc = self.get_ingredient(doc)  # doc是一个食材列表
            doc.sort()
            corpus_sorted.append(doc)
        return corpus_sorted

    def get_window(self, corpus):  # 计算每个文档的平均长度
        lengths = [len(doc) for doc in corpus]
        avg_len = float(sum(lengths)) / len(lengths)
        return round(avg_len)

    def train_ingredient_model(self, save_path):
        print(f"Length of corpus: {len(self.corpus)}")
        model_cbow = Word2Vec(
            self.corpus, sg=0, workers=8, window=self.get_window(self.corpus), min_count=1, vector_size=50
        )
        model_cbow.save(save_path)
        print("Word2Vec model successfully trained !")
        print(f'saved in {save_path}')

    def load_ingredient_model(self, model_path):
        """load recommend model
        Args:
            model_path (str): the path of model
        Returns:
            model (TfidfEmbeddingVectorizer): recommend model
        """
        model = Word2Vec.load(model_path)
        # 标准化嵌入
        # model.init_sims(replace=True)
        if model:
            print("Successfully load Word2Vec model!")
        tfidf_vec_trr = TfidfEmbeddingVectorizer(model)
        tfidf_vec_trr.fit(self.corpus)
        print('Successfully load TfidfEmbedding model!')
        return tfidf_vec_trr

    def recipe_similarity(self, recipe1, recipe2, tfidf_vec_model):

        emb1 = tfidf_vec_model.transform([recipe1])[0].reshape(1, -1)
        emb2 = tfidf_vec_model.transform([recipe2])[0].reshape(1, -1)
        similarity = cosine_similarity(emb1, emb2)
        return similarity

    def mean_similarity_in_meal(self, meal, tfidf_vec_model):
        """compute the similarity between the recipes combination

        Args:
            meal (list[list[str]]): e.g.:[['黄瓜','番茄'],['土豆','牛肉'],['猪肉','白菜'],['鸡蛋','番茄']] 
            tfidf_vec_model (TfidfEmbeddingVectorizer): recommend model

        Returns:
            similarity: the similarity between the recipes combination
        """
        sum_similarity = []
        for i in range(len(meal)-1):
            for j in range(i+1, len(meal)):
                temp = self.recipe_similarity(
                    meal[i], meal[j], tfidf_vec_model)
                sum_similarity.append(temp)
        return (np.array(sum_similarity).mean()+1)/2  # 计算这个套餐的平均相似度,映射到0-1

    def get_nutrition_similarity(self, recipe_nutrition, user_need_nutrition):
        """compute the nutrition similarity

        Args:
            recipe_nutrition (list): the recipe's nutrition
            user_need_nutrition (list): the nutrition that user needs

        Returns:
            similarity (double) : nutrition similarity
        """
        recipe_nutrition = np.array(recipe_nutrition)
        user_need_nutrition = np.array(user_need_nutrition)
        cos_sim = cosine_similarity(recipe_nutrition.reshape(
            1, -1), user_need_nutrition.reshape(1, -1))
        return cos_sim[0][0]

    def similarity_of_meal_and_ingredients(self, meal, ingredients_list, tfidf_vec_model):
        """compute the similarity of the combination of recipes with the ingredients
        Args:
            meal (list[list[str]]): e.g.: [['黄瓜','番茄'],['土豆','牛肉'],['猪肉','白菜'],['鸡蛋','番茄']] 
            ingredients_list (list[str]):  e.g.: ['牛肉','番茄','土豆','白菜','鱼','猪肉','胡萝卜','鸡蛋','黄瓜']
            tfidf_vec_model (TfidfEmbeddingVectorizer): recommend model

        Returns:
            similarity (double): the similarity score 
        """
        sum_similarity = []
        for r in meal:
            sum_similarity.append(self.recipe_similarity(
                r, ingredients_list, tfidf_vec_model))
        return (np.array(sum_similarity).mean()+1)/2  # 映射到0-1

    def get_meal_score(self, it, ingredients_list, the_family, tfidf_vec_model):
        """to compute a meal's score

        Args:
            it (tuple[int]): a combination of the index of the recipes
            ingredients_list (list): ingredients list
            the_family (family): based on the family to recommend
            tfidf_vec_model (TfidfEmbeddingVectorizer): recommend model

        Returns:
            score (int) : the score of the combination of recipes
        """
        user_need_nutrition = [0, 0, 0]  # 糖，热量，脂肪
        allergen = []
        for u in the_family.user_list:  # 将所有家庭成员的过敏原汇总
            allergen.extend(u.allergen)
            user_need_nutrition[0] += u.need_nutrition[0]  # 计算家庭成员所需的营养
            user_need_nutrition[1] += u.need_nutrition[1]
            user_need_nutrition[2] += u.need_nutrition[2]
        meal_nutrition = [0, 0, 0]  # 糖，热量，脂肪

        meal = []
        for index in it:
            recipe_ingredient = self.get_ingredient(self.data['parsed'][index])
            meal.append(recipe_ingredient)  # 将食材加入到这顿meal中
            meal_nutrition[0] += float(self.data['tang'][index][:-1])
            meal_nutrition[1] += float(self.data['reliang'][index][:-1])
            meal_nutrition[2] += float(self.data['zhifang'][index][:-1])
            # 将菜谱中的营养相加
        for r in meal:  # 判断meal有没有用户过敏的食材，若有，直接0分
            for ing in r:
                if ing in allergen:
                    return 0

        meal_score1 = self.mean_similarity_in_meal(
            meal, tfidf_vec_model)  # 做的菜中的平均相似度，越小越好
        meal_score2 = self.similarity_of_meal_and_ingredients(
            meal, ingredients_list, tfidf_vec_model)  # 做的菜和原料的相似度，越高越好
        meal_score3 = self.get_nutrition_similarity(
            meal_nutrition, user_need_nutrition)  # 计算用户所需营养和meal中的营养相似度

        return meal_score2*meal_score3/meal_score1

    def get_topn_meals(self, ingredients, the_family, n=5, search_num=100):
        '''
            Parameters:
                data:最终菜谱csv
                ingredients:原料列表
                the_family:用户家庭信息
                tfidf_vec_model:推荐模型
                n:n表示返回前几个菜谱
                search_num:搜索次数
            Returns:
                获得前n个菜谱推荐
        '''
        search_recipes_num = 10  # 子搜索集合的菜谱数
        search_num = 100
        scores = []
        for i in range(search_num):
            search_from = np.random.choice(
                len(self.data), search_recipes_num, replace=False)  # 从所有菜谱中取一些菜谱作为子搜索区域
            itering = combinations(
                search_from, the_family.user_num+1)  # 获得这些菜谱其中的所有组合
            itering = list(itering)
            for it in itering:  # 在子搜索区域搜索
                score = self.get_meal_score(
                    it, ingredients, the_family, self.model)
                scores.append((it, score))

        print(f'选择 {len(scores)} 种组合')
        scores.sort(key=lambda x: x[1], reverse=True)
        scores = scores[:n]

        res = pd.DataFrame(columns=['name', 'score'])
        count = 0
        for score in scores:
            name = ''
            for index in score[0]:
                name += self.data['name'][index]+' '
            res.at[count, 'name'] = name
            res.at[count, 'score'] = score[1]
            count += 1
        return res
