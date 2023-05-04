from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import torch
import datetime
CUSTOM_WEIGHT={'score1':1,'score2':1,'score3':1,'score4':1,'score5':1,'score6':1}
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


class User:  # user类
    def __init__(self, need_nutrition=[],category=[],kouwei=[],allergen=[]):
        self.allergen = allergen
        self.category=category
        self.kouwei=kouwei
        self.need_nutrition = need_nutrition


class Family:
    def __init__(self, user_list=[],weight_dic=CUSTOM_WEIGHT):
        self.user_list = user_list
        self.user_num = len(self.user_list)
        self.weight_dic=weight_dic
        self.check_weight()
    def check_weight(self):
        sum_weight=0
        for v in self.weight_dic.values():
            sum_weight+=v
        for k in self.weight_dic.keys():
            self.weight_dic[k]/=sum_weight #归一化
class Score:
    def __init__(self,score_dic:dict):
        assert len(score_dic.keys())==7
        self.score_dic=score_dic
    def __lt__(self,other): #重写排序函数   
        return self.score_dic['final_score']<other.score_dic['final_score']
class Result:
    def __init__(self,score:Score,it:tuple):
        self.score=score
        self.it=it
    def __lt__(self,other): #重写排序函数   
        return self.score>other.score
class DayDietRec:
    def __init__(self, data_path, model_save_path):
        """init

        Args:
            data_path (str): the path of the csv file
            model_save_path (str): the path of the Word2Vec model
        """
        self.data = pd.read_csv(data_path)
        self.corpus = self.get_and_sort_corpus_ingredient(self.data)
        self.train_ingredient_model(model_save_path)
        self.model = self.load_ingredient_model(model_save_path)
        self.recipe_category_emb=torch.nn.Embedding(len(self.data),64)
        self.recipe_category_emb.load_state_dict(torch.load('recipe_category_emb.emb'))
        self.user_category_emb=torch.nn.Embedding(5,64)
        self.user_category_emb.load_state_dict(torch.load('user_category_emb.emb'))
        
        self.recipe_kouwei_emb=torch.nn.Embedding(len(self.data),128)
        self.recipe_kouwei_emb.load_state_dict(torch.load('recipe_kouwei_emb.emb'))
        self.user_kouwei_emb=torch.nn.Embedding(20,128)
        self.user_kouwei_emb.load_state_dict(torch.load('user_kouwei_emb.emb'))
        self.time_food=self.get_time_food('time.txt')
        
        print('推荐系统初始化成功!')
    def get_time_food(self,path):
        dic={}
        f=open(path,'r',encoding='utf-8')
        lines=f.readlines()
        for line in lines:
            food,date=line.split()
            dic[food]=date
        f.close()
        return dic
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
            self.corpus, sg=0, workers=8, window=self.get_window(self.corpus), min_count=1, vector_size=128
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
        return 1-(np.array(sum_similarity).mean()+1)/2  # 计算这个套餐的平均相似度,映射到0-1

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
        return (cos_sim[0][0]+1)/2 #映射到0-1

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
    def get_category(self,s:str):
        if s=='无':
            return []
        temp=s.split(';')
        if '' in temp:
            temp.remove('')
        return temp
    @torch.no_grad()
    def user_category_score(self,u:User,ind:int): #for category score
        if u.category==[]:
            return 0.5
        user_emb=[]
        for c in u.category:
            assert type(c)==int
            user_emb.append(self.user_category_emb(torch.tensor(c,dtype=torch.long)))
        user_emb=torch.vstack(user_emb).mean(axis=0)
        recipe_emb=self.recipe_category_emb(torch.tensor([ind],dtype=torch.long))     
        user_emb_normed=torch.nn.functional.normalize(user_emb.unsqueeze(0))
        recipe_emb_normed=torch.nn.functional.normalize(recipe_emb)
        score=torch.mul(user_emb_normed,recipe_emb_normed).sum().numpy() #score是[-1,1]
        score=(score+1)/2 #映射到0-1
        return score
    def get_category_score(self,it,the_family:Family):
        score=[]
        for user in the_family.user_list:
            for i in it:
                score.append(self.user_category_score(user,i))
        return np.array(score).mean()
    
    
    @torch.no_grad()
    def user_kouwei_score(self,u:User,ind:int): #for category score
        if u.kouwei==[]:
            return 0.5
        user_kouwei_emb=[]
        for c in u.kouwei:
            assert type(c)==int
            user_kouwei_emb.append(self.user_kouwei_emb(torch.tensor(c,dtype=torch.long)))
            
        user_kouwei_emb=torch.vstack(user_kouwei_emb).mean(axis=0)
        recipe_kouwei_emb=self.recipe_kouwei_emb(torch.tensor([ind],dtype=torch.long))
        
        user_kouwei_emb_normed=torch.nn.functional.normalize(user_kouwei_emb.unsqueeze(0))
        recipe_kouwei_emb_normed=torch.nn.functional.normalize(recipe_kouwei_emb)
        score=torch.mul(user_kouwei_emb_normed,recipe_kouwei_emb_normed).sum().numpy() #score几乎是[-1,1]
        score=(score+1)/2 #映射到0-1
        return score
   
    def get_kouwei_score(self,it,the_family:Family):
        score=[]
        for user in the_family.user_list:
            for i in it:
                score.append(self.user_kouwei_score(user,i))
        return np.array(score).mean()
    def compute_time_score(self,ingredient_list):
        score_list=[]
        for ing in ingredient_list:
            score=0.5
            if ing in self.time_food.keys():
                a=min(datetime.date.today().month,int(self.time_food[ing]))
                b=max(datetime.date.today().month,int(self.time_food[ing]))
                distance=min(b-a,a+12-b)
                score=(6-distance)/6
            score_list.append(score)
        return np.array(score_list).mean()
    def get_time_score(self,it):
        score=[]
        for ind in it:
            ingredient_list=self.get_ingredient(self.data.parsed[ind])
            score.append(self.compute_time_score(ingredient_list))
        return np.array(score).mean()
    def compute_final_score(self,score_dic:dict,weight_dic:dict):
        final_score=0.0
        for k,v in score_dic.items():      
            final_score+=v*weight_dic[k]
        return final_score
    
        
    def get_meal_score(self, it, ingredients_list, the_family:Family, tfidf_vec_model):
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
        for u in the_family.user_list:  # 将所有家庭成员的过敏原汇总      
            if u.need_nutrition==[]:
                continue      
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
        
        score_dic={}
        meal_score1 = self.mean_similarity_in_meal(
            meal, tfidf_vec_model)  # meal的食材搭配合理性（不重复度）0-1
        meal_score2 = self.similarity_of_meal_and_ingredients(
            meal, ingredients_list, tfidf_vec_model)  # 做的菜和原料的相似度，越高越好  食材契合（度）
        meal_score3 = self.get_nutrition_similarity(
            meal_nutrition, user_need_nutrition)  # 计算用户所需营养和meal中的营养相似度  营养结构（契合度） 
        meal_score4 = self.get_category_score(it,the_family)  # 用户类别得分  特殊需求           
        
        meal_score5=self.get_time_score(it)                      # 季节时令得分 季节时令
        meal_score6=self.get_kouwei_score(it,the_family)       # 饮食偏好
        
        
        score_dic['score1']=meal_score1
        score_dic['score2']=meal_score2
        score_dic['score3']=meal_score3
        score_dic['score4']=meal_score4
        score_dic['score5']=meal_score5
        score_dic['score6']=meal_score6
        
        
        score_dic['final_score']=self.compute_final_score(score_dic,the_family.weight_dic)
        return Score(score_dic)
        #return (meal_score2*meal_score3*meal_score4)/(meal_score1*meal_score1)
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
        search_recipes_num =30   # 子搜索集合的菜谱数
        search_num = len(self.data)//search_recipes_num
        max_son_search_num=100
        
        results=[]
        allergen=[]
        for u in the_family.user_list:  # 将所有家庭成员的过敏原汇总
            allergen.extend(u.allergen)
        for i in tqdm(range(search_num)):
            search_from = np.random.choice(
                len(self.data), search_recipes_num, replace=False)  # 从所有菜谱中取一些菜谱作为子搜索区域
            itering = combinations(
                search_from, the_family.user_num+1)  # 获得这些菜谱其中的所有组合
            itering = list(itering)
            random.shuffle(itering) #打乱组合顺序，避免连续很多个组合差异不大
            num=0
            for it in itering:  # 在子搜索区域搜索   
                ok=1
                for ind in it:
                    recipe_ingredient = self.get_ingredient(self.data['parsed'][ind])
                    for ing in recipe_ingredient:
                        if ing in allergen:
                            ok=0
                if ok==0:
                    continue  
                num+=1
                if num>=max_son_search_num:  #别在一个子区域搜索太久
                    break
                score = self.get_meal_score(
                    it, ingredients, the_family, self.model)
                results.append(Result(score,it))
        
        
        
        print(f'选择 {len(results)} 种组合')
        results=sorted(results)
        results = results[:n]
        res = pd.DataFrame(columns=['name', 'final_score','score1','score2','score3','score4','score5','score6'])
        count = 0
        for result in results:
            name = ''
            for index in result.it:
                name += self.data['name'][index]+' '
            res.at[count, 'name'] = name
            res.at[count, 'final_score'] = result.score.score_dic['final_score']
            res.at[count, 'score1'] = result.score.score_dic['score1']
            res.at[count, 'score2'] = result.score.score_dic['score2']
            res.at[count, 'score3'] = result.score.score_dic['score3']
            res.at[count, 'score4'] = result.score.score_dic['score4']
            res.at[count, 'score5'] = result.score.score_dic['score5']
            res.at[count, 'score6'] = result.score.score_dic['score6']

            count += 1
        return res
