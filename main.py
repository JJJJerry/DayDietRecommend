from classhelper import User,Family,DayDietRec
import time
#1.套餐食材之间的相似度
#2.套餐食材与输入食材相似度
#3.套餐营养与用户所需营养的相似度
#4.套餐菜谱与用户类别的契合度
#5.套餐食材与时令的契合度
#6.套餐菜谱味道与用户所喜爱味道的契合度
WEIGHT={'score1':0.1,'score2':0.1,'score3':0.1,'score4':0.1,'score5':1,'score6':0.1}
#权重之和应该为1
#TODO:
# 1.现在用户只有其所喜爱味道的属性，后期可以加上其不喜欢的味道。多一个属性
# 2.加菜谱数量，现在只有460分菜谱，比较少
# 3.用户种类现在只有5个，后期想加上美食杰上的更多标签
if __name__ == '__main__':
    test_input_ingredients_list = ['猪肉','鱼','牛肉']
    # height weight 运动量 -> 糖、热量、脂肪
    # 系统时间 午饭 
    # 晚饭 -> 剩余
    # 剩余热量=（身高+体重）+运动量-已经吃的
    # 早 20%剩余-> 脂肪糖分（标准）
    
    #https://zhuanlan.zhihu.com/p/352590130
    #营养:：糖、热量、脂肪
    #用户标签:高血脂人群、高血压人群、高血糖人群、减肥人群、儿童
    #口味:['酸辣味', '酸甜味', '家常味', '酱香味', '香辣味', '咸鲜味', '甜味', '鱼香味', '麻辣味',  '咖喱味', '茄汁味', '豆瓣味', '黑椒味', '蒜香味', '葱香味', '果味', '椒麻味','五香味', '奶香味','其它口味']
    
    test_user1 = User(need_nutrition=[], kouwei=[15],category=[],allergen=[])
    test_user2 = User(need_nutrition=[1, 3, 0.1], kouwei=[],category=[],allergen=[])
    test_user3 = User(need_nutrition=[5, 1, 0.1], kouwei=[15],category=[0],allergen=[])
    test_user_list = [test_user1, test_user2, test_user3]
    
    test_family = Family(test_user_list,weight_dic=WEIGHT)
    rec = DayDietRec('final.csv', 'model_cbow.bin')
    start_time = time.time()
    print(rec.get_topn_meals(test_input_ingredients_list, test_family, n=50))
    end_time = time.time()
    print(f'cost: {end_time-start_time}s')
    
