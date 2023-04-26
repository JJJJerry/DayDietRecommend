from classhelper import user, family, DayDietRec
import time
if __name__ == '__main__':
    test_input_ingredients_list = [
        '牛肉','白菜','番茄','土豆']
    test_user1 = user(need_nutrition=[5, 200, 30], category=['减肥人群'],allergen=['土豆'])
    test_user2 = user(need_nutrition=[10, 300, 30], category=['减肥人群'],allergen=['螃蟹'])
    test_user3 = user(need_nutrition=[5, 100, 10], category=['减肥人群'],allergen=['鱼'])
    test_user_list = [test_user1, test_user2, test_user3]
    test_family = family(test_user_list)
    rec = DayDietRec('final.csv', 'model_cbow.bin')
    start_time = time.time()
    print(rec.get_topn_meals(test_input_ingredients_list, test_family, n=50))
    end_time = time.time()
    print(f'cost: {end_time-start_time}s')
