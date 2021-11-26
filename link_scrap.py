from newspaper import Article 
import yake
from keytotext import pipeline 
import random 
import torch
import pickle
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_splitter import SentenceSplitter, split_text_into_sentences

splitter = SentenceSplitter(language='en') 
 
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device) 

def get_response(input_text,num_return_sequences):
  batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text 

article_one = Article('https://mailchimp.com/marketing-glossary/digital-marketing')
article_two = Article('https://www.investopedia.com/terms/d/digital-marketing.asp') 
article_three = Article('https://www.marketingevolution.com/marketing-essentials/what-is-a-digital-marketing-platform-marketing-evolution') 
article_four = Article('https://www.webfx.com/internet-marketing/actionable-digital-marketing-strategies.html') 
article_five = Article('https://www.o8.agency/blog/what-digital-marketing-strategy-and-how-create-one') 
article_six = Article('https://www.equinetacademy.com/what-is-digital-marketing/') 
article_seven = Article('https://www.cmswire.com/digital-marketing/the-top-4-digital-marketing-categories/') 
article_eight = Article('https://www.reachfirst.com/different-types-of-digital-marketing-campaigns-explained/') 
article_nine = Article('https://blueinteractiveagency.com/seo-blog/2018/01/9-types-of-internet-marketing-strategies/') 
article_ten = Article('https://leverageedu.com/blog/types-of-digital-marketing/')  

article_one.download() 
article_one.parse() 

article_two.download() 
article_two.parse() 

article_three.download() 
article_three.parse() 

article_four.download() 
article_four.parse() 

article_five.download() 
article_five.parse() 

article_six.download() 
article_six.parse() 

article_seven.download() 
article_seven.parse() 

article_eight.download() 
article_eight.parse() 

article_nine.download() 
article_nine.parse() 

article_ten.download() 
article_ten.parse() 

text_one  = article_one.text 
text_two  = article_two.text 
text_three  = article_three.text 
text_four  = article_four.text 
text_five  = article_five.text 
text_six  = article_six.text 
text_seven  = article_seven.text 
text_eight  = article_eight.text 
text_nine  = article_nine.text 
text_ten  = article_ten.text  

def keywords(article,language, max_ngram_size,deduplication_threshold,numOfKeywords): 
    keyword_list =  [ ] 
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(article)  
    for kw in keywords:
        # print(kw[0]) 
        # print(type(kw[0])) 
        keyword_list.append(kw[0]) 
    return keyword_list 

def double_keywords(article): 
    keyword_list = keywords(article,"en",2,0.6,100) 
    double_keywords = [] 
    for words in keyword_list: 
        req_keyword = words
        double_Word = words.split(" ") 
        if len(double_Word) <= 1:
            pass
            # print("single_word", words)
        else:
            double_keywords.append(words) 
            # print("double_word",words) 
    return double_keywords 

def picking_random_keywords(article,n):
    double_keyword = double_keywords(article) 
    ten_keyword = random.sample(double_keyword,n)    
    # print(ten_keyword) 

    return ten_keyword 

def para_splitting(article_text):

    headings = [ ] 
    para = [] 
    splitted_text = article_text.split('\n\n') 

    for items in splitted_text:
        count = 0
        temp_para = items 
        # print(temp_para) 
        words = items.split(" ") 
        # print(words) 
        for word in words: 
            # print(word) 
            count += 1 
            # print(count) 
        if count <= 7: 
            headings.append(temp_para)
            # print("heading: ",temp_para) 
            # print("*****************")
        elif count > 7:
            para.append(temp_para) 
            # print("paragraph: ",temp_para)
            # print("*****************")
        else:
            pass 
            # print("else text: ", temp_para)

    return headings, para 

def keyword_check(string,ten_keyword,index,count,fetch_sentences): 
    for words in ten_keyword: 
        key = 0
        if (string.find(words) == -1): 
            pass
        else: 
            # print("keyword: " + "'" +words+"'" + " is present in: " + string) 
            key = str(index) + ":" + str(count) 
            fetch_sentences[key] = string 
            # print("************",index) 
            # print("**********",count) 
            # print(key) 
            # print("Found_word: ",words) 
            ten_keyword.remove(words) 
            # print("********", ten_keyword) 
            break 

    return fetch_sentences 

def sentence_fetching(para_list,ten_keyword): 
    fetch_sentences = {} 
    for index in range(len(para_list)):
        paragarphs = para_list[index]   
        index += 1
        # print("Paragarph no # " + str(index) + ": " + paragarphs)  
        # print("******************************************") 
        temp_para = paragarphs 
        sentences = paragarphs.split(".") 
        # print(sentences) 
        sentences = sentences[:-1] 
        count = 1
        for sentence in sentences: 
            temp_sentence = sentence 
            # print("sentence # " + str(count) + " : " + sentence)  
            sentences_dict = keyword_check(sentence,ten_keyword,index,count,fetch_sentences) 
            # dict_value = sentences_dict[key] 
            # print(type(dict_value)) 
            # found_word = dict_value[1] 
            # print("Found_word: ",found_word) 
            # ten_keyword.remove(found_word) 
            # print(sentences_dict) 
            count += 1 

    return sentences_dict

def para_check(para_list,sentences_dict):
    selected_para = {}
    count = 1 
    for keys in sentences_dict:
        # print("Sentence # " + str(count)) 
        # print(keys) 
        selected_sentence = (sentences_dict[keys]) 
        count += 1 
        for para in para_list: 
            if (para.find(selected_sentence) == -1): 
                pass
            else: 
                # print("sentence: " + "count" +selected_sentence)  
                # key = str(index) + ":" + str(count) 
                new_key = keys.split(":")[0]
                # print("new_key: ", new_key)
                selected_para[new_key] = para 
                para_list.remove(para) 
                # print("********", para_list) 
                break
                # print(key) 
                # print("sentence: ",selected_sentence)  

    return selected_para

print("***********************First article**********************************")

#1st Article
one_ten_keyword = picking_random_keywords(text_one,10) 
print("Ten keywords list", one_ten_keyword)
heading_list_one,para_list_one = para_splitting(text_one) 
para_list_one_lenght = len(para_list_one) 
one_sentences_dict = sentence_fetching(para_list_one,one_ten_keyword) 
one_selected_para = para_check(para_list_one,one_sentences_dict)  
print("*****************************************")  

count = 1 

for keys in one_selected_para:
    print("paragraph # " + str(count)) 
    print(keys)
    print(one_selected_para[keys])  
    count += 1 

print("***********************Second article**********************************")

#2nd Article 
two_ten_keyword = picking_random_keywords(text_two,10) 
print("Ten keywords list", two_ten_keyword) 
heading_list_two,para_list_two = para_splitting(text_two) 
para_list_two_lenght = len(para_list_two) 
two_sentences_dict = sentence_fetching(para_list_two,two_ten_keyword) 
two_selected_para = para_check(para_list_two,two_sentences_dict) 
print("*********************************************")   

count = 1 

for keys in two_selected_para: 
    print("paragraph # " + str(count)) 
    print(keys)
    print(two_selected_para[keys])  
    count += 1 

print("************************Third Article********************************")

#3rd Article 
three_ten_keyword = picking_random_keywords(text_three,10) 
print("Ten keywords list", three_ten_keyword) 
heading_list_three,para_list_three = para_splitting(text_three) 
para_list_three_lenght = len(para_list_three) 
three_sentences_dict = sentence_fetching(para_list_three,three_ten_keyword) 
three_selected_para = para_check(para_list_three,three_sentences_dict) 
print("*********************************************")   

count = 1 

for keys in three_selected_para: 
    print("paragraph # " + str(count)) 
    print(keys)
    print(three_selected_para[keys])  
    count += 1 

print("**************************Fourth article********************************")

#4rth Article 
four_ten_keyword = picking_random_keywords(text_four,10) 
print("Ten keywords list", four_ten_keyword) 
heading_list_four,para_list_four = para_splitting(text_four) 
para_list_four_lenght = len(para_list_four) 
four_sentences_dict = sentence_fetching(para_list_four,four_ten_keyword) 
four_selected_para = para_check(para_list_four,four_sentences_dict) 
print("*********************************************")   

count = 1 

for keys in four_selected_para: 
    print("paragraph # " + str(count)) 
    print(keys)
    print(four_selected_para[keys]) 
    count += 1 

print("***************************Fifth Article*******************************")

#5th Article 
five_ten_keyword = picking_random_keywords(text_five,10) 
print("Ten keywords list", five_ten_keyword) 
heading_list_five,para_list_five = para_splitting(text_five) 
para_list_five_lenght = len(para_list_five) 
five_sentences_dict = sentence_fetching(para_list_five,five_ten_keyword) 
five_selected_para = para_check(para_list_five,five_sentences_dict)  
print("*********************************************")   

count = 1 

for keys in five_selected_para: 
    print("paragraph # " + str(count)) 
    print(keys)
    print(five_selected_para[keys])  
    count += 1 

print("**************************Six Article******************************") 

#6th Article 
six_ten_keyword = picking_random_keywords(text_six,10) 
print("Ten keywords list", six_ten_keyword) 
heading_list_six,para_list_six = para_splitting(text_six) 
para_list_six_lenght = len(para_list_six) 
six_sentences_dict = sentence_fetching(para_list_six,six_ten_keyword) 
six_selected_para = para_check(para_list_six,six_sentences_dict) 
print("*********************************************")   

count = 1 

for keys in six_selected_para: 
    print("paragraph # " + str(count)) 
    print(keys)
    print(six_selected_para[keys])  
    count += 1 

print("**************************Seven Article******************************") 

#6th Article 
seven_ten_keyword = picking_random_keywords(text_seven,10) 
print("Ten keywords list", seven_ten_keyword) 
heading_list_seven,para_list_seven = para_splitting(text_seven) 
para_list_seven_lenght = len(para_list_seven) 
seven_sentences_dict = sentence_fetching(para_list_seven,seven_ten_keyword) 
seven_selected_para = para_check(para_list_seven,seven_sentences_dict) 
print("*********************************************")   

count = 1 

for keys in seven_selected_para: 
    print("paragraph # " + str(count)) 
    print(keys)
    print(seven_selected_para[keys])  
    count += 1 

print("**************************Eight Article******************************") 

#6th Article 
eight_ten_keyword = picking_random_keywords(text_eight,10) 
print("Ten keywords list", eight_ten_keyword) 
heading_list_eight,para_list_eight = para_splitting(text_eight) 
para_list_eight_lenght = len(para_list_eight) 
eight_sentences_dict = sentence_fetching(para_list_eight,eight_ten_keyword) 
eight_selected_para = para_check(para_list_eight,eight_sentences_dict) 
print("*********************************************")   

count = 1 

for keys in eight_selected_para: 
    print("paragraph # " + str(count)) 
    print(keys)
    print(eight_selected_para[keys])   
    count += 1 

print("**************************Nine Article******************************") 

#6th Article 
nine_ten_keyword = picking_random_keywords(text_nine,10) 
print("Ten keywords list", nine_ten_keyword) 
heading_list_nine,para_list_nine = para_splitting(text_nine) 
para_list_nine_lenght = len(para_list_nine) 
nine_sentences_dict = sentence_fetching(para_list_nine,nine_ten_keyword) 
nine_selected_para = para_check(para_list_nine,nine_sentences_dict) 
print("*********************************************")   

count = 1 

for keys in nine_selected_para: 
    print("paragraph # " + str(count)) 
    print(keys)
    print(nine_selected_para[keys])  
    count += 1 


print("**************************Ten Article******************************") 

#6th Article 
ten_ten_keyword = picking_random_keywords(text_ten,10) 
print("Ten keywords list", ten_ten_keyword) 
heading_list_ten,para_list_ten = para_splitting(text_ten) 
para_list_ten_lenght = len(para_list_ten) 
ten_sentences_dict = sentence_fetching(para_list_ten,ten_ten_keyword) 
ten_selected_para = para_check(para_list_ten,ten_sentences_dict) 
print("*********************************************")   

count = 1 

for keys in ten_selected_para: 
    print("paragraph # " + str(count)) 
    print(keys)
    print(ten_selected_para[keys])  
    count += 1 

print("*****************************Complete*********************************")

def list_lenght(list1_len,list2_len,list3_len,list4_len,list5_len,list6_len,list7_len,list8_len,list9_len,list10_len):
    len_lst = []
    for len in range(1):
        len_lst.append(list1_len)
        len_lst.append(list2_len) 
        len_lst.append(list3_len)
        len_lst.append(list4_len)
        len_lst.append(list5_len)
        len_lst.append(list6_len)
        len_lst.append(list7_len)
        len_lst.append(list8_len)
        len_lst.append(list9_len)
        len_lst.append(list10_len) 
    print("lenght list: ", len_lst) 
    max_lenght = max(len_lst) 
    # print("Largest element is:", max_lenght) 

    return max_lenght

max_lenght = list_lenght(para_list_one_lenght,para_list_two_lenght,para_list_three_lenght,para_list_four_lenght,para_list_five_lenght,para_list_six_lenght,para_list_seven_lenght,para_list_eight_lenght,para_list_nine_lenght,para_list_ten_lenght)
print("Largest element is:", max_lenght)  

def para_selection(dict1,dict2,dict3,dict4,dict5,dict6,dict7,dict8,dict9,dict10,max_lenght): 
    final_para = [ ] 
    for index in range(max_lenght):
        # print (index) 
        para_list = [ ]
        para_lenght = { } 
        index += 1
        key = str(index) 
        # print(key) 
        if key in dict1.keys():
            print("key: ", key) 
            print("Dictioanry: 1")
            print("*************")
            para = dict1[key] 
            para_list.append(para) 
        if key in dict2.keys(): 
            print("key: ", key) 
            print("Dictioanry: 2")
            print("*************")
            para = dict2[key]
            para_list.append(para)
        if key in dict3.keys():  
            print("key: ", key) 
            print("Dictioanry: 3")
            print("*************")
            para = dict3[key]
            para_list.append(para)
        if key in dict4.keys(): 
            print("key: ", key) 
            print("Dictioanry: 4")
            print("*************")
            para = dict4[key]
            para_list.append(para)
        if key in dict5.keys(): 
            print("key: ", key) 
            print("Dictioanry: 5")
            print("*************")
            para = dict5[key]
            para_list.append(para)
        if key in dict6.keys(): 
            print("key: ", key) 
            print("Dictioanry: 6")
            print("*************")
            para = dict6[key] 
            para_list.append(para) 
        if key in dict7.keys(): 
            print("key: ", key) 
            print("Dictioanry: 7")
            print("*************")
            para = dict7[key] 
            para_list.append(para) 
        if key in dict8.keys(): 
            print("key: ", key) 
            print("Dictioanry: 8")
            print("*************")
            para = dict8[key] 
            para_list.append(para) 
        if key in dict9.keys(): 
            print("key: ", key) 
            print("Dictioanry: 9")
            print("*************")
            para = dict9[key] 
            para_list.append(para) 
        if key in dict10.keys(): 
            print("key: ", key) 
            print("Dictioanry: 10")
            print("*************")
            para = dict10[key] 
            para_list.append(para) 
        else: 
            print("***************Key not found in any***************") 

        if para_list != None: 
            for para in para_list:
                print("selected para's: ", para) 
                temp_para = para
                para_len = len(temp_para.split())  
                print("para's lenght: ", para_len)  
                para_lenght[para_len] = para 
        else: 
            print("Para list empty") 
            
        if para_lenght: 
            para_lenght_list = list(para_lenght.keys()) 
            print("para lenght list: ", para_lenght_list) 
            max_lenght = max(para_lenght_list) 
            print("Maz lenght : ", max_lenght) 
            selected_para = para_lenght[max_lenght] 
            print("Final max words para : ", selected_para) 
            selected_para_len = len(selected_para.split()) 
            if selected_para_len >= 30:
                final_para.append(selected_para) 
            else: 
                print("Selected para lenght less than 15 words: ",selected_para) 
        else: 
            print("Para list empty") 
    
    return final_para 

final_para_list = para_selection(one_selected_para,two_selected_para,three_selected_para,four_selected_para,five_selected_para,six_selected_para,seven_selected_para,eight_selected_para,nine_selected_para,ten_selected_para,max_lenght) 

print("Final para list lenght")  
print(len(final_para_list)) 

count = 1

for para in final_para_list:
    print("para # " + str(count) + " : "+ para) 
    count += 1 
    print("*********************************************")  

def paraphrase(final_list):
    paraphrased_text_list = [] 
    for para in final_para_list:  
        paraphrase = [] 
        print("para: ",para) 
        sentence_list = splitter.split(para) 
        print("sentence_list: ",sentence_list) 

        for i in sentence_list: 
            if i[0] == '.': 
                print("I == :", i) 
                pass
            else:
                a = get_response(i,1) 
                print("paraphrase sentences: ",a) 
                paraphrase.append(a) 
            
        paraphrase2 = [' '.join(x) for x in paraphrase] 
        print("first join sentences: ",paraphrase2) 
        # paraphrase3 = [' '.join(x for x in paraphrase2)] 
        # print("second join sentences: ",paraphrase3) 
        paraphrased_text = str(paraphrase2).strip('[]').strip("'")
        print("complete paraphrase sentence: ",paraphrased_text) 
        paraphrased_text_list.append(paraphrased_text) 
    
    return paraphrased_text_list 

final_paraphrase_text = paraphrase(final_para_list) 
print("Paraphrase list lenght")
print(len(final_paraphrase_text)) 

count = 1

for para in final_paraphrase_text:
    print("para # " + str(count) + " : "+ para) 
    count += 1 
    print("*********************************************")  

# pickle.dump(final_paraphrase_text,open('final_paraphrase_text_list.pkl','wb'))  

print("*************List pickle dump**********************") 


# final_paraphrase_text = pickle.load(open("final_paraphrase_text_list.pkl", 'rb')) 

print("*************Article creation**********************")  

def converted_article(paraphrase_list): 
    article="" 
    for para in paraphrase_list:
        # print(paraphrase_list[index]) 
        print("para: ",para) 
        print("*********************************************") 
        # article = article+final_paraphrase_text[index]+"\n\n" 
        article = article+para + "\n\n"

    return article

new_article = converted_article(final_paraphrase_text)  

print("*************New Article**********************")   

print(new_article) 