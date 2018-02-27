import re
import os
import nltk
import math
from random import randint

from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

tr = eval(open("train").read())
te = eval(open("test").read())

train_data = list()
test_data = list()
all_words = set()
data = list()

def extract():
	all_files = os.listdir('C:\\Users\\Rambhadra\\Desktop\\NLU\\gutenberg/')
	for file in all_files:
		sentence = []
		f = open("gutenberg/" + file)
		lines = f.readlines()
		#words = lines.split(' ')		
		for line in lines:
			words = line.split(' ')
			for word in words:
				if word != '\n' and word != '':
					if word[-1] == '.' and word not in ['Mr.','Mrs.'] :
						sentence.append(re.sub(r'[^\w\s]','',word.strip().lower()))
						data.append(' '.join(sentence))
						sentence = []
					else:
						sentence.append(re.sub(r'[^\w\s]','',word.strip().lower()))
	
	for i in range(0,len(data)):
		if i < 0.7 * len(data):
			train_data.append(data[i])
		else:
			test_data.append(data[i])



def extract_train():
	for file in tr:	
		f = open("brown/" + file)	
		data = list()
		
		for i in f.readlines():
			if i != '\n':
				data = data + [i]
				
		#doc = list()
		for i in data:
			sentence = list()
			for j in i.split(' '):
				w = j.strip().split('/')
				
				if len(w) > 1 and len(w[1]) > 0:
					pos = 'v' if (w[1][0] == 'v' or w[1][0] == 'b') else 'n'
					word = lmtzr.lemmatize(re.sub(r'[^\w\s]','',w[0].lower()) ,pos)
					all_words.add(word)
					sentence.append(word) if word is not '' else None
			sentence_combined = ' '.join(sentence)
			train_data.append(sentence_combined) if len(sentence) > 3 else None
			
		#train_data.append(doc)
		#break
		

def extract_test():
	for file in te:	
		f = open("brown/" + file)	
		data = list()
		
		for i in f.readlines():
			if i != '\n':
				data = data + [i]
				
		doc = list()
		for i in data:
			sentence = list()
			for j in i.split(' '):
				w = j.strip().split('/')
				
				if len(w) > 1 and len(w[1]) > 0:
					pos = 'v' if (w[1][0] == 'v' or w[1][0] == 'b') else 'n'
					word = lmtzr.lemmatize(re.sub(r'[^\w\s]','',w[0].lower()) ,pos)
					all_words.add(word)
					sentence.append(word) if word is not '' else None
			sentence_combined = ' '.join(sentence)
			test_data.append(sentence_combined) if len(sentence) > 3 else None
			
		#test_data.append(doc)
		#break
gram = dict()
	
def train1(ngram):
	for k in range(0,ngram):
		gram[k] = dict()
	for sentences in train_data:
		for k in range(1,ngram):
			n = k
			words = sentences.split(' ')				
			new_words = ['~'.join(words[i:i+n]) for i in range(1,len(words) - (n - 1))]
			for word in new_words:
				if word in gram[k]:
					gram[k][word] += 1
				else:
					gram[k][word] = 1
	
gram_d = dict()
def n_dict(ngram):
	for k in range(0,ngram):
		gram_d[k] = dict()
	for sentences in train_data:
		for k in range(1,ngram):
			n = k
			words = sentences.split(' ')				
			new_words = ['~'.join(words[i:i+n]) for i in range(1,len(words) - (n - 1))]
			for word in new_words:
				new = word.split('~')
				d = gram_d[k]
				for w in new:
					if w in d and new.index(w) == len(new) - 1:
						d[w] += 1
					elif w in d:
						d = d[w]
					elif new.index(w) == len(new) - 1:
						d[w] = 1
					else:
						d[w] = dict()
						d = d[w]
def create(ngram):
	for dd in range(0,10):
		try:
			mine = list()

			for i in range(1,11):
				#print(mine)
				f = 0 
				n = min(i,ngram - 1)
				d = []
				gg = -1
				for j in range(n,0,-1):
					if f == 0:				
						d = gram_d[j]
						temp = mine[-min(j,len(mine)):]
						#print(mine,j,temp,n,d)
						for k in temp:
							if temp.index(k) == len(temp) - 1 :
								if k in d:
									if isinstance(d[k],dict):
										d = d[k]
									f = 1
									gg = j
							elif k in d:
								#print("hi")
								d = d[k]		
				list_sort = []
				#print(i,gg)
				for q in d:	
					list_sort.append([d[q],q])
				#print(list_sort)
				#print(i)
				for q in list_sort:
					if isinstance(q[0],dict):
						list_sort.remove(q)
				list_sort.sort(reverse=True)
				#print(i)
				mine.append(list_sort[randint(0,min(len(d)-1, 10))][1])
			print(mine)
		except:
			pass
		
def test_n1(ngram):
	count = 0
	total = dict()
	V = len(all_words)
	for i in gram[1]:
		count += gram[1][i]
	
	for k in range(0,ngram):
		total[k] = 0
	
	for sentences in test_data:
		for k in range(1,ngram):
			total_prob = 0
			if k == 1:
				words = sentences.split(' ')
				for w in words:
					if w in gram[1]:
						c = gram[1][w]
						total_prob += math.log(c / count)
				total[1] += math.exp((-1/len(words))* total_prob)
			else:
				n = k
				words = sentences.split(' ')		
				new_words = ['~'.join(words[i:i+n]) for i in range(1,len(words) - (n - 1))]
				for i in new_words:
					c = 1
					d = count
					for j in range(0,k):
						w1 = i.split('~')
						this = '~'.join(w1[j:])
						
						if this in gram[k - j]:
							c = gram[k - j][this]
							if j == k-1:
								this2 = '~'.join(w1[j:])
							else:
								this2 = '~'.join(w1[j:-1])
							if this == this2:
								d = count
							else:
								d = gram[k - (j+1)][this2]
							break
					total_prob += math.log(c / d)
				total[k] += math.exp((-1/len(words)) * total_prob)
	for k in range(0,ngram):
		print(total[k] / len(test_data))
		

def test_n2(ngram):
	count = 0
	total = dict()
	V = len(all_words)
	for i in gram[1]:
		count += gram[1][i]
	
	for k in range(0,ngram):
		total[k] = 0
	
	for sentences in test_data:
		for k in range(1,ngram):
			total_prob = 0
			if k == 1:
				words = sentences.split(' ')
				for w in words:
					if w in gram[1]:
						c = gram[1][w]
						total_prob += math.log(c / count)
				total[1] += total_prob
			else:
				n = k
				words = sentences.split(' ')		
				new_words = ['~'.join(words[i:i+n]) for i in range(1,len(words) - (n - 1))]
				for i in new_words:
					c = 1
					d = count
					for j in range(0,k):
						w1 = i.split('~')
						this = '~'.join(w1[j:])
						if this in gram[k - j]:
							c = gram[k - j][this]
							if j == k-1:
								this2 = '~'.join(w1[j:])
							else:
								this2 = '~'.join(w1[j:-1])
							if this == this2:
								d = count
							else:
								d = gram[k - (j+1)][this2]
							break
					total_prob += math.log(c / d)
				total[k] += total_prob
	for k in range(0,ngram):
		print(math.exp((-1/count) * total[k]))



ngram = 8
#print("6 gut,brown gut")
#print("extracting")
#extract()
extract_train()
extract_test()

#print("training")
n_dict(ngram)
train1(ngram)

#print("testing")
#test_n1(ngram)
#test_n2(ngram)

print("creating")
create(ngram)


