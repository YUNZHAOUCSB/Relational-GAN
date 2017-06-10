maxlen=57 #max length of each sentence
word2id={}
id2word={}
with open('./Cause_Effect_nosym1.txt','r') as f:
	words = f.read().split()
f.close()
for idx,item in enumerate(sorted(set(words))):#lowercase
	word2id[item] = idx
	id2word[idx] = item
print idx
word2id[' ']=idx+1
id2word[idx+1]=' '	
g=open('./converted_equal.txt','a')
j=0
with open('./Cause_Effect_nosym1.txt','r') as f:
	for line in f.readlines():
		obj=''
		count=0
		for word in line.split():
			obj=obj+str(word2id[word])+' '
			count+=1
		for i in range(maxlen-count):
			obj=obj+str(word2id[' '])+ ' '
		obj=obj+'\n'
		j+=1
		print j
		print count
		#print obj
		g.write(obj)
print j 