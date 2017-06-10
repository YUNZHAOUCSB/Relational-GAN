maxlen=0 #max length of each sentence
word2id={}
id2word={}
with open('./Cause_Effect_nosym1.txt','r') as f:
	words = f.read().split()
for idx,item in enumerate(sorted(set(words))):
	word2id[item] = idx
	id2word[idx] = item
word2id[' ']=idx+1
id2word[idx+1]=' '	
#f.close()

g=open('./converted.txt','a')
obj=''
with open('./Cause_Effect_nosym1.txt','r') as f:
	for line in f.readlines():
		maxlen=max(len(line.split()),maxlen)
		for word in line.split():
			obj=obj+str(word2id[word])+' '
		obj=obj+'\n'
		print obj
		g.write(obj)
print maxlen