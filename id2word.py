import numpy as np
word2id={}
id2word={}
with open('./Cause_Effect_nosym1.txt','r') as f:
	words = f.read().split()
for idx,item in enumerate(sorted(set(words))):
	word2id[item] = idx
	id2word[idx] = item	
word2id[' ']=idx+1
id2word[idx+1]=' '
print idx	

g=open('./generated_seq_withoutadversarial.txt','w')

f=np.load('./generate_file_withoutadversarial.txt.npy')
for line in f:
	obj=''
	for id in np.array2string(line)[1:-1].split():
		if int(id)>4436:
			id=4436
		obj=obj+id2word[int(id)]+' '
	obj=obj+'\n'
	print obj
	g.write(obj)