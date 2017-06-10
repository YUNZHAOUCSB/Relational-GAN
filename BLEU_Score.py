import nltk

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

f=open('Cause_Effect_nosym1.txt')
ref=f.readlines()
g=open('generated_seq_withoutadversarial.txt')
i=0
score=[]
for line in g.readlines():
	score.append(nltk.translate.bleu_score.sentence_bleu(ref,line))
	i+=1
	print i
print "mean:%s"%mean(score)
