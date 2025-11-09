import sacrebleu

references = [["this is a pen", "that is a pencil"]]
candidates = ["this is a pen"]

score = sacrebleu.corpus_bleu(candidates, references)
print(score.score)