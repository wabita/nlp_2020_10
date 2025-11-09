import subprocess

def ja2en(text):
    
    tra = subprocess.run(["bash", "pro/92/92.sh"],input = text+"\n" , text=True, capture_output=True)
    result= tra.stdout 
    for line in result.splitlines():
        if line.startswith("H-"):
            txt=line.split("\t")
            return txt[2]

ja= "晩年 に 希 玄 と い う 異称 も 用い た 。"
en= "Later in his life he also went by the name Kigen ."
tr_en = ja2en(ja)
print( f"翻訳結果 : {tr_en}" )
print( f"正解例 : {en}" )
