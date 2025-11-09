JA_train = "data/tok/kyoto-train.ja"
EN_train = "data/tok/kyoto-train.en"

JA_dev = "data/tok/kyoto-dev.ja"
EN_dev = "data/tok/kyoto-dev.en"

JA_test = "data/tok/kyoto-test.ja"
EN_test = "data/tok/kyoto-test.en"

def load_data(ja,en):
    with open(ja) as f:
        ja_lines = f.readlines()
    with open(en) as f:
        en_lines = f.readlines()
    return len(ja_lines) if len(ja_lines) == len(en_lines) else (ja_lines,en_lines)
print(f"train:{load_data(JA_train,EN_train)}行")
print(f"dev:{load_data(JA_dev,EN_dev)}行")
print(f"test:{load_data(JA_test,EN_test)}行")