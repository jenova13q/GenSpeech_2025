# ASR: Распознавание русских чисел (1000..999999)

Проект группового задания: компактная CTC-модель **с нуля** для распознавания **русских числительных** по аудио.  
Вход: **16 кГц**, параметров **< 5M**. Инференс: greedy (по умолчанию) или с LM (опционально).

## Состав репозитория

- `sg_3 (1).ipynb` — **единственный ноутбук** с полным пайплайном:
  - подготовка данных (ресемплинг 16 кГц, нормализация меток),
  - модель CRDNN (Conv + BiGRU + Linear),
  - обучение CTC с аугментациями (speed perturbation, SpecAugment),
  - оценка на `dev` (CER по спикерам, H-mean inD/ooD),
  - построение KenLM,
  - экспорт артефактов для инференса.
- `README.md` — этот файл.

**Артефакты**:
- `best_infer.pt` — веса модели (`state_dict`) для инференса,
- `LABELS.json` — словарь (важно: `LABELS[0] == ""`, пробел присутствует),
- `ru_nums.klm` — бинарь KenLM.

---

## Как запустить инференс на Kaggle (submission notebook)

Создайте **публичный** Kaggle Notebook и вставьте один блок, который:
1) скачает артефакты из GitHub Release,  
2) определит модель и препроцесс (как в обучении),  
3) прогонит `test/` и сохранит `submission.csv`.

> Ниже минимальный блок без LM (быстро и надёжно на CPU). LM можно добавить отдельно.

```python
# --- 1) Скачать артефакты с GitHub Release ---
import subprocess, pathlib, json, torch, torch.nn as nn, torchaudio, pandas as pd
from pathlib import Path

ART = pathlib.Path("artifacts"); ART.mkdir(exist_ok=True)

def dl(name):
    url = f"https://github.com/jenova13q/GenSpeech_2025/tree/hw_3(group_hw_1)"
    subprocess.run(["curl","-L",url,"-o",str(ART/name)], check=True)

for f in ["best_infer.pt","LABELS.json"]:
    dl(f)  # LM не обязателен на Kaggle CPU

# --- 2) Параметры, данные, словарь ---
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DATAROOT = Path("/kaggle/input/asr-numbers-recognition-in-russian")
LABELS = json.load(open(ART/"LABELS.json","r",encoding="utf-8"))
V = len(LABELS); ID2CHAR = {i:c for i,c in enumerate(LABELS)}

# --- 3) Модель и признаки ---
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=5, s=2, p=2):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(c_in, c_out, k, stride=s, padding=p, bias=False),
                                 nn.BatchNorm1d(c_out), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)

class CRDNN(nn.Module):
    def __init__(self, n_mels, vocab_size, rnn_hidden=256, rnn_layers=2):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(n_mels,128,5,2,2),
            ConvBlock(128,   128,5,2,2),
            ConvBlock(128,   128,5,1,2),
        )
        self.proj = nn.Linear(128, rnn_hidden*2)
        self.rnn  = nn.GRU(input_size=rnn_hidden*2, hidden_size=rnn_hidden,
                           num_layers=rnn_layers, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(rnn_hidden*2, vocab_size)

    def forward(self, spec):              # [B, M, T]
        x = self.conv(spec)               # [B, C, T']
        x = x.transpose(1,2)              # [B, T', C]
        x = self.proj(x)                  # [B, T', 2H]
        x, _ = self.rnn(x)                # [B, T', 2H]
        logits = self.classifier(x)       # [B, T', V]
        return logits.transpose(0,1)      # [T', B, V]

def load_audio(path, target_sr=16000):
    wav, sr = torchaudio.load(str(path))
    if wav.size(0) > 1: wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr: wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav / max(wav.abs().max().item(), 1e-6)
    return wav.squeeze(0)

class LogMelSpec(nn.Module):
    def __init__(self, sr=16000, n_fft=400, hop=160, n_mels=80):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, win_length=n_fft, hop_length=hop,
            f_min=20.0, f_max=sr/2, n_mels=n_mels, power=2.0
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power")
    def forward(self, wav):
        S = self.mel(wav); S = self.db(S)
        return (S - S.mean()) / (S.std() + 1e-5)

def ctc_greedy(logits_t_b_v, blank_id=0):
    T,B,V = logits_t_b_v.shape; hyps=[]
    for b in range(B):
        path = logits_t_b_v[:,b,:].argmax(-1).cpu().numpy()
        last = blank_id; out=[]
        for p in path:
            p=int(p)
            if p!=last and p!=blank_id: out.append(p)
            last=p
        hyps.append("".join(ID2CHAR[i] for i in out))
    return hyps

# слова → число (диапазон) — короткая версия
import re
UNITS={"ноль":0,"один":1,"одна":1,"два":2,"две":2,"три":3,"четыре":4,"пять":5,"шесть":6,"семь":7,"восемь":8,"девять":9}
TEENS={"десять":10,"одиннадцать":11,"двенадцать":12,"тринадцать":13,"четырнадцать":14,
       "пятнадцать":15,"шестнадцать":16,"семнадцать":17,"восемнадцать":18,"девятнадцать":19}
TENS={"двадцать":20,"тридцать":30,"сорок":40,"пятьдесят":50,"шестьдесят":60,"семьдесят":70,"восемьдесят":80,"девяносто":90}
HUNDS={"сто":100,"двести":200,"триста":300,"четыреста":400,"пятьсот":500,"шестьсот":600,"семьсот":700,"восемьсот":800,"девятьсот":900}
THOUS={"тысяча","тысячи","тысяч"}
def _tok(s):
    s=s.lower(); s=re.sub(r"[^ а-яё-]"," ",s).replace("-"," "); s=re.sub(r"\s+"," ",s).strip(); return s.split()
def _chunk(tok):
    i,n=0,0
    while i<len(tok):
        w=tok[i]
        if w in HUNDS: n+=HUNDS[w]; i+=1; continue
        if w in TEENS: n+=TEENS[w]; i+=1; continue
        if w in TENS:
            n+=TENS[w]; i+=1
            if i<len(tok) and tok[i] in UNITS: n+=UNITS[tok[i]]; i+=1
            continue
        if w in UNITS: n+=UNITS[w]; i+=1; continue
        if w=="и": i+=1; continue
        break
    return n,i
def words_to_int_safe(text):
    tok=_tok(text)
    if any(t in THOUS for t in tok):
        idx=min(i for i,t in enumerate(tok) if t in THOUS)
        th,_=_chunk(tok[:idx]); rs,_=_chunk(tok[idx+1:])
        v=th*1000+rs
    else:
        v,_=_chunk(tok)
    return int(max(1000, min(999999, v)))

# --- 4) Загрузка весов и инференс ---
model = CRDNN(80, V, 256, 2).to(DEVICE).eval()
state = torch.load(ART/"best_infer.pt", map_location=DEVICE)
model.load_state_dict(state, strict=True)
feat = LogMelSpec().to(DEVICE)

def transcribe_file(path):
    wav = load_audio(path)
    spec = feat(wav).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(spec)
    hyp = ctc_greedy(logits)[0]
    return words_to_int_safe(hyp)

def run_test(test_csv, root, out_csv="submission.csv"):
    df = pd.read_csv(test_csv)
    rows=[]
    for fn in df["filename"].tolist():
        y = transcribe_file(root/fn)
        rows.append({"filename": fn, "transcription": int(y)})
    sub = pd.DataFrame(rows); sub.to_csv(out_csv, index=False)
    return out_csv, sub

out_csv, sub_df = run_test(DATAROOT/"test.csv", DATAROOT)
print("Saved:", out_csv); sub_df.head()
```

## Проверка формата перед сабмитом
``` python
# 1) базовые инварианты
assert LABELS[0] == "", "LABELS[0] должен быть пустой строкой (blank)"
assert " " in LABELS, "В LABELS должен быть пробел"
assert model.classifier.out_features == len(LABELS), "Vocab V != classifier.out_features"

# 2) формат сабмита
import pandas as pd
test_df = pd.read_csv(DATAROOT/"test.csv")
assert set(sub_df.columns) == {"filename","transcription"}
assert len(sub_df) == len(test_df)
assert sub_df["filename"].str.startswith("test/").all()
assert sub_df["transcription"].astype(int).between(1000, 999999).all()
```

## Результаты

**Dev (текст, CER):**
- Greedy: **~9.86%**
- + KenLM 3-gram (α=0.7, β=1.0): **~6.12%**

**Dev (цифры, CER; гармонич. среднее inD/ooD):**
- Greedy + fuzzy denorm: **~17.8%** H-mean

**Kaggle (официальная метрика: CER по цифрам, H-mean):**
- Greedy — Public: **27.183%**, Private: **29.800%**
- **+ LM (3-gram, α=0.7, β=1.0)** — Public: **10.944%**, Private: **14.472%**

*Причины разницы dev↔Kaggle: digits-CER строже, а также на Kaggle без LM (CPU) качество ниже; с LM улучшается значительно (но проблемы с ```No module named 'kenlm'```).*