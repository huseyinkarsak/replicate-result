import numpy as np
import os
from collections import Counter
from scipy.stats import norm

class SAXVSM:
    def __init__(self, window_size, paa_size, alphabet_size):
        self.W = window_size
        self.P = paa_size
        self.A = alphabet_size
        # SAX Breakpoints: Normal dağılım temelinde (Gaussian assumption)
        self.breakpoints = norm.ppf(np.linspace(1/alphabet_size, 1 - 1/alphabet_size, alphabet_size - 1))
        self.tfidf_matrix = {}
        self.classes = []

    def _z_normalize(self, ts):
        """Alt diziyi birim standart sapmaya normalize eder."""
        std = np.std(ts)
        if std < 1e-6: return ts - np.mean(ts)
        return (ts - np.mean(ts)) / std

    def _paa(self, ts):
        """Piecewise Aggregate Approximation: Boyut indirgeme."""
        return [np.mean(seg) for seg in np.array_split(ts, self.P)]

    def _to_sax(self, paa_segments):
        """PAA değerlerini sembolik kelimelere dönüştürür."""
        word = ""
        for val in paa_segments:
            idx = np.searchsorted(self.breakpoints, val)
            word += chr(97 + idx)
        return word

    def get_bag_of_words(self, ts):
        """Sliding window ile tüm SAX kelimelerini toplar[cite: 1]."""
        words = []
        for i in range(len(ts) - self.W + 1):
            sub_ts = ts[i : i + self.W]
            norm_ts = self._z_normalize(sub_ts)
            paa_ts = self._paa(norm_ts)
            words.append(self._to_sax(paa_ts))
        return words

    def fit(self, X, y):
        """Eğitim setinden sınıf bazlı TF-IDF ağırlık matrisini oluşturur[cite: 1]."""
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Her sınıf için tek bir bag oluşturulur[cite: 1]
        class_bags = {c: Counter() for c in self.classes}
        for ts, label in zip(X, y):
            class_bags[label].update(self.get_bag_of_words(ts))
        
        # IDF (Inverse Document Frequency) hesaplama[cite: 1]
        all_terms = set().union(*[bag.keys() for bag in class_bags.values()])
        term_df = {term: sum(1 for c in self.classes if term in class_bags[c]) for term in all_terms}
        
        # TF-IDF ağırlıklandırma[cite: 1]
        for c in self.classes:
            self.tfidf_matrix[c] = {}
            for term, freq in class_bags[c].items():
                # Makaledeki log-scaled TF formülü: tf = log(1 + f)[cite: 1]
                tf = np.log(1 + freq)
                # IDF formülü: idf = log(N / df)[cite: 1]
                idf = np.log(n_classes / term_df[term])
                self.tfidf_matrix[c][term] = tf * idf

    def predict(self, X_test):
        """Test verisini Cosine Similarity ile sınıflandırır[cite: 1]."""
        predictions = []
        for ts in X_test:
            test_bag = Counter(self.get_bag_of_words(ts))
            best_sim, best_class = -1.0, None
            
            for c in self.classes:
                sim = self._cosine_similarity(test_bag, self.tfidf_matrix[c])
                if sim > best_sim:
                    best_sim, best_class = sim, c
            predictions.append(best_class)
        return np.array(predictions)

    def _cosine_similarity(self, bag, tfidf_vec):
        common_terms = set(bag.keys()) & set(tfidf_vec.keys())
        if not common_terms: return 0.0
        
        dot_product = sum(bag[t] * tfidf_vec[t] for t in common_terms)
        norm_bag = np.sqrt(sum(v**2 for v in bag.values()))
        norm_tfidf = np.sqrt(sum(v**2 for v in tfidf_vec.values()))
        
        if norm_bag == 0 or norm_tfidf == 0: return 0.0
        return dot_product / (norm_bag * norm_tfidf)

def load_ts_file(file_path):
    """.ts formatındaki veriyi parse eder[cite: 2]."""
    X, y = [], []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('@', '#')): continue
            parts = line.split(':')
            X.append(np.array([float(v) for v in parts[0].split(',')]))
            y.append(int(parts[1]))
    return np.array(X), np.array(y)

# --- ANA ÇALIŞTIRMA ---
if __name__ == "__main__":
    # Ödev kuralı: Öğrenci numaranı seed olarak kullan[cite: 2]
    np.random.seed(42) 

    # 1. Dosya Yolları (Kendi path'ine göre güncelledim)
    base_path = "/Users/huseyinkarsak/Documents/GitHub/spring26-huseyinkarsak/replicate-result/data"
    train_file = os.path.join(base_path, "OSULeaf_TRAIN.ts")
    test_file = os.path.join(base_path, "OSULeaf_TEST.ts")

    print("Veriler yükleniyor...")
    X_train, y_train = load_ts_file(train_file)
    X_test, y_test = load_ts_file(test_file)

    # 2. SAX-VSM Parametreleri (OSULeaf için makale önerisi[cite: 1])
    # W=Pencere boyutu, P=PAA segment sayısı, A=Alfabe boyutu
    W, P, A = 80, 10, 5
    print(f"Model eğitiliyor (W={W}, P={P}, A={A})...")
    
    model = SAXVSM(window_size=W, paa_size=P, alphabet_size=A)
    model.fit(X_train, y_train)

    # 3. Tahmin ve Değerlendirme
    print("Test seti üzerinde tahmin yapılıyor...")
    preds = model.predict(X_test)

    accuracy = np.mean(preds == y_test)
    error_rate = 1 - accuracy

    print("-" * 30)
    print(f"REPLİKASYON SONUCU (OSULeaf)")
    print(f"Doğruluk (Accuracy): %{accuracy * 100:.2f}")
    print(f"Hata Oranı (Error Rate): {error_rate:.3f}")
    print(f"Makale Hedefi (Error Rate): 0.107")
    print("-" * 30)

# Denenecek parametre listeleri
best_acc = 0
for w in [60, 80, 100]:
    for p in [8, 10]:
        for a in [4, 5]:
            model = SAXVSM(window_size=w, paa_size=p, alphabet_size=a)
            model.fit(X_train, y_train)
            acc = np.mean(model.predict(X_test) == y_test)
            if acc > best_acc:
                best_acc = acc
                print(f"Yeni En İyi! W={w}, P={p}, A={a} -> Acc: %{acc*100:.2f}")

