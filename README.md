# 🧠 Türkçe Doğal Dil İşleme (NLP) ve Duygu Analizi

Ürün yorumları üzerinde **Türkçe kök bulma (Stemming)**, **N-Gram vektörleştirme (`CountVectorizer`)** ve **Multinomial Naive Bayes** sınıflandırması kullanarak duygu analizi (Sentiment Analysis) ve hibrit ürün önerisi gerçekleştiren Python projesi.

## ✨ Özellikler

- **Türkçe Ön İşleme:** `NLTK` ve `TurkishStemmer` ile noktalama işaretlerinin temizlenmesi, Türkçe etkisiz kelimelerin (*stop-words*) filtrelenmesi ve kelime köklerinin çıkarılması.
- **N-Gram Vektörleştirme:** `CountVectorizer(ngram_range=(1, 2))` ile tekli ve ikili kelime gruplarının çıkarılması.
- **Duygu Sınıflandırması:** `MultinomialNB` modeli ile yorumların Olumlu/Olumsuz olarak sınıflandırılması ve doğruluk/sınıflandırma raporunun üretilmesi.
- **Makro & Mikro Analiz Paneli:** Genel memnuniyet oranının hesaplanması, %50 kritik eşik altında alternatif ürün önerisi (İçerik Tabanlı Filtreleme) ve en iyi/en kötü 5 yorumun güven skorlarıyla sıralanması.
- **İnteraktif Test Modu:** Kullanıcının terminalden girdiği yeni yorumların anlık olarak analiz edilmesi.

## 🛠️ Kurulum ve Çalıştırma

1. **Depoyu klonlayın:**
   ```bash
   git clone https://github.com/Firatdondar/do-al-dil-i-leme.git
   cd do-al-dil-i-leme
   ```

2. **Gerekli kütüphaneleri yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Analizi başlatın:**
   ```bash
   python analiz.py
   ```
