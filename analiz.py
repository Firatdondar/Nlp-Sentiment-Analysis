import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import nltk
import sys
from TurkishStemmer import TurkishStemmer

# Türkçe kök bulucu (Stemmer) kütüphanesini tanımlıyoruz
nltk.download(['punkt', 'punkt_tab', 'stopwords'], quiet=True)
from nltk.corpus import stopwords

stemmer = TurkishStemmer()
turkce_stop_words = set(stopwords.words('turkish'))

def turkce_kok_bul(metin):
    # Cümleyi kelimelerine ayırıp noktalama/stop-word filtrelemesi yaparak kökleri buluyoruz
    kelimeler = nltk.word_tokenize(metin.lower())
    kokler = [stemmer.stem(kelime) for kelime in kelimeler if kelime.isalpha() and kelime not in turkce_stop_words]
    return " ".join(kokler)

# =====================================================================
# ADIM 1: VERİ KÜMESİ VE HİBRİT DERLEM (CORPUS) OLUŞTURMA [cite: 8, 10, 43]
# =====================================================================
# Kendi hazırladığımız harici txt dosyasından veri setimizi okuyoruz.
# (Ayraç olarak '|' kullandık, sütun isimlerini 'yorum' ve 'duygu' olarak belirledik)
try:
    df = pd.read_csv('yorumlar.txt', sep='|', names=['yorum', 'duygu'])
except FileNotFoundError:
    print("\nHATA: 'yorumlar.txt' dosyası bulunamadı! Lütfen dosyanın analiz.py ile aynı dizinde olduğundan emin olun.\n")
    sys.exit(1)

# İYİLEŞTİRME 1: Tüm yorumları Türkçe köklerine indirgiyoruz
df['temiz_yorum'] = df['yorum'].apply(turkce_kok_bul)

# =====================================================================
# ADIM 1.5: VERİ SETİNİN EĞİTİM VE TEST OLARAK AYRILMASI
# =====================================================================
# Verinin %80'i modelin eğitimi, %20'si test edilmesi için ayrılır.
X_train, X_test, y_train, y_test = train_test_split(
    df['temiz_yorum'], 
    df['duygu'], 
    test_size=0.20, 
    random_state=42
)

# =====================================================================
# ADIM 2: VEKTÖRLEŞTİRME (COUNTVECTORIZER) [cite: 24, 25, 75]
# =====================================================================
# ngram_range=(1, 2) diyerek hem tekli kelimeleri hem de yan yana gelen ikilileri saydırıyoruz.
vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print("--- YENİ GELİŞMİŞ SÖZLÜK (TEKLİ VE İKİLİ KELİME GRUPLARI) ---")
print(vectorizer.get_feature_names_out())
print("-" * 40)

# =====================================================================
# ADIM 3: MULTINOMIAL NAIVE BAYES MODELİNİN EĞİTİLMESİ [cite: 28, 82]
# =====================================================================
# Bayes Teoremi tabanlı istatistiksel NLP modelini eğitiyoruz[cite: 27, 29].
clf_model = MultinomialNB()
clf_model.fit(X_train_vectorized, y_train)

# =====================================================================
# ADIM 3.5: MODELİN EVALUASYONU VE BAŞARIM ANALİZİ
# =====================================================================
y_pred = clf_model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*20 + " MODEL BAŞARIM RAPORU " + "="*20)
print(f"Model Genel Doğruluk Oranı (Accuracy): %{accuracy * 100:.2f}")
print("\nSınıflandırma Metrikleri:")
print(classification_report(y_test, y_pred, target_names=['Olumsuz', 'Olumlu']))

# =====================================================================
# ADIM 4: MAKRO DÜZEYDE GENEL ORANLARIN HESAPLANMASI
# =====================================================================
# Veri setindeki tüm yorumların model tarafından tahmin edilen genel duygu dağılımı hesaplanır.
tum_tahminler = clf_model.predict(vectorizer.transform(df['temiz_yorum']))
toplam_yorum_y = len(tum_tahminler)

# Sınıflandırma modelindeki 2. sınıfı (index 1) 'Olumlu' kabul ediyoruz (Adım 6'daki mantıkla paralel olarak)
olumlu_sayisi_y = np.sum(tum_tahminler == clf_model.classes_[1])
olumsuz_sayisi_y = toplam_yorum_y - olumlu_sayisi_y

genel_olumlu_oran = (olumlu_sayisi_y / toplam_yorum_y) * 100
genel_olumsuz_oran = (olumsuz_sayisi_y / toplam_yorum_y) * 100

print("\n" + "="*20 + " GENEL MAKRO ÖZET PANELİ " + "="*20)
print(f"İncelenen Ürün: Y Telefonu (Kategori: Telefon)")
print(f"Toplam Analiz Edilen Yorum Sayısı: {toplam_yorum_y}")

if genel_olumlu_oran >= genel_olumsuz_oran:
    print(f"GENEL DURUM: [ OLUMLU ]")
    print(f"Açıklama: İncelemelerin %{genel_olumlu_oran:.2f}'si OLUMLU, %{genel_olumsuz_oran:.2f}'si OLUMSUZ yöndedir.")
else:
    print(f"GENEL DURUM: [ OLUMSUZ ]")
    print(f"Açıklama: İncelemelerin %{genel_olumsuz_oran:.2f}'si OLUMSUZ, %{genel_olumlu_oran:.2f}'si OLUMLU yöndedir.")
print("="*65)

# =====================================================================
# ADIM 5: HİBRİT ÖNERİ SİSTEMİNİN DEVREYE GİRMESİ
# =====================================================================
# Eğer ürünün memnuniyet oranı %50 kritik eşiğinin altındaysa İçerik Tabanlı Filtreleme tetiklenir.
KRITIK_ESIK = 50.0

if genel_olumlu_oran < KRITIK_ESIK:
    print("\n⚠️ UYARI: Bu ürün hakkında kullanıcılar tarafından yoğun olumsuz geri bildirim yapılmıştır!")
    print("💡 Siteden ayrılmak yerine, aynı kategorideki en yüksek memnuniyetli şu alternatife göz atabilirsiniz:")
    
    # Simülasyonumuzdaki diğer ürün olan X Telefonu'nun memnuniyet oranının %90 olduğunu belirtiyoruz
    print("-> X Telefonu (Müşteri Memnuniyet Oranı: %90.00)")
    print("-" * 65)

# =====================================================================
# ADIM 6: MİKRO GÜVEN SKORU VE DUYGU YOĞUNLUĞUNA GÖRE AKILLI SIRALAMA
# =====================================================================
# Veri setindeki (yorumlar.txt) tüm yorumların olasılıklarını modelimize hesaplatıyoruz.
tum_vektorler = vectorizer.transform(df['temiz_yorum'])
tum_olasiliklar = clf_model.predict_proba(tum_vektorler)

yorum_skorlari = []
for i, yorum in enumerate(df['yorum']):
    olumsuz_yuzde = tum_olasiliklar[i][0] * 100
    olumlu_yuzde = tum_olasiliklar[i][1] * 100
    yorum_skorlari.append({'yorum': yorum, 'olumsuz_yuzde': olumsuz_yuzde, 'olumlu_yuzde': olumlu_yuzde})

# Listeleri olumsuzluk ve olumluluk yüzdelerine göre ayrı ayrı büyükten küçüğe sıralıyoruz
olumsuz_sirali = sorted(yorum_skorlari, key=lambda x: x['olumsuz_yuzde'], reverse=True)
olumlu_sirali = sorted(yorum_skorlari, key=lambda x: x['olumlu_yuzde'], reverse=True)

print("\n" + "-"*15 + " MİKRO DETAY: EN KÖTÜ 5 YORUM " + "-"*15)
for i, res in enumerate(olumsuz_sirali[:5], 1):
    print(f"Sıra #{i} [Yapay Zeka Olumsuzluk Derecesi: %{res['olumsuz_yuzde']:.2f}]")
    print(f"Metin: '{res['yorum']}'\n")

print("-" * 15 + " MİKRO DETAY: EN İYİ 5 YORUM " + "-" * 15)
for i, res in enumerate(olumlu_sirali[:5], 1):
    print(f"Sıra #{i} [Yapay Zeka Olumluluk Derecesi: %{res['olumlu_yuzde']:.2f}]")
    print(f"Metin: '{res['yorum']}'\n")
print("-" * 80)

while True:
    secim = input("\nGeri kalan mikroları (diğer tüm yorumları) da görmek ister misiniz? (E/H): ").upper().strip()
    if secim == 'E':
        # En kötü 5 ve en iyi 5 arasına düşen geri kalan 90 yorumu alıyoruz (5'ten başlayıp sondan 5'e kadar)
        geri_kalanlar = olumsuz_sirali[5:-5]
        
        print("\n" + "-"*15 + " GERİ KALAN YORUMLAR (Olumsuzdan Olumluya Sıralı) " + "-"*15)
        for i, res in enumerate(geri_kalanlar, 6):
            print(f"Sıra #{i} [Olumsuzluk Derecesi: %{res['olumsuz_yuzde']:.2f}]")
            print(f"Metin: '{res['yorum']}'\n")
        print("-" * 80)
        break # Döngüden çık
    elif secim == 'H' or secim == '': # 'H' veya boş Enter tuşunu "Hayır" olarak kabul et
        print("\nKalan detayların incelemesi iptal edildi. Programın sonraki aşamasına geçiliyor.")
        break # Döngüden çık
    else:
        print("Geçersiz seçim. Lütfen sadece 'E' (Evet) veya 'H' (Hayır) girin.")

# =====================================================================
# ADIM 7: KENDİ YORUMUNU TEST ET (İNTERAKTİF)
# =====================================================================
print("\n" + "*"*65)
test_yorumu = input("Yapay zekayı test etmek için ürün hakkında bir yorum yazın (Çıkmak için boş bırakıp Enter'a basın): ")

if test_yorumu.strip():
    # Kullanıcının girdiği test yorumunu da aynı kök bulma işleminden geçiriyoruz
    test_yorumu_temiz = turkce_kok_bul(test_yorumu)
    test_vektoru = vectorizer.transform([test_yorumu_temiz])
    olasilik = clf_model.predict_proba(test_vektoru)
    
    olumsuz_skoru = olasilik[0][0] * 100
    olumlu_skoru = olasilik[0][1] * 100
    
    print("\n--- YAPAY ZEKA ANALİZ SONUCU ---")
    print(f"Senin Yorumun: '{test_yorumu}'")
    print(f"Olumlu Olma İhtimali: %{olumlu_skoru:.2f}")
    print(f"Olumsuz Olma İhtimali: %{olumsuz_skoru:.2f}")
    print("*"*65)