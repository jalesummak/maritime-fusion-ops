# Projenin Türkçe özeti

Bu çalışmada uydu kaynaklı termal gözlemleri, çok dilli haberleri ve olay kayıtlarını bir araya getirerek analistin inceleyeceği adayları nasıl seçebileceğimizi araştırdım.

Eylül 2025–Şubat 2026 döneminde üç geniş çalışma bölgesini inceledim. Temizlik sonrasında 11.525 uydu gözlemi ve 3.589 haber vardı. Uydu gözlemlerini 3.289 aday termal olaya dönüştürdüm; haberlerle zaman, konum ve başlık bilgisi üzerinden ilişkilendirdim.

İlk önemli zorluk aynı sıcak kaynağın tekrar eden gözlemlerini ayrı olaylar saymaktı. Günlük gruplamayı üç günlük pencerelerle geliştirdim. Daha sonra haber eşleşmesinin gerçek çatışma anlamına gelmediğini dikkate alarak destek düzeylerini ayırdım.

Random Forest, Şubat ayında 16 haber destekli olayın 13'ünü buldu; ayrıca bu etiketi taşımayan 15 olayı işaretledi. Ancak beş aylık karşılaştırmada yalnızca bölge ve liman bilgisi kullanan model daha iyi toplu sonuç verdi. Bu bulgu, uydu yoğunluğunun bu deneylerde her ay ek değer sağlamadığını gösterdi.

UCDP, UKMTO/JMIC, liman ve kara-deniz katmanlarını ekledim. Deniz olarak sınıflanan 132 aday içinden yedi olayı Sentinel görüntüleri ve AIS/SAR bağlamıyla daha ayrıntılı inceledim. Üç olay olağan gemi veya liman bağlamıyla uyumluydu; dört olay açıklanamadı. İncelenen yedi olayda nedensel çatışma doğrulaması yapılamadı.

Son olarak Streamlit arayüzü, PostgreSQL bağlantı seçeneği, Docker paketi, güncel NASA verisi çeken bir betik ve analist geri bildirimi ekledim. Canlı uyarılar şu anda güven düzeyi ve FRP kurallarıyla üretiliyor; geçmiş Random Forest modeli canlıda çalışmıyor.

Ortaya çıkan ürün, kanıtları düzenleyen ve incelemeyi destekleyen bir prototip. Gerçek çatışmayı önceden bildiren veya kendi başına karar veren doğrulanmış bir sistem değil.

Tekrarlanabilirliğin açık sınırı: ilk BigQuery sorgusu ve dokunulmamış haber dışa aktarımı bu sürümde bulunmuyor. Eğitim notebookları sadeleştirilmiş bir yeniden kurulum; sonuçlar bölümünde özgün araştırmanın kaydedilmiş bulguları ayrıca sunuluyor.
