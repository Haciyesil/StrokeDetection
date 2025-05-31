# StrokeDetection

StrokeDetection, 3D Slicer platformu üzerinde çalışan, MRI görüntülerinde inme bölgelerini otomatik olarak tespit etmek ve segmentasyon yapmak için geliştirilmiş bir Python modülüdür. Derin öğrenme tabanlı bu modül, klinik uygulamalarda inme tanısını desteklemek için tasarlanmıştır.

## Özellikler

- UNet derin öğrenme modeli kullanarak inme segmentasyonu
- Test zamanı veri artırma (TTA) ile daha güvenilir tahminler
- 3D tutarlılık kontrolü ve hacim analizi
- Dilim kalınlığına göre adaptif eşik değeri hesaplama
- Detaylı metrik hesaplama (Dice, Sensitivity, Specificity, vb.)
- Beyin maskesi oluşturma ve akıllı ön işleme teknikleri
- Görselleştirme seçenekleri (2D overlay ve 3D model)
- En iyi vakalardan model ince ayarı yapma
- Kapsamlı AISD veri seti analizi

## Gereksinimler

- Python 3.10+
- 3D Slicer 5.0+
- PyTorch 1.10+
- NumPy, SciPy, OpenCV
- MONAI (opsiyonel)

## Kurulum

1. 3D Slicer'ı [resmi sitesinden](https://download.slicer.org/) indirin ve kurun
2. Bu repository'yi klonlayın:
   ```
   git clone https://github.com/Haciyesil/StrokeDetection.git
   ```
3. Klasörü 3D Slicer'ın modül dizinine kopyalayın:
   - Windows: `%USERPROFILE%\AppData\Roaming\NA-MIC\Slicer\Modules`
   - macOS: `~/Library/Application Support/NA-MIC/Slicer/Modules`
   - Linux: `~/.config/NA-MIC/Slicer/Modules`
4. 3D Slicer'ı yeniden başlatın

## Kullanım

1. 3D Slicer'ı açın
2. Modules menüsünden "StrokeDetection" modülünü seçin
3. Görüntü ve maske klasörlerini belirtin
4. Segmentasyon modeli (.pth dosyası) seçin
5. Eşik değerini ayarlayın (varsayılan: 0.65)
6. "Segment Selected Image" butonuna tıklayın
7. Sonuçları ve metrikleri inceleyin

## Gelişmiş Özellikler

- **AISD Veri Seti Analizi**: "AISD verisetini segmente et" butonu ile veri seti üzerinde toplu analiz yapabilirsiniz
- **Model İnce Ayarı**: "En İyi Vakalardan Model Eğit ve Segmente Et" butonu ile en iyi vakaları kullanarak modeli ince ayarlayabilirsiniz
- **3D Görselleştirme**: "Show 3D Visualization" seçeneği ile segmentasyon sonuçlarını 3D olarak görselleştirebilirsiniz
- **Model Debugging**: "Debug Model" butonu ile model dosyasını detaylı analiz edebilirsiniz

## Hata Ayıklama

Eğer modül yüklenirken veya çalışırken sorun yaşıyorsanız:

1. 3D Slicer'ın Python konsolunu açın (View > Python Interactor)
2. Hata mesajlarını kontrol edin
3. Gerekli kütüphanelerin yüklü olduğundan emin olun
4. GPU desteği için CUDA kurulumunuzu kontrol edin

## Katkı Sağlama

Projeye katkıda bulunmak için:

1. Bu repository'yi fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeniozelllik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik: X eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeniozellik`)
5. Pull Request açın

## Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.

## İletişim

Sorularınız veya geri bildirimleriniz için lütfen iletişime geçin:
haciyesil8@gmail.com
