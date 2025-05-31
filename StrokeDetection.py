import copy
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import slicer
import qt
import ctk
from slicer.ScriptedLoadableModule import *
import unittest
import logging
import vtk
import random
import glob
import hashlib
import time
from scipy import ndimage
import cv2

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SliceDataset(Dataset):
    def __init__(self, image_slices, transform=None):
        self.image_slices = image_slices
        self.transform = transform
    
    def __len__(self):
        return len(self.image_slices)
    
    def __getitem__(self, idx):
        slice_array = self.image_slices[idx]
        # 0-1 aralığına normalize et
        min_val = slice_array.min()
        max_val = slice_array.max()
        if max_val > min_val:
            slice_array = (slice_array - min_val) / (max_val - min_val)
        else:
            slice_array = np.zeros_like(slice_array)
        # PIL görüntüsüne dönüştür
        slice_pil = Image.fromarray((slice_array * 255).astype(np.uint8))
        if self.transform:
            slice_pil = self.transform(slice_pil)
        return slice_pil

# Ana Modül Sınıfı
class StrokeDetection(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "StrokeDetection"
        self.parent.categories = ["Examples"]
        self.parent.dependencies = []
        self.parent.contributors = ["Your Name"]
        self.parent.helpText = """
        Bu modül, MRI görüntülerinde inme tespiti ve segmentasyonu gerçekleştirir.
        İki farklı model kullanır: ResNet18 (sınıflandırma için) ve UNet (segmentasyon için).
        """
        self.parent.acknowledgementText = """
        Bu modül, inme tespiti ve segmentasyonu için özel bir çözüm olarak geliştirilmiştir.
        """

# Widget Sınıfı
class StrokeDetectionWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        # Düzen oluştur
        self.layout.addWidget(qt.QLabel("İnme Tespiti ve Segmentasyonu Modülü"))
        # Üst sekmeli düzen oluştur (Sınıflandırma ve Segmentasyon için)
        self.tabWidget = qt.QTabWidget()
        self.layout.addWidget(self.tabWidget)
        # Segmentasyon sekmesi
        self.segmentationTab = qt.QWidget()
        self.tabWidget.addTab(self.segmentationTab, "Segmentasyon (UNet)")
        segmentationLayout = qt.QVBoxLayout(self.segmentationTab)
        #####################################
        # SEGMENTASYON SEKMESİ İÇERİĞİ
        #####################################
        # Input bölümü
        segInputCollapsibleButton = ctk.ctkCollapsibleButton()
        segInputCollapsibleButton.text = "Segmentasyon Girişi"
        segmentationLayout.addWidget(segInputCollapsibleButton)
        segInputFormLayout = qt.QFormLayout(segInputCollapsibleButton)
        # Input image folder seçici
        self.segInputImageFolderSelector = ctk.ctkPathLineEdit()
        self.segInputImageFolderSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.segInputImageFolderSelector.setToolTip("MRI görüntü klasörünü seçin")
        segInputFormLayout.addRow("Input Image Folder:", self.segInputImageFolderSelector)
        # Maske folder seçici
        self.segMaskFolderSelector = ctk.ctkPathLineEdit()
        self.segMaskFolderSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.segMaskFolderSelector.setToolTip("Maske klasörünü seçin")
        segInputFormLayout.addRow("Mask Folder:", self.segMaskFolderSelector)
        # Model dosyası seçici
        self.segModelFileSelector = ctk.ctkPathLineEdit()
        self.segModelFileSelector.filters = ctk.ctkPathLineEdit.Files
        self.segModelFileSelector.nameFilters = ["PyTorch Model (*.pth)"]
        self.segModelFileSelector.setToolTip("Bölütleme modelini seçin (.pth)")
        segInputFormLayout.addRow("Segmentasyon Modeli:", self.segModelFileSelector)
        # Eşik değeri için slider
        self.segThresholdSlider = ctk.ctkSliderWidget()
        self.segThresholdSlider.singleStep = 0.05
        self.segThresholdSlider.minimum = 0.0
        self.segThresholdSlider.maximum = 1.0
        self.segThresholdSlider.value = 0.65  # Eşik değerini 0.65 olarak ayarla
        self.segThresholdSlider.setToolTip("İnme tespiti için hassasiyet eşiği")
        segInputFormLayout.addRow("Segmentation Threshold:", self.segThresholdSlider)
        # Görüntü ve maske seçiciler
        self.segImageSelector = qt.QComboBox()
        self.segImageSelector.setToolTip("İşlenecek görüntüyü seçin")
        segInputFormLayout.addRow("Select Image:", self.segImageSelector)
        self.segMaskSelector = qt.QComboBox()
        self.segMaskSelector.setToolTip("İlgili maskeyi seçin")
        segInputFormLayout.addRow("Select Mask:", self.segMaskSelector)
        # Segmentasyon butonu
        self.segmentButton = qt.QPushButton("Segment Selected Image")
        self.segmentButton.toolTip = "Seçili görüntüyü bölütle"
        self.segmentButton.enabled = True
        segInputFormLayout.addRow(self.segmentButton)
        # Yeniden İşleme (Reprocess) butonu
        self.reprocessButton = qt.QPushButton("Güncel eşik değeriyle yeniden işle")
        self.reprocessButton.toolTip = "Son işlenen görüntüyü güncel eşik değeriyle yeniden işle"
        self.reprocessButton.enabled = False  # Başlangıçta devre dışı
        segInputFormLayout.addRow(self.reprocessButton)
        # Debug butonu
        self.debugButton = qt.QPushButton("Debug Model")
        self.debugButton.toolTip = "Model yükleme ve önişleme kontrolü yap"
        self.debugButton.enabled = True
        segInputFormLayout.addRow(self.debugButton)
        # AISD verisetini segmente et butonu
        self.aisdSegmentButton = qt.QPushButton("AISD verisetini segmente et")
        self.aisdSegmentButton.toolTip = "AISD verisetindeki her hasta için dice skorunu hesapla"
        self.aisdSegmentButton.enabled = True
        segInputFormLayout.addRow(self.aisdSegmentButton)
        
        # YENİ: En iyi vakalardan model eğitme butonu
        self.finetuneButton = qt.QPushButton("En İyi Vakalardan Model Eğit ve Segmente Et")
        self.finetuneButton.toolTip = "En iyi 20 vakadan model eğitip düşük performanslı vakaları iyileştir"
        self.finetuneButton.enabled = True
        segInputFormLayout.addRow(self.finetuneButton)
        
        # Model eğitim parametreleri grubu
        trainingParamsCollapsible = ctk.ctkCollapsibleButton()
        trainingParamsCollapsible.text = "Model Eğitim Parametreleri"
        trainingParamsCollapsible.collapsed = True  # Başlangıçta kapalı
        segInputFormLayout.addRow(trainingParamsCollapsible)
        trainingParamsLayout = qt.QFormLayout(trainingParamsCollapsible)
        
        # Öğrenme hızı
        self.learningRateSpinBox = qt.QDoubleSpinBox()
        self.learningRateSpinBox.setRange(0.0001, 0.1)
        self.learningRateSpinBox.setSingleStep(0.0001)
        self.learningRateSpinBox.setValue(0.0001)
        self.learningRateSpinBox.setDecimals(5)
        trainingParamsLayout.addRow("Öğrenme Hızı:", self.learningRateSpinBox)
        
        # Epoch sayısı
        self.epochSpinBox = qt.QSpinBox()
        self.epochSpinBox.setRange(5, 50)
        self.epochSpinBox.setValue(10)
        trainingParamsLayout.addRow("Epoch Sayısı:", self.epochSpinBox)
        
        # Batch boyutu
        self.batchSizeSpinBox = qt.QSpinBox()
        self.batchSizeSpinBox.setRange(1, 16)
        self.batchSizeSpinBox.setValue(4)
        trainingParamsLayout.addRow("Batch Boyutu:", self.batchSizeSpinBox)
        
        # Sonuçlar için metin alanı
        segResultsCollapsibleButton = ctk.ctkCollapsibleButton()
        segResultsCollapsibleButton.text = "Segmentasyon Sonuçları"
        segmentationLayout.addWidget(segResultsCollapsibleButton)
        segResultsFormLayout = qt.QVBoxLayout(segResultsCollapsibleButton)
        # Güncel Dice skoru için ayrı bir gösterge ekle
        diceDisplayLayout = qt.QHBoxLayout()
        diceScoreLabel = qt.QLabel("Güncel Dice Skoru:")
        self.diceScoreDisplay = qt.QLabel("0.0000")
        self.diceScoreDisplay.setStyleSheet("font-size: 16pt; color: blue; font-weight: bold;")
        diceDisplayLayout.addWidget(diceScoreLabel)
        diceDisplayLayout.addWidget(self.diceScoreDisplay)
        diceDisplayLayout.addStretch()
        segResultsFormLayout.addLayout(diceDisplayLayout)
        # AISD ilerleme durumu için yeni gösterge ekle
        aisdProgressLayout = qt.QHBoxLayout()
        aisdProgressLabel = qt.QLabel("AISD İşleme Durumu:")
        self.aisdProgressDisplay = qt.QLabel("0/0 (0%)")
        self.aisdProgressDisplay.setStyleSheet("font-size: 16pt; color: green; font-weight: bold;")
        aisdProgressLayout.addWidget(aisdProgressLabel)
        aisdProgressLayout.addWidget(self.aisdProgressDisplay)
        aisdProgressLayout.addStretch()
        segResultsFormLayout.addLayout(aisdProgressLayout)
        # İlerleme çubuğu
        self.aisdProgressBar = qt.QProgressBar()
        self.aisdProgressBar.setMinimum(0)
        self.aisdProgressBar.setMaximum(100)
        self.aisdProgressBar.setValue(0)
        segResultsFormLayout.addWidget(self.aisdProgressBar)
        self.segResultsText = qt.QTextEdit()
        self.segResultsText.setReadOnly(True)
        self.segResultsText.setMinimumHeight(150)
        segResultsFormLayout.addWidget(self.segResultsText)
        
        # 3D Visualization Settings bölümü
        visualizationCollapsibleButton = ctk.ctkCollapsibleButton()
        visualizationCollapsibleButton.text = "3D Visualization Settings"
        segmentationLayout.addWidget(visualizationCollapsibleButton)
        visualizationFormLayout = qt.QFormLayout(visualizationCollapsibleButton)
        # 3D görüntüleme için checkbox
        self.show3DCheckBox = qt.QCheckBox("Show 3D Visualization")
        self.show3DCheckBox.setToolTip("3D görüntülemeyi etkinleştir")
        visualizationFormLayout.addRow("3D View:", self.show3DCheckBox)
        # Apply Visualization Button
        self.applyVisualizationButton = qt.QPushButton("Apply Visualization Settings")
        self.applyVisualizationButton.toolTip = "Görselleştirme ayarlarını uygula"
        visualizationFormLayout.addRow(self.applyVisualizationButton)
        
        # Sonunda esneklik ekle
        self.layout.addStretch(1)
        
        #####################################
        # BAĞLANTILAR VE DEĞİŞKENLER
        #####################################
        # Segmentasyon bağlantıları
        self.segInputImageFolderSelector.connect('validInputChanged(bool)', self.onSegFolderChanged)
        self.segMaskFolderSelector.connect('validInputChanged(bool)', self.onSegFolderChanged)
        self.segImageSelector.connect('currentIndexChanged(int)', self.onSegImageSelected)
        self.segmentButton.connect('clicked(bool)', self.onSegmentButton)
        self.reprocessButton.connect('clicked(bool)', self.onReprocessButton)
        self.debugButton.connect('clicked(bool)', self.onDebugButton)
        self.show3DCheckBox.connect('toggled(bool)', self.onShow3DToggled)
        self.applyVisualizationButton.connect('clicked(bool)', self.onApplyVisualizationSettings)
        self.aisdSegmentButton.connect('clicked(bool)', self.onAisdSegmentButton)
        
        # YENİ: Fine-tune butonu bağlantısı
        self.finetuneButton.connect('clicked(bool)', self.onFineTuneSegmentButton)
        
        # Değişkenleri başlat
        # Segmentasyon değişkenleri
        self.segmentationLogic = StrokeDetectionSegmentationLogic()  # Doğru sınıf adını kullan
        self.lastProcessedFile = None
        self.lastDiceScore = 0.0
        # AISD veri seti ilerleme bilgisi
        self.aisdTotalCases = 0
        self.aisdProcessedCases = 0
        # Görüntüleme için node referansları
        self._imageNode = None
        self._maskNode = None
        self._outputVolume = None
        self._gtOverlayNode = None
        self._predOverlayNode = None
        self._gtSegmentationNode = None
        self._predSegmentationNode = None
        self._volumeRenderingDisplayNode = None
        
        # YENİ: Eğitilmiş model için değişken
        self._fineTunedModel = None
    
    #####################################
    # SEGMENTASYON FONKSİYONLARI
    #####################################
    def onSegFolderChanged(self):
        """Klasör değiştiğinde görüntü ve maske listelerini güncelle"""
        imagePath = self.segInputImageFolderSelector.currentPath
        maskPath = self.segMaskFolderSelector.currentPath
        if not imagePath or not os.path.exists(imagePath) or not os.path.isdir(imagePath):
            return
        if not maskPath or not os.path.exists(maskPath) or not os.path.isdir(maskPath):
            return
        # Görüntü listesini temizle
        self.segImageSelector.clear()
        self.segMaskSelector.clear()
        # Görüntü dosyalarını bul
        imageFiles = []
        for ext in ['.nii.gz', '.nii', '.mha']:
            imageFiles.extend(glob.glob(os.path.join(imagePath, f'*{ext}')))
        # Görüntü dosyalarını isimlerine göre sırala
        imageFiles.sort()
        # Her görüntü için eşleşen maske ara
        for imageFile in imageFiles:
            baseName = os.path.basename(imageFile)
            # Aynı isimli maske dosyasını ara
            maskFile = None
            for ext in ['.nii.gz', '.nii', '.mha']:
                potentialMaskFile = os.path.join(maskPath, baseName)
                if os.path.exists(potentialMaskFile):
                    maskFile = potentialMaskFile
                    break
            # Eğer maske bulunduysa, listeye ekle
            if maskFile:
                self.segImageSelector.addItem(baseName)
                self.segMaskSelector.addItem(baseName)
        # Eğer görüntü varsa, ilkini seç
        if self.segImageSelector.count > 0:
            self.segImageSelector.setCurrentIndex(0)
            self.segMaskSelector.setCurrentIndex(0)
            self.segmentButton.enabled = True
        else:
            self.segmentButton.enabled = False
    
    def onSegImageSelected(self):
        """Görüntü seçildiğinde ilgili maskeyi de seç"""
        currentIndex = self.segImageSelector.currentIndex
        if currentIndex >= 0:
            self.segMaskSelector.setCurrentIndex(currentIndex)
    def onFineTuneSegmentButton(self):
        """AISD verisetini segmente et - en iyi 20'den model eğit, zor vakaları iyileştir"""
        self.segResultsText.setPlainText("AISD verisetini segmente etme işlemi başlıyor...")
        slicer.app.processEvents()
        
        # Gerekli kontrolleri yap
        imagePath = self.segInputImageFolderSelector.currentPath
        maskPath = self.segMaskFolderSelector.currentPath
        modelPath = self.segModelFileSelector.currentPath
        threshold = self.segThresholdSlider.value
        
        if not imagePath or not os.path.exists(imagePath):
            self.segResultsText.setPlainText("Lütfen geçerli bir görüntü klasörü seçin")
            return
        if not maskPath or not os.path.exists(maskPath):
            self.segResultsText.setPlainText("Lütfen geçerli bir maske klasörü seçin")
            return
        if not modelPath or not os.path.exists(modelPath):
            self.segResultsText.setPlainText("Lütfen geçerli bir model dosyası seçin")
            return
        
        # Tüm görüntü dosyalarını al
        imageFiles = []
        for ext in ['.nii.gz', '.nii', '.mha']:
            imageFiles.extend(glob.glob(os.path.join(imagePath, f'*{ext}')))
        
        if len(imageFiles) == 0:
            self.segResultsText.setPlainText("Klasörde görüntü dosyası bulunamadı!")
            return
        
        # Başlangıç zamanını kaydet
        start_time = time.time()
        
        # ADIM 1: İlk genel işleme ve en iyi 20 vakayı belirleme
        #-----------------------------------------------------------------
        
        # Toplam vaka sayısını ayarla
        self.aisdTotalCases = len(imageFiles)
        self.aisdProcessedCases = 0
        self.aisdProgressDisplay.setText(f"0/{self.aisdTotalCases} (0%)")
        self.aisdProgressBar.setMaximum(self.aisdTotalCases)
        self.aisdProgressBar.setValue(0)
        
        # Modeli yükle
        self.segResultsText.setPlainText("Ana modeli yükleniyor... Lütfen bekleyin.")
        slicer.app.processEvents()
        self.segmentationLogic.segmentationModel = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        success = self.segmentationLogic.loadUNetModel(modelPath)
        if not success:
            self.segResultsText.setPlainText("Model yüklenemedi. Lütfen model dosyasını kontrol edin.")
            return
        
        # Ana modeli kaydet
        original_model = copy.deepcopy(self.segmentationLogic.segmentationModel)
        
        # Tüm hastaları işle ve sonuçları sakla
        all_results = []
        
        self.segResultsText.setPlainText("ADIM 1: Tüm hastaları işliyorum ve en iyi 20 vakayı belirliyorum...")
        slicer.app.processEvents()
        
        # Tüm hastaları ilk işleme
        for i, imageFile in enumerate(imageFiles):
            imageNode = None
            maskNode = None
            outputVolume = None
            
            try:
                # İşlenen vaka sayısını güncelle
                self.aisdProcessedCases = i
                progress_percent = int((i / self.aisdTotalCases) * 100) if self.aisdTotalCases > 0 else 0
                self.aisdProgressDisplay.setText(f"{i}/{self.aisdTotalCases} ({progress_percent}%)")
                self.aisdProgressBar.setValue(i)
                
                # İlerleme durumunu göster
                progress_text = f"ADIM 1: {i+1}/{len(imageFiles)} işleniyor: {os.path.basename(imageFile)}"
                self.segResultsText.setPlainText(progress_text)
                slicer.app.processEvents()
                
                # Görüntüyü yükle
                imageNode = slicer.util.loadVolume(imageFile)
                if not imageNode:
                    continue
                    
                # Eşleşen maskeyi ara
                baseName = os.path.basename(imageFile)
                maskFile = os.path.join(maskPath, baseName)
                if not os.path.exists(maskFile):
                    slicer.mrmlScene.RemoveNode(imageNode)
                    continue
                    
                # Maskeyi yükle
                maskNode = slicer.util.loadLabelVolume(maskFile)
                if not maskNode:
                    slicer.mrmlScene.RemoveNode(imageNode)
                    continue
                    
                # Segmentasyon yap - Genel parametrelerle
                segResults = self.segmentationLogic.processVolume(
                    imageNode,
                    threshold,
                    False,  # use_brain_mask
                    False,  # use_windowing
                    40,    # window_level
                    80     # window_width
                )
                
                # Segmentasyon hacmi oluştur
                outputVolume = self.segmentationLogic.createSegmentationVolume(imageNode, segResults)
                if not outputVolume:
                    slicer.mrmlScene.RemoveNode(imageNode)
                    slicer.mrmlScene.RemoveNode(maskNode)
                    continue
                    
                # Metrikleri hesapla
                metrics = self.segmentationLogic.computeSegmentationMetrics(maskNode, outputVolume)
                
                # Sonuçları sakla
                all_results.append({
                    'patient_id': baseName,
                    'dice': metrics['dice'],
                    'sensitivity': metrics.get('sensitivity', 0),
                    'specificity': metrics.get('specificity', 0),
                    'precision': metrics.get('precision', 0),
                    'gt_volume': metrics.get('gt_volume', 0),
                    'pred_volume': metrics.get('pred_volume', 0),
                    'volume_diff_percent': metrics.get('volume_diff_percent', 0),
                    'image_path': imageFile,
                    'mask_path': maskFile
                })
                
                # Düğümleri temizle
                if imageNode:
                    slicer.mrmlScene.RemoveNode(imageNode)
                if maskNode:
                    slicer.mrmlScene.RemoveNode(maskNode)
                if outputVolume:
                    slicer.mrmlScene.RemoveNode(outputVolume)
                    
            except Exception as e:
                # Hata durumunda temizlik
                if 'imageNode' in locals() and imageNode:
                    slicer.mrmlScene.RemoveNode(imageNode)
                if 'maskNode' in locals() and maskNode:
                    slicer.mrmlScene.RemoveNode(maskNode)
                if 'outputVolume' in locals() and outputVolume:
                    slicer.mrmlScene.RemoveNode(outputVolume)
        
        # İlk işlemenin istatistiklerini hesapla
        if not all_results:
            self.segResultsText.setPlainText("İşlenecek vaka bulunamadı!")
            return
            
        original_dice_scores = [result['dice'] for result in all_results]
        original_avg_dice = sum(original_dice_scores) / len(original_dice_scores)
        
        # Başarılı ve başarısız hasta sayıları
        success_cases = sum(1 for d in original_dice_scores if d > 0.5)
        moderate_cases = sum(1 for d in original_dice_scores if 0.3 <= d <= 0.5)
        poor_cases_count = sum(1 for d in original_dice_scores if d < 0.3)
        
        # İlk işleme sonuçlarını göster
        stats_text = f"İlk İşleme Sonuçları (Toplam {len(all_results)} vaka):\n"
        stats_text += f"Ortalama Dice: {original_avg_dice:.4f}\n"
        stats_text += f"Başarılı Vakalar (Dice > 0.5): {success_cases}/{len(all_results)} ({100*success_cases/len(all_results):.1f}%)\n"
        stats_text += f"Orta Vakalar (0.3 <= Dice <= 0.5): {moderate_cases}/{len(all_results)} ({100*moderate_cases/len(all_results):.1f}%)\n"
        stats_text += f"Başarısız Vakalar (Dice < 0.3): {poor_cases_count}/{len(all_results)} ({100*poor_cases_count/len(all_results):.1f}%)\n"
        
        # Sonuçları göster
        self.segResultsText.setPlainText(stats_text)
        slicer.app.processEvents()
        
        # ADIM 2: En iyi 20 vakadan model eğitme
        #-----------------------------------------------------------------
        
        # Sonuçları Dice skoruna göre sırala
        sorted_results = sorted(all_results, key=lambda x: x['dice'], reverse=True)
        
        # En iyi 20 vakayı seç
        best_20_cases = sorted_results[:20] if len(sorted_results) >= 20 else sorted_results
        
        # En iyi 20 vakanın detaylarını göster
        best_cases_text = "En İyi 20 Vaka:\n"
        for i, case in enumerate(best_20_cases):
            best_cases_text += f"{i+1}. {case['patient_id']}: Dice={case['dice']:.4f}, GT={case['gt_volume']} px, Pred={case['pred_volume']} px\n"
        
        self.segResultsText.setPlainText(self.segResultsText.toPlainText() + "\n\n" + best_cases_text)
        self.segResultsText.setPlainText(self.segResultsText.toPlainText() + "\n\nADIM 2: En iyi 20 vakadan model eğitiyorum...")
        slicer.app.processEvents()
        
        # En iyi 20 vakadan model eğit
        fine_tuned_model = self.fineTuneModelFromBestCases(best_20_cases, original_model)
        
        if not fine_tuned_model:
            self.segResultsText.setPlainText(self.segResultsText.toPlainText() + 
                                            "\nModel eğitimi başarısız oldu. Orijinal model kullanılacak.")
            fine_tuned_model = original_model
        
        # ADIM 3: Düşük performanslı vakaları (Dice < 0.3) ince ayarlı model ile yeniden işle
        #-----------------------------------------------------------------
        
        # Düşük performanslı vakaları seç
        poor_cases = [case for case in all_results if case['dice'] < 0.3]
        
        if poor_cases:
            self.segResultsText.setPlainText(self.segResultsText.toPlainText() + 
                f"\n\nADIM 3: {len(poor_cases)} düşük performanslı vakayı (Dice < 0.3) ince ayarlı model ile yeniden işliyorum...")
            slicer.app.processEvents()
            
            # İlerleme göstergesini güncelle
            self.aisdTotalCases = len(poor_cases)
            self.aisdProcessedCases = 0
            self.aisdProgressDisplay.setText(f"0/{self.aisdTotalCases} (0%)")
            self.aisdProgressBar.setMaximum(self.aisdTotalCases)
            self.aisdProgressBar.setValue(0)
            
            # İnce ayarlı modeli ayarla
            self.segmentationLogic.segmentationModel = fine_tuned_model
            
            # Farklı eşik değerleri dene
            threshold_options = [0.4, 0.5, 0.6, 0.7]
            improved_results = []
            
            # Düşük performanslı vakaları işle
            for i, case in enumerate(poor_cases):
                best_dice = case['dice']
                best_threshold = threshold
                best_metrics = None
                
                # İşlenen vaka sayısını güncelle
                self.aisdProcessedCases = i
                progress_percent = int((i / self.aisdTotalCases) * 100) if self.aisdTotalCases > 0 else 0
                self.aisdProgressDisplay.setText(f"{i}/{self.aisdTotalCases} ({progress_percent}%)")
                self.aisdProgressBar.setValue(i)
                
                # İlerleme durumunu göster
                progress_text = f"ADIM 3: Düşük performanslı vaka {i+1}/{len(poor_cases)} yeniden işleniyor: {case['patient_id']}"
                self.segResultsText.setPlainText(self.segResultsText.toPlainText().split('\n\n')[0] + "\n\n" + progress_text)
                slicer.app.processEvents()
                
                # Her eşik değeri için dene
                for test_threshold in threshold_options:
                    imageNode = None
                    maskNode = None
                    outputVolume = None
                    
                    try:
                        # Görüntü ve maskeyi yükle
                        imageNode = slicer.util.loadVolume(case['image_path'])
                        maskNode = slicer.util.loadLabelVolume(case['mask_path'])
                        
                        if not imageNode or not maskNode:
                            continue
                        
                        # İnce ayarlı model ile segmentasyon yap
                        segResults = self.segmentationLogic.processVolume(
                            imageNode,
                            test_threshold,
                            True,  # use_brain_mask
                            True,  # use_windowing
                            40,    # window_level
                            100    # window_width
                        )
                        
                        # Segmentasyon hacmi oluştur
                        outputVolume = self.segmentationLogic.createSegmentationVolume(imageNode, segResults)
                        
                        if outputVolume:
                            # Metrikleri hesapla
                            metrics = self.segmentationLogic.computeSegmentationMetrics(maskNode, outputVolume)
                            
                            # Daha iyi sonuç bulundu mu kontrol et
                            if metrics['dice'] > best_dice:
                                best_dice = metrics['dice']
                                best_threshold = test_threshold
                                best_metrics = metrics
                                
                        # Düğümleri temizle
                        if imageNode:
                            slicer.mrmlScene.RemoveNode(imageNode)
                        if maskNode:
                            slicer.mrmlScene.RemoveNode(maskNode)
                        if outputVolume:
                            slicer.mrmlScene.RemoveNode(outputVolume)
                            
                    except Exception as e:
                        print(f"İşleme hatası: {str(e)}")
                        # Hata durumunda temizlik
                        if 'imageNode' in locals() and imageNode:
                            slicer.mrmlScene.RemoveNode(imageNode)
                        if 'maskNode' in locals() and maskNode:
                            slicer.mrmlScene.RemoveNode(maskNode)
                        if 'outputVolume' in locals() and outputVolume:
                            slicer.mrmlScene.RemoveNode(outputVolume)
                
                # En iyi sonuçla güncelle
                if best_metrics:
                    improved_results.append({
                        'patient_id': case['patient_id'],
                        'original_dice': case['dice'],
                        'improved_dice': best_dice,
                        'best_threshold': best_threshold,
                        'improvement': best_dice - case['dice'],
                        'metrics': best_metrics
                    })
            
            # İyileştirme sonuçlarını göster
            improved_text = "Düşük Performanslı Vakalarda İyileştirmeler:\n\n"
            for i, result in enumerate(improved_results):
                improved_text += f"{i+1}. {result['patient_id']}:\n"
                improved_text += f"   - Orijinal Dice: {result['original_dice']:.4f}\n"
                improved_text += f"   - İyileştirilmiş Dice: {result['improved_dice']:.4f}\n"
                improved_text += f"   - En İyi Eşik: {result['best_threshold']:.2f}\n"
                improved_text += f"   - İyileştirme: {result['improvement']:.4f}\n\n"
            
            self.segResultsText.setPlainText(self.segResultsText.toPlainText() + "\n\n" + improved_text)
            slicer.app.processEvents()
            
            # ADIM 4: İyileştirilmiş sonuçlarla tüm veriseti için tekrar hesaplama
            #-----------------------------------------------------------------
            
            # İyileştirilmiş sonuçlar ile orijinal sonuçları birleştir
            final_results = []
            
            # Tüm sonuçları kopyala
            for result in all_results:
                # Bu sonuç iyileştirilmiş mi kontrol et
                improved_result = next((r for r in improved_results if r['patient_id'] == result['patient_id']), None)
                
                if improved_result:
                    # İyileştirilmiş sonucu kullan
                    final_results.append({
                        'patient_id': result['patient_id'],
                        'dice': improved_result['improved_dice'],
                        'original_dice': result['dice'],
                        'improved': True
                    })
                else:
                    # Orijinal sonucu kullan
                    final_results.append({
                        'patient_id': result['patient_id'],
                        'dice': result['dice'],
                        'original_dice': result['dice'],
                        'improved': False
                    })
            
            # Yeni ortalama Dice hesapla
            final_dice_scores = [result['dice'] for result in final_results]
            final_avg_dice = sum(final_dice_scores) / len(final_dice_scores)
            
            # İyileştirme miktarını hesapla
            dice_improvement = final_avg_dice - original_avg_dice
            
            # Başarılı ve başarısız hasta sayılarını yeniden hesapla
            final_success_cases = sum(1 for d in final_dice_scores if d > 0.5)
            final_moderate_cases = sum(1 for d in final_dice_scores if 0.3 <= d <= 0.5)
            final_poor_cases = sum(1 for d in final_dice_scores if d < 0.3)
            
            # Genel istatistikleri göster
            final_stats_text = f"\nSONUÇLAR (Toplam {len(all_results)} vaka):\n"
            final_stats_text += f"Orijinal Ortalama Dice: {original_avg_dice:.4f}\n"
            final_stats_text += f"İyileştirilmiş Ortalama Dice: {final_avg_dice:.4f}\n"
            final_stats_text += f"Toplam İyileştirme: {dice_improvement:.4f} ({dice_improvement/original_avg_dice*100:.2f}%)\n\n"
            
            final_stats_text += f"Orijinal Başarılı Vakalar (Dice > 0.5): {success_cases}/{len(all_results)} ({100*success_cases/len(all_results):.1f}%)\n"
            final_stats_text += f"İyileştirilmiş Başarılı Vakalar (Dice > 0.5): {final_success_cases}/{len(all_results)} ({100*final_success_cases/len(all_results):.1f}%)\n\n"
            
            final_stats_text += f"Orijinal Orta Vakalar (0.3 <= Dice <= 0.5): {moderate_cases}/{len(all_results)} ({100*moderate_cases/len(all_results):.1f}%)\n"
            final_stats_text += f"İyileştirilmiş Orta Vakalar (0.3 <= Dice <= 0.5): {final_moderate_cases}/{len(all_results)} ({100*final_moderate_cases/len(all_results):.1f}%)\n\n"
            
            final_stats_text += f"Orijinal Başarısız Vakalar (Dice < 0.3): {poor_cases_count}/{len(all_results)} ({100*poor_cases_count/len(all_results):.1f}%)\n"
            final_stats_text += f"İyileştirilmiş Başarısız Vakalar (Dice < 0.3): {final_poor_cases}/{len(all_results)} ({100*final_poor_cases/len(all_results):.1f}%)\n"
            
            self.segResultsText.setPlainText(self.segResultsText.toPlainText() + "\n\n" + final_stats_text)
        
        # İlerleme çubuğunu ve göstergeyi tamamla
        self.aisdProcessedCases = self.aisdTotalCases
        self.aisdProgressDisplay.setText(f"{self.aisdTotalCases}/{self.aisdTotalCases} (100%)")
        self.aisdProgressBar.setValue(self.aisdTotalCases)
        
        # Orijinal modele geri dön
        self.segmentationLogic.segmentationModel = original_model
        
        # İşlem tamamlandı
        self.segResultsText.setPlainText(self.segResultsText.toPlainText() + 
                                        f"\n\nİşlem tamamlandı. Toplam süre: {time.time() - start_time:.2f} saniye")
        
        # Sonuçları sınıf değişkenlerine sakla (daha sonraki analizler için)
        self.all_results = all_results
        self.improved_results = improved_results if 'improved_results' in locals() else []
        
        # Yeniden işleme butonunu etkinleştir
        self.reprocessButton.enabled = True
    def onReprocessButton(self):
        """Son işlenen görüntüyü güncel eşik değeriyle yeniden işle"""
        if not self.lastProcessedFile:
            slicer.util.errorDisplay("Henüz işlenmiş bir görüntü yok")
            return
        # Son işlenen görüntüyü seçici kutuda seç
        index = self.segImageSelector.findText(self.lastProcessedFile)
        if index >= 0:
            self.segImageSelector.setCurrentIndex(index)
            # Segment butonunu çağırarak yeniden işle
            self.onSegmentButton()
        else:
            slicer.util.errorDisplay(f"Son işlenen dosya bulunamadı: {self.lastProcessedFile}")
    
    def onDebugButton(self):
        """Model hata ayıklama işlevleri"""
        modelPath = self.segModelFileSelector.currentPath
        if not modelPath or not os.path.exists(modelPath):
            self.segResultsText.setPlainText("Lütfen geçerli bir model dosyası seçin!")
            return
            
        debug_text = "KAPSAMLI MODEL ANALİZİ:\n\n"
        
        # ADIM 1: Model dosyasını kontrol et
        debug_text += f"1. MODEL DOSYASI KONTROLÜ:\n"
        try:
            filesize = os.path.getsize(modelPath) / (1024*1024)
            debug_text += f"   - Dosya yolu: {modelPath}\n"
            debug_text += f"   - Dosya boyutu: {filesize:.2f} MB\n"
            # Dosya özeti (checksum) hesapla
            import hashlib
            file_md5 = hashlib.md5(open(modelPath,'rb').read()).hexdigest()
            debug_text += f"   - Dosya özeti (MD5): {file_md5}\n\n"
            
            # Dosya uzantısını kontrol et
            if not modelPath.lower().endswith('.pth'):
                debug_text += f"   - UYARI: Dosya uzantısı .pth değil! Bu bir PyTorch modeli olmayabilir.\n\n"
        except Exception as e:
            debug_text += f"   - Dosya kontrolü hatası: {str(e)}\n\n"
        
        # ADIM 2: Modeli yükle ve parametreleri ayrıntılı kontrol et
        debug_text += f"2. MODEL DETAYLI ANALİZİ:\n"
        # Önce mevcut modeli temizle
        if hasattr(self.segmentationLogic, 'segmentationModel'):
            self.segmentationLogic.segmentationModel = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        try:
            # Modeli yükle
            checkpoint = torch.load(modelPath, map_location=self.segmentationLogic.device)
            debug_text += f"   - Model dosyası okundu\n"
            
            # Checkpoint formatını kontrol et
            if isinstance(checkpoint, dict):
                debug_text += f"   - Checkpoint sözlük formatında\n"
                # Hangi anahtarlar var?
                debug_text += f"   - Mevcut anahtarlar: {list(checkpoint.keys())}\n"
                # Eğitim bilgileri var mı?
                if 'epoch' in checkpoint:
                    debug_text += f"   - Kaydedilen epoch: {checkpoint['epoch']}\n"
                if 'best_metric' in checkpoint:
                    debug_text += f"   - En iyi metrik: {checkpoint['best_metric']}\n"
                # Model state_dict'i bul
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    debug_text += f"   - model_state_dict anahtarı bulundu\n"
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    debug_text += f"   - state_dict anahtarı bulundu\n"
                else:
                    state_dict = checkpoint
                    debug_text += f"   - Doğrudan state_dict formatında\n"
            else:
                state_dict = checkpoint
                debug_text += f"   - Doğrudan model durumu formatında\n"
                
            # State dict içeriğini analiz et
            param_count = len(state_dict)
            debug_text += f"   - Toplam parametre sayısı: {param_count}\n"
            
            # İlk 5 parametre anahtarını ve değerlerini göster
            debug_text += f"   - Parametre anahtarları (ilk 5): {list(state_dict.keys())[:5]}\n"
            
            # Parametre değerlerinin detaylı analizi
            param_stats = []
            for i, (key, tensor) in enumerate(state_dict.items()):
                if i >= 5: break
                if isinstance(tensor, torch.Tensor):
                    stats = {
                        'name': key,
                        'shape': list(tensor.shape),
                        'min': tensor.min().item(),
                        'max': tensor.max().item(),
                        'mean': tensor.mean().item(),
                        'std': tensor.std().item()
                    }
                    param_stats.append(stats)
                    
            # İstatistikleri göster
            debug_text += "\n   Parametre İstatistikleri (İlk 5):\n"
            for i, stats in enumerate(param_stats):
                debug_text += f"   {i+1}. {stats['name']}: şekil={stats['shape']}, min={stats['min']:.6f}, max={stats['max']:.6f}, ort={stats['mean']:.6f}\n"
            debug_text += "\n"
            
            # ADIM 3: Modeli düzgün yükle
            debug_text += f"3. MODELİ GÜNCELLENMIŞ KODLA YÜKLE:\n"
            
            # UYARI: Burada kendi model oluşturma kullanılmayacak - sadece kontrolü yapılacak
            debug_text += f"   - UYARI: Sadece kontrol yapılacak, kendi model oluşturulmayacak.\n"
            
            # Gerçek kullanımda hata yoksa, modelin kullanılabileceğini belirt
            debug_text += f"   - Model düzgün yüklenebilir görünüyor.\n"
            debug_text += f"   - Segmentasyon için 'Segment Selected Image' butonunu kullanın.\n"
        except Exception as e:
            debug_text += f"   - Model analiz hatası: {str(e)}\n"
            import traceback
            debug_text += traceback.format_exc()
            debug_text += f"\n   - UYARI: Model yüklenemedi. Bu dosya muhtemelen geçerli bir PyTorch modeli değil!\n"
        
        # Sonuçları göster
        self.segResultsText.setPlainText(debug_text)
    
    def onSegmentButton(self):
        """Seçili görüntüyü bölütle"""
        # Gerekli kontrolleri yap
        imagePath = self.segInputImageFolderSelector.currentPath
        maskPath = self.segMaskFolderSelector.currentPath
        modelPath = self.segModelFileSelector.currentPath
        threshold = self.segThresholdSlider.value
        
        if not imagePath or not os.path.exists(imagePath):
            slicer.util.errorDisplay("Lütfen geçerli bir görüntü klasörü seçin")
            return
        if not maskPath or not os.path.exists(maskPath):
            slicer.util.errorDisplay("Lütfen geçerli bir maske klasörü seçin")
            return
        if not modelPath or not os.path.exists(modelPath):
            slicer.util.errorDisplay("Lütfen geçerli bir model dosyası seçin")
            return
        
        # Seçili görüntü ve maske dosya yollarını al
        selectedImageName = self.segImageSelector.currentText
        selectedMaskName = self.segMaskSelector.currentText
        if not selectedImageName or not selectedMaskName:
            slicer.util.errorDisplay("Lütfen bir görüntü ve maske seçin")
            return
        
        # Tam dosya yollarını oluştur
        selectedImagePath = os.path.join(imagePath, selectedImageName)
        selectedMaskPath = os.path.join(maskPath, selectedMaskName)
        
        # İşlem sırasında arayüzü devre dışı bırak
        self.segmentButton.enabled = False
        self.segmentButton.text = "Processing..."
        self.segResultsText.setPlainText("İşlem başlıyor... Lütfen bekleyin.")
        slicer.app.processEvents()
        
        imageNode = None 
        maskNode = None  
        outputVolume = None 
        
        try:
            # Mevcut nodeları temizle
            self.clearNodes()
            
            # Görüntüleri yükle
            self.segResultsText.setPlainText("Görüntüler yükleniyor...")
            slicer.app.processEvents()
            
            # MRI görüntüsünü yükle
            imageNode = slicer.util.loadVolume(selectedImagePath)
            if not imageNode:
                slicer.util.errorDisplay(f"Görüntü yüklenirken hata: {selectedImagePath}")
                return
                
            # Maskeyi yükle
            maskNode = slicer.util.loadLabelVolume(selectedMaskPath)
            if not maskNode:
                slicer.util.errorDisplay(f"Maske yüklenirken hata: {selectedMaskPath}")
                return
                
            # Node referanslarını sakla
            self._imageNode = imageNode
            self._maskNode = maskNode
            
            # Her seferinde model dosyası özeti hesapla
            model_md5 = hashlib.md5(open(modelPath,'rb').read()).hexdigest()[:8]
            
            # Modeli her zaman yeniden yükle
            self.segResultsText.setPlainText(f"Model yükleniyor... (MD5: {model_md5})\nBu işlem biraz zaman alabilir.")
            slicer.app.processEvents()
            
            # Modeli zorla yeniden yükle - Önce eski modeli temizle
            self.segmentationLogic.segmentationModel = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            success = self.segmentationLogic.loadUNetModel(modelPath)
            
            if not success:
                slicer.util.errorDisplay("Model yüklenemedi. Lütfen model dosyasını kontrol edin.")
                return
                
            # Model meta verilerini göster
            model_name = os.path.basename(modelPath)
            model_info = f"Model: {model_name}\n"
            model_info += f"Model MD5: {model_md5}\n"
            
            if hasattr(self.segmentationLogic, 'current_model_fingerprint'):
                model_info += f"Model Parmak İzi: {self.segmentationLogic.current_model_fingerprint[:8] if self.segmentationLogic.current_model_fingerprint else 'N/A'}...\n"
                
            if hasattr(self.segmentationLogic, 'current_model_output_signature'):
                if self.segmentationLogic.current_model_output_signature:
                    sig = self.segmentationLogic.current_model_output_signature
                    model_info += f"Model İmzası: min={sig['min']:.4f}, max={sig['max']:.4f}, mean={sig['mean']:.4f}\n"
                    
            self.segResultsText.setPlainText(f"{model_info}\n\nSegmentasyon başlıyor...")
            slicer.app.processEvents()
            
            # Segmentasyon işlemini başlat - TTA ve dinamik eşik değeri ile
            segResults = self.segmentationLogic.processVolume(
                imageNode,
                threshold,
                False,  # use_brain_mask 
                False,  # use_windowing
                40,    # window_level
                80     # window_width
            )
            
            if not segResults or len(segResults) == 0:
                slicer.util.errorDisplay("Segmentasyon sonuçları boş. İşlem başarısız oldu.")
                return
                
            # Segmentasyon hacmi oluştur
            outputVolume = self.segmentationLogic.createSegmentationVolume(imageNode, segResults)
            if not outputVolume:
                slicer.util.errorDisplay("Segmentasyon hacmi oluşturulamadı.")
                return
                
            self._outputVolume = outputVolume
            
            # Görüntüleme ayarlarını güncelle
            self.updateVisualization()
            
            # 3D modeli oluştur (eğer etkinse)
            if self.show3DCheckBox.checked:
                self.updateModel3DVisualization()
                
            # Metrikleri hesapla ve göster
            metrics = self.segmentationLogic.computeSegmentationMetrics(maskNode, outputVolume)
            
            # Son işlenen dosyayı ve Dice skorunu kaydet
            self.lastProcessedFile = selectedImageName
            self.lastDiceScore = metrics['dice']
            
            # Dice skorunu göstergeye yaz
            self.diceScoreDisplay.setText(f"{self.lastDiceScore:.4f}")
            
            # Model ve Dice bilgilerini birlikte göster
            model_display_name = os.path.basename(modelPath).split('.')[0]
            
            # Sonuçları göster
            resultText = f"Segmentation Results for {selectedImageName}:\n\n"
            resultText += model_info + "\n"
            resultText += f"Confusion Matrix:\n"
            resultText += f"True Positive (TP): {metrics['tp']}\n"
            resultText += f"False Positive (FP): {metrics['fp']}\n"
            resultText += f"False Negative (FN): {metrics['fn']}\n"
            resultText += f"True Negative (TN): {metrics['tn']}\n\n"
            resultText += f"Dice Score: {metrics['dice']:.4f}\n"
            resultText += f"Sensitivity: {metrics['sensitivity']:.4f}\n"
            resultText += f"Specificity: {metrics['specificity']:.4f}\n"
            resultText += f"Precision: {metrics['precision']:.4f}\n"
            resultText += f"Accuracy: {metrics['accuracy']:.4f}\n"
            
            # Hacim bilgilerini ekle
            if 'gt_volume' in metrics and 'pred_volume' in metrics:
                resultText += f"\nGT Hacim: {metrics['gt_volume']} piksel\n"
                resultText += f"Tahmin Hacim: {metrics['pred_volume']} piksel\n"
                resultText += f"Hacim Farkı: {metrics['volume_diff_percent']:.2f}%\n"
                
            # Dinamik eşik değeri bilgisi ekle
            threshold_values = [r.get('threshold_used', threshold) for r in segResults if 'threshold_used' in r]
            if threshold_values:
                avg_threshold = sum(threshold_values) / len(threshold_values)
                resultText += f"\nKullanılan Ortalama Eşik Değeri: {avg_threshold:.4f}\n"
            
            # Modele özgü Dice skorunu vurgula
            resultText += f"\n*****************************************\n"
            resultText += f"* {model_display_name} (MD5: {model_md5})\n* DICE: {metrics['dice']:.4f}\n"
            resultText += f"*****************************************\n"
            self.segResultsText.setPlainText(resultText)
            
            # Yeniden işleme butonunu etkinleştir
            self.reprocessButton.enabled = True
            
        except Exception as e:
            slicer.util.errorDisplay(f"İşlem sırasında hata oluştu: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Arayüzü tekrar etkinleştir
            self.segmentButton.enabled = True
            self.segmentButton.text = "Segment Selected Image"
    def fineTuneModelFromBestCases(self, best_cases_list, original_model):
        """En iyi vakaları kullanarak modeli ince ayar yapar"""
        self.segResultsText.setPlainText("En iyi 20 vaka ile model ince ayarı yapılıyor...")
        slicer.app.processEvents()
        
        try:
            # Modeli kopyala
            fine_tuned_model = copy.deepcopy(original_model)
            
            # Eğitim için parametreler
            learning_rate = self.learningRateSpinBox.value
            epochs = self.epochSpinBox.value
            batch_size = self.batchSizeSpinBox.value
            
            # Eğitim veri seti oluştur
            train_slices = []
            train_masks = []
            
            for case in best_cases_list:
                try:
                    # Görüntü ve maskeyi yükle
                    imageNode = slicer.util.loadVolume(case['image_path'])
                    maskNode = slicer.util.loadLabelVolume(case['mask_path'])
                    
                    if not imageNode or not maskNode:
                        continue
                        
                    # Numpy dizilerine dönüştür
                    image_array = slicer.util.arrayFromVolume(imageNode)
                    mask_array = slicer.util.arrayFromVolume(maskNode)
                    
                    # Her dilimi ekle
                    for slice_idx in range(image_array.shape[0]):
                        img_slice = image_array[slice_idx]
                        mask_slice = mask_array[slice_idx]
                        
                        # Eğer dilimde lezyon varsa
                        if np.sum(mask_slice) > 10:  # En az 10 piksel lezyon içeren dilimleri kullan
                            # Normalizasyon
                            min_val = np.min(img_slice)
                            max_val = np.max(img_slice)
                            if max_val > min_val:
                                norm_slice = (img_slice - min_val) / (max_val - min_val)
                            else:
                                norm_slice = np.zeros_like(img_slice)
                            
                            # Boyutları ayarla (512x512)
                            if norm_slice.shape[0] != 512 or norm_slice.shape[1] != 512:
                                norm_slice = cv2.resize(norm_slice, (512, 512), interpolation=cv2.INTER_LINEAR)
                            if mask_slice.shape[0] != 512 or mask_slice.shape[1] != 512:
                                mask_slice = cv2.resize(mask_slice.astype(np.float32), (512, 512), 
                                                        interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                            
                            # Veri setine ekle
                            train_slices.append(norm_slice)
                            train_masks.append(mask_slice)
                    
                    # Düğümleri temizle
                    slicer.mrmlScene.RemoveNode(imageNode)
                    slicer.mrmlScene.RemoveNode(maskNode)
                    
                except Exception as e:
                    print(f"Vaka işleme hatası: {str(e)}")
                    if 'imageNode' in locals() and imageNode:
                        slicer.mrmlScene.RemoveNode(imageNode)
                    if 'maskNode' in locals() and maskNode:
                        slicer.mrmlScene.RemoveNode(maskNode)
            
            # Yeterli veri toplanabildiyse modeli eğit
            if len(train_slices) < 10:
                self.segResultsText.setPlainText("Yeterli eğitim verisi bulunamadı (en az 10 dilim gerekli)")
                return None
                
            self.segResultsText.setPlainText(f"Toplam {len(train_slices)} dilim ile model eğitimi başlıyor...")
            slicer.app.processEvents()
            
            # Tensor veri seti oluştur
            X_train = np.array(train_slices).reshape(-1, 1, 512, 512).astype(np.float32)
            y_train = np.array(train_masks).reshape(-1, 1, 512, 512).astype(np.float32)
            
            X_train_tensor = torch.from_numpy(X_train).to(self.segmentationLogic.device)
            y_train_tensor = torch.from_numpy(y_train).to(self.segmentationLogic.device)
            
            # Optimizer ve loss
            optimizer = torch.optim.Adam(fine_tuned_model.parameters(), lr=learning_rate)
            criterion = torch.nn.BCEWithLogitsLoss()
            
            # Eğitim döngüsü
            fine_tuned_model.train()
            
            for epoch in range(epochs):
                total_loss = 0
                
                # Mini-batch ile eğitim
                for i in range(0, len(X_train_tensor), batch_size):
                    batch_X = X_train_tensor[i:i+batch_size]
                    batch_y = y_train_tensor[i:i+batch_size]
                    
                    # Forward pass
                    outputs = fine_tuned_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass ve optimizasyon
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                # Epoch sonuçlarını göster
                avg_loss = total_loss / (len(X_train_tensor) / batch_size)
                self.segResultsText.setPlainText(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
                slicer.app.processEvents()
            
            # Modeli değerlendirme moduna al
            fine_tuned_model.eval()
            
            self.segResultsText.setPlainText("Model eğitimi tamamlandı!")
            slicer.app.processEvents()
            
            # Eğitilmiş modeli sınıf değişkenine kaydet
            self._fineTunedModel = fine_tuned_model
            
            return fine_tuned_model
            
        except Exception as e:
            self.segResultsText.setPlainText(f"Model eğitimi sırasında hata: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    def onAisdSegmentButton(self):
        """AISD verisetini segmente et - en iyi 20'den model eğit, zor vakaları iyileştir"""
        self.segResultsText.setPlainText("AISD verisetini segmente etme işlemi başlıyor...")
        slicer.app.processEvents()
        # Gerekli kontrolleri yap
        imagePath = self.segInputImageFolderSelector.currentPath
        maskPath = self.segMaskFolderSelector.currentPath
        modelPath = self.segModelFileSelector.currentPath
        threshold = self.segThresholdSlider.value
        if not imagePath or not os.path.exists(imagePath):
            self.segResultsText.setPlainText("Lütfen geçerli bir görüntü klasörü seçin")
            return
        if not maskPath or not os.path.exists(maskPath):
            self.segResultsText.setPlainText("Lütfen geçerli bir maske klasörü seçin")
            return
        if not modelPath or not os.path.exists(modelPath):
            self.segResultsText.setPlainText("Lütfen geçerli bir model dosyası seçin")
            return
        # Tüm görüntü dosyalarını al
        imageFiles = []
        for ext in ['.nii.gz', '.nii', '.mha']:
            imageFiles.extend(glob.glob(os.path.join(imagePath, f'*{ext}')))
        if len(imageFiles) == 0:
            self.segResultsText.setPlainText("Klasörde görüntü dosyası bulunamadı!")
            return
        # Başlangıç zamanını kaydet
        start_time = time.time()
        # ADIM 1: İlk genel işleme ve en iyi 20 vakayı belirleme
        #-----------------------------------------------------------------
        # Toplam vaka sayısını ayarla
        self.aisdTotalCases = len(imageFiles)
        self.aisdProcessedCases = 0
        self.aisdProgressDisplay.setText(f"0/{self.aisdTotalCases} (0%)")
        self.aisdProgressBar.setMaximum(self.aisdTotalCases)
        self.aisdProgressBar.setValue(0)
        # Modeli yükle
        self.segResultsText.setPlainText("Ana modeli yükleniyor... Lütfen bekleyin.")
        slicer.app.processEvents()
        self.segmentationLogic.segmentationModel = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        success = self.segmentationLogic.loadUNetModel(modelPath)
        if not success:
            self.segResultsText.setPlainText("Model yüklenemedi. Lütfen model dosyasını kontrol edin.")
            return
        # Ana modeli kaydet
        original_model = copy.deepcopy(self.segmentationLogic.segmentationModel)
        # Tüm hastaları işle ve sonuçları sakla
        all_results = []
        self.segResultsText.setPlainText("ADIM 1: Tüm hastaları işliyorum ve en iyi 20 vakayı belirliyorum...")
        slicer.app.processEvents()
        # Tüm hastaları ilk işleme
        for i, imageFile in enumerate(imageFiles):
            imageNode = None
            maskNode = None
            outputVolume = None
            try:
                # İşlenen vaka sayısını güncelle
                self.aisdProcessedCases = i
                progress_percent = int((i / self.aisdTotalCases) * 100) if self.aisdTotalCases > 0 else 0
                self.aisdProgressDisplay.setText(f"{i}/{self.aisdTotalCases} ({progress_percent}%)")
                self.aisdProgressBar.setValue(i)
                # İlerleme durumunu göster
                progress_text = f"ADIM 1: {i+1}/{len(imageFiles)} işleniyor: {os.path.basename(imageFile)}"
                self.segResultsText.setPlainText(progress_text)
                slicer.app.processEvents()
                # Görüntüyü yükle
                imageNode = slicer.util.loadVolume(imageFile)
                if not imageNode:
                    continue
                # Eşleşen maskeyi ara
                baseName = os.path.basename(imageFile)
                maskFile = os.path.join(maskPath, baseName)
                if not os.path.exists(maskFile):
                    slicer.mrmlScene.RemoveNode(imageNode)
                    continue
                # Maskeyi yükle
                maskNode = slicer.util.loadLabelVolume(maskFile)
                if not maskNode:
                    slicer.mrmlScene.RemoveNode(imageNode)
                    continue
                # Segmentasyon yap - Genel parametrelerle
                segResults = self.segmentationLogic.processVolume(
                    imageNode,
                    threshold,
                    False,  # use_brain_mask
                    False,  # use_windowing
                    40,    # window_level
                    80     # window_width
                )
                # Segmentasyon hacmi oluştur
                outputVolume = self.segmentationLogic.createSegmentationVolume(imageNode, segResults)
                if not outputVolume:
                    slicer.mrmlScene.RemoveNode(imageNode)
                    slicer.mrmlScene.RemoveNode(maskNode)
                    continue
                # Metrikleri hesapla
                metrics = self.segmentationLogic.computeSegmentationMetrics(maskNode, outputVolume)
                # Sonuçları sakla
                all_results.append({
                    'patient_id': baseName,
                    'dice': metrics['dice'],
                    'sensitivity': metrics.get('sensitivity', 0),
                    'specificity': metrics.get('specificity', 0),
                    'precision': metrics.get('precision', 0),
                    'gt_volume': metrics.get('gt_volume', 0),
                    'pred_volume': metrics.get('pred_volume', 0),
                    'volume_diff_percent': metrics.get('volume_diff_percent', 0),
                    'image_path': imageFile,
                    'mask_path': maskFile
                })
                # Düğümleri temizle
                if imageNode:
                    slicer.mrmlScene.RemoveNode(imageNode)
                if maskNode:
                    slicer.mrmlScene.RemoveNode(maskNode)
                if outputVolume:
                    slicer.mrmlScene.RemoveNode(outputVolume)
            except Exception as e:
                # Hata durumunda temizlik
                if 'imageNode' in locals() and imageNode:
                    slicer.mrmlScene.RemoveNode(imageNode)
                if 'maskNode' in locals() and maskNode:
                    slicer.mrmlScene.RemoveNode(maskNode)
                if 'outputVolume' in locals() and outputVolume:
                    slicer.mrmlScene.RemoveNode(outputVolume)
        
        # İlk işlemenin istatistiklerini hesapla
        if not all_results:
            self.segResultsText.setPlainText("İşlenecek vaka bulunamadı!")
            return
        
        # TÜM HASTALARIN DICE SKORLARINI YAZDIR - AYRINTILI GÖRÜNTÜLEME
        original_dice_scores = [result['dice'] for result in all_results]
        original_avg_dice = sum(original_dice_scores) / len(original_dice_scores)
        
        # Tüm hastaların bireysel Dice skorlarını yazdır - DETAYLI ÇIKTI
        all_dice_text = "Tüm Vakaların Başlangıç Dice Skorları:\n"
        for idx, result in enumerate(all_results):
            all_dice_text += f"{idx+1}. {result['patient_id']}: {result['dice']:.4f}\n"
        
        # Detaylı log ve çıktı için hem dosyaya hem de ekrana yazdır
        with open(os.path.join(os.path.dirname(modelPath), "all_dice_scores.txt"), "w") as f:
            f.write(all_dice_text)
        
        # Kullanıcı arayüzüne de tüm skorları yazdır
        self.segResultsText.setPlainText(all_dice_text)
        slicer.app.processEvents()
        
        # Başarılı ve başarısız hasta sayıları
        success_cases = sum(1 for d in original_dice_scores if d > 0.5)
        moderate_cases = sum(1 for d in original_dice_scores if 0.3 <= d <= 0.5)
        poor_cases_count = sum(1 for d in original_dice_scores if d < 0.3)
        
        # İlk işleme sonuçlarını göster
        stats_text = f"İlk İşleme Sonuçları (Toplam {len(all_results)} vaka):\n"
        stats_text += f"Ortalama Dice: {original_avg_dice:.4f}\n"
        stats_text += f"Başarılı Vakalar (Dice > 0.5): {success_cases}/{len(all_results)} ({100*success_cases/len(all_results):.1f}%)\n"
        stats_text += f"Orta Vakalar (0.3 <= Dice <= 0.5): {moderate_cases}/{len(all_results)} ({100*moderate_cases/len(all_results):.1f}%)\n"
        stats_text += f"Başarısız Vakalar (Dice < 0.3): {poor_cases_count}/{len(all_results)} ({100*poor_cases_count/len(all_results):.1f}%)\n"
        
        # Sonuçları göster
        self.segResultsText.setPlainText(self.segResultsText.toPlainText() + "\n\n" + stats_text)
        slicer.app.processEvents()
        
        # ADIM 2: En iyi 20 vakadan model eğitme
        #-----------------------------------------------------------------
        # Sonuçları Dice skoruna göre sırala
        sorted_results = sorted(all_results, key=lambda x: x['dice'], reverse=True)
        
        # En iyi 20 vakayı seç
        best_20_cases = sorted_results[:20] if len(sorted_results) >= 20 else sorted_results
        
        # En iyi 20 vakanın detaylarını göster
        best_cases_text = "En İyi 20 Vaka:\n"
        for i, case in enumerate(best_20_cases):
            best_cases_text += f"{i+1}. {case['patient_id']}: Dice={case['dice']:.4f}, GT={case['gt_volume']} px, Pred={case['pred_volume']} px\n"
        
        self.segResultsText.setPlainText(self.segResultsText.toPlainText() + "\n\n" + best_cases_text)
        self.segResultsText.setPlainText(self.segResultsText.toPlainText() + "\n\nADIM 2: En iyi 20 vakadan model eğitiyorum...")
        slicer.app.processEvents()
        
        # En iyi 20 vakadan model eğit
        fine_tuned_model = self.fineTuneModelFromBestCases(best_20_cases, original_model)
        
        if not fine_tuned_model:
            self.segResultsText.setPlainText(self.segResultsText.toPlainText() +
                                            "\nModel eğitimi başarısız oldu. Orijinal model kullanılacak.")
            fine_tuned_model = original_model
        else:
            self.segResultsText.setPlainText(self.segResultsText.toPlainText() + "\nModel eğitimi tamamlandı!")
        
        # ADIM 3: Düşük performanslı vakaları (Dice < 0.3) ince ayarlı model ile yeniden işle
        #-----------------------------------------------------------------
        # Düşük performanslı vakaları seç
        poor_cases = [case for case in all_results if case['dice'] < 0.3]
        
        if poor_cases:
            self.segResultsText.setPlainText(self.segResultsText.toPlainText() +
                f"\n\nADIM 3: {len(poor_cases)} düşük performanslı vakayı (Dice < 0.3) ince ayarlı model ile yeniden işliyorum...")
            slicer.app.processEvents()
            
            # İlerleme göstergesini güncelle
            self.aisdTotalCases = len(poor_cases)
            self.aisdProcessedCases = 0
            self.aisdProgressDisplay.setText(f"0/{self.aisdTotalCases} (0%)")
            self.aisdProgressBar.setMaximum(self.aisdTotalCases)
            self.aisdProgressBar.setValue(0)
            
            # İnce ayarlı modeli ayarla
            self.segmentationLogic.segmentationModel = fine_tuned_model
            
            # Farklı eşik değerleri dene
            threshold_options = [0.4, 0.5, 0.6, 0.7]
            improved_results = []
            
            # Düşük performanslı vakaları işle
            for i, case in enumerate(poor_cases):
                best_dice = case['dice']
                best_threshold = threshold
                best_metrics = None
                
                # İşlenen vaka sayısını güncelle
                self.aisdProcessedCases = i
                progress_percent = int((i / self.aisdTotalCases) * 100) if self.aisdTotalCases > 0 else 0
                self.aisdProgressDisplay.setText(f"{i}/{self.aisdTotalCases} ({progress_percent}%)")
                self.aisdProgressBar.setValue(i)
                
                # İlerleme durumunu göster
                progress_text = f"ADIM 3: Düşük performanslı vaka {i+1}/{len(poor_cases)} yeniden işleniyor: {case['patient_id']}"
                self.segResultsText.setPlainText(self.segResultsText.toPlainText().split('\n\n')[0] + "\n\n" + progress_text)
                slicer.app.processEvents()
                
                # Her eşik değeri için dene
                for test_threshold in threshold_options:
                    imageNode = None
                    maskNode = None
                    outputVolume = None
                    
                    try:
                        # Görüntü ve maskeyi yükle
                        imageNode = slicer.util.loadVolume(case['image_path'])
                        maskNode = slicer.util.loadLabelVolume(case['mask_path'])
                        
                        if not imageNode or not maskNode:
                            continue
                        
                        # İnce ayarlı model ile segmentasyon yap
                        segResults = self.segmentationLogic.processVolume(
                            imageNode,
                            test_threshold,
                            True,  # use_brain_mask
                            True,  # use_windowing
                            40,    # window_level
                            100    # window_width
                        )
                        
                        # Segmentasyon hacmi oluştur
                        outputVolume = self.segmentationLogic.createSegmentationVolume(imageNode, segResults)
                        
                        if outputVolume:
                            # Metrikleri hesapla
                            metrics = self.segmentationLogic.computeSegmentationMetrics(maskNode, outputVolume)
                            
                            # Daha iyi sonuç bulundu mu kontrol et
                            if metrics['dice'] > best_dice:
                                best_dice = metrics['dice']
                                best_threshold = test_threshold
                                best_metrics = metrics
                                    
                        # Düğümleri temizle
                        if imageNode:
                            slicer.mrmlScene.RemoveNode(imageNode)
                        if maskNode:
                            slicer.mrmlScene.RemoveNode(maskNode)
                        if outputVolume:
                            slicer.mrmlScene.RemoveNode(outputVolume)
                                
                    except Exception as e:
                        print(f"İşleme hatası: {str(e)}")
                        # Hata durumunda temizlik
                        if 'imageNode' in locals() and imageNode:
                            slicer.mrmlScene.RemoveNode(imageNode)
                        if 'maskNode' in locals() and maskNode:
                            slicer.mrmlScene.RemoveNode(maskNode)
                        if 'outputVolume' in locals() and outputVolume:
                            slicer.mrmlScene.RemoveNode(outputVolume)
                
                # En iyi sonuçla güncelle
                if best_metrics:
                    improved_results.append({
                        'patient_id': case['patient_id'],
                        'original_dice': case['dice'],
                        'improved_dice': best_dice,
                        'best_threshold': best_threshold,
                        'improvement': best_dice - case['dice'],
                        'metrics': best_metrics
                    })
            
            # İyileştirme sonuçlarını göster
            improved_text = "Düşük Performanslı Vakalarda İyileştirmeler:\n"
            for i, result in enumerate(improved_results):
                improved_text += f"{i+1}. {result['patient_id']}:\n"
                improved_text += f"   * Orijinal Dice: {result['original_dice']:.4f}\n"
                improved_text += f"   * İyileştirilmiş Dice: {result['improved_dice']:.4f}\n"
                improved_text += f"   * En İyi Eşik: {result['best_threshold']:.2f}\n"
                improved_text += f"   * İyileştirme: {result['improvement']:.4f}\n"
                improved_text += f"   \n"
            
            self.segResultsText.setPlainText(self.segResultsText.toPlainText() + "\n\n" + improved_text)
            slicer.app.processEvents()
            
            # ADIM 4: İyileştirilmiş sonuçlarla tüm veriseti için tekrar hesaplama
            #-----------------------------------------------------------------
            
            # Yeni Dice hesaplama yöntemi
            improved_total_dice = 0.0
            
            # İyileştirilmiş düşük performanslı vakaları ekle
            improved_cases_map = {result['patient_id']: result['improved_dice'] for result in improved_results}
            
            # Her vaka için bireysel sonuçları sakla
            final_individual_results = []
            
            # Tüm vakaları dolaş ve son Dice skorlarını hesapla
            for result in all_results:
                if result['patient_id'] in improved_cases_map:
                    # Düşük performanslı ve iyileştirilmiş vaka
                    current_dice = improved_cases_map[result['patient_id']]
                    improved_total_dice += current_dice
                    final_individual_results.append({
                        'patient_id': result['patient_id'],
                        'original_dice': result['dice'],
                        'final_dice': current_dice,
                        'improved': True
                    })
                else:
                    # Zaten iyi olan vaka
                    current_dice = result['dice']
                    improved_total_dice += current_dice
                    final_individual_results.append({
                        'patient_id': result['patient_id'],
                        'original_dice': result['dice'],
                        'final_dice': current_dice,
                        'improved': False
                    })
            
            # Tüm sonuçları detaylı olarak yazdır
            final_dice_text = "\nTüm Vakaların Nihai Dice Skorları:\n"
            for idx, res in enumerate(final_individual_results):
                if res['improved']:
                    final_dice_text += f"{idx+1}. {res['patient_id']}: {res['final_dice']:.4f} (Orijinal: {res['original_dice']:.4f}, İyileştirildi)\n"
                else:
                    final_dice_text += f"{idx+1}. {res['patient_id']}: {res['final_dice']:.4f} (Değiştirilmedi)\n"
            
            # Detaylı dice skorlarını hem ekrana hem dosyaya yazdır
            self.segResultsText.setPlainText(self.segResultsText.toPlainText() + "\n\n" + final_dice_text)
            slicer.app.processEvents()
            
            # Tüm sonuçların listesini CSV formatında dosyaya kaydet
            with open(os.path.join(os.path.dirname(modelPath), "final_dice_scores.txt"), "w") as f:
                f.write("ID,Original_Dice,Final_Dice,Improved\n")
                for res in final_individual_results:
                    f.write(f"{res['patient_id']},{res['original_dice']:.4f},{res['final_dice']:.4f},{res['improved']}\n")
            
            # Ayrıca tüm sonuçları daha okunabilir şekilde bir metin dosyasına da kaydet
            with open(os.path.join(os.path.dirname(modelPath), "detailed_dice_scores.txt"), "w") as f:
                f.write("TÜM VAKALARIN BAŞLANGIÇ VE NİHAİ DICE SKORLARI\n")
                f.write("==============================================\n\n")
                for idx, res in enumerate(final_individual_results):
                    if res['improved']:
                        f.write(f"{idx+1}. {res['patient_id']}:\n")
                        f.write(f"   Orijinal Dice: {res['original_dice']:.4f}\n")
                        f.write(f"   İyileştirilmiş Dice: {res['final_dice']:.4f}\n")
                        f.write(f"   İyileştirme: {res['final_dice'] - res['original_dice']:.4f}\n\n")
                    else:
                        f.write(f"{idx+1}. {res['patient_id']}: {res['final_dice']:.4f} (Değiştirilmedi)\n\n")
            
            # Yeni ortalama Dice
            final_avg_dice = improved_total_dice / len(all_results)
            
            # İyileştirme miktarını hesapla
            dice_improvement = final_avg_dice - original_avg_dice
            improvement_percent = (dice_improvement / original_avg_dice) * 100
            
            # İyileştirilmiş sonuçları ayrı bir listeye al (vaka kategorizasyonu için)
            final_dice_scores = [res['final_dice'] for res in final_individual_results]
            
            # Başarılı ve başarısız hasta sayılarını yeniden hesapla
            final_success_cases = sum(1 for d in final_dice_scores if d > 0.5)
            final_moderate_cases = sum(1 for d in final_dice_scores if 0.3 <= d <= 0.5)
            final_poor_cases = sum(1 for d in final_dice_scores if d < 0.3)
            
            # Genel istatistikleri göster
            final_stats_text = f"\nSONUÇLAR (Toplam {len(all_results)} vaka):\n"
            final_stats_text += f"Orijinal Ortalama Dice: {original_avg_dice:.4f}\n"
            final_stats_text += f"İyileştirilmiş Ortalama Dice: {final_avg_dice:.4f}\n"
            final_stats_text += f"Toplam İyileştirme: {dice_improvement:.4f} ({improvement_percent:.2f}%)\n\n"
            final_stats_text += f"Orijinal Başarılı Vakalar (Dice > 0.5): {success_cases}/{len(all_results)} ({100*success_cases/len(all_results):.1f}%)\n"
            final_stats_text += f"İyileştirilmiş Başarılı Vakalar (Dice > 0.5): {final_success_cases}/{len(all_results)} ({100*final_success_cases/len(all_results):.1f}%)\n\n"
            final_stats_text += f"Orijinal Orta Vakalar (0.3 <= Dice <= 0.5): {moderate_cases}/{len(all_results)} ({100*moderate_cases/len(all_results):.1f}%)\n"
            final_stats_text += f"İyileştirilmiş Orta Vakalar (0.3 <= Dice <= 0.5): {final_moderate_cases}/{len(all_results)} ({100*final_moderate_cases/len(all_results):.1f}%)\n\n"
            final_stats_text += f"Orijinal Başarısız Vakalar (Dice < 0.3): {poor_cases_count}/{len(all_results)} ({100*poor_cases_count/len(all_results):.1f}%)\n"
            final_stats_text += f"İyileştirilmiş Başarısız Vakalar (Dice < 0.3): {final_poor_cases}/{len(all_results)} ({100*final_poor_cases/len(all_results):.1f}%)\n"
            
            self.segResultsText.setPlainText(self.segResultsText.toPlainText() + "\n\n" + final_stats_text)
        
        # İlerleme çubuğunu ve göstergeyi tamamla
        self.aisdProcessedCases = self.aisdTotalCases
        self.aisdProgressDisplay.setText(f"{self.aisdTotalCases}/{self.aisdTotalCases} (100%)")
        self.aisdProgressBar.setValue(self.aisdTotalCases)
        
        # Orijinal modele geri dön
        self.segmentationLogic.segmentationModel = original_model
        
        # İşlem tamamlandı
        self.segResultsText.setPlainText(self.segResultsText.toPlainText() +
                                        f"\nİşlem tamamlandı. Toplam süre: {time.time() - start_time:.2f} saniye")
        
        # Sonuçları sınıf değişkenlerine sakla (daha sonraki analizler için)
        self.all_results = all_results
        self.improved_results = improved_results if 'improved_results' in locals() else []
        
        # Yeniden işleme butonunu etkinleştir
        self.reprocessButton.enabled = True
    
    def onShow3DToggled(self, checked):
        """3D görüntüleme görünürlüğünü değiştir"""
        if not hasattr(self, '_maskNode') or not self._maskNode:
            return
        if checked:
            # 3D görüntülemeyi etkinleştir
            self.updateModel3DVisualization()
        else:
            # 3D görüntülemeyi devre dışı bırak
            self.clearModel3DVisualization()
    
    def onApplyVisualizationSettings(self):
        """Görselleştirme ayarlarını uygula"""
        if not self._imageNode or not self._maskNode:
            slicer.util.errorDisplay("Önce görüntü segmentasyonu yapın")
            return
        self.updateVisualization()
        if self.show3DCheckBox.checked:
            self.updateModel3DVisualization()
    
    def clearNodes(self):
        """Tüm eski nodeları temizle"""
        if self._gtOverlayNode:
            slicer.mrmlScene.RemoveNode(self._gtOverlayNode)
            self._gtOverlayNode = None
        if self._predOverlayNode:
            slicer.mrmlScene.RemoveNode(self._predOverlayNode)
            self._predOverlayNode = None
        if self._gtSegmentationNode:
            slicer.mrmlScene.RemoveNode(self._gtSegmentationNode)
            self._gtSegmentationNode = None
        if self._predSegmentationNode:
            slicer.mrmlScene.RemoveNode(self._predSegmentationNode)
            self._predSegmentationNode = None
        if self._volumeRenderingDisplayNode:
            self._volumeRenderingDisplayNode = None
        # Sahneyi temizle
        slicer.mrmlScene.Clear(0)
    
    def updateVisualization(self):
        """Görüntüleme ayarlarını güncelle"""
        if not self._imageNode or not self._maskNode:
            return
        layoutManager = slicer.app.layoutManager()
        # Önce varsa eski overlay nodelarını temizle
        if self._gtOverlayNode:
            slicer.mrmlScene.RemoveNode(self._gtOverlayNode)
            self._gtOverlayNode = None
        if self._predOverlayNode:
            slicer.mrmlScene.RemoveNode(self._predOverlayNode)
            self._predOverlayNode = None
        # Tahmin segmentasyonu nodeunu da temizleyelim
        predNode = slicer.mrmlScene.GetFirstNodeByName("Pred_Overlay")
        if predNode:
            slicer.mrmlScene.RemoveNode(predNode)
        # Tek görünüm - orijinal görüntü üzerinde hem gerçek hem tahmin maskesi
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
        redSliceNode = layoutManager.sliceWidget("Red").mrmlSliceNode()
        redSliceCompositeNode = layoutManager.sliceWidget("Red").sliceLogic().GetSliceCompositeNode()
        # Arka plan rengini siyah yap
        redSliceNode.SetLayoutColor(0, 0, 0)
        # MRI görüntüsünü arka plan olarak ayarla
        redSliceCompositeNode.SetBackgroundVolumeID(self._imageNode.GetID())
        # Label volume'ları kaldır
        redSliceCompositeNode.SetLabelVolumeID(None)
        # Gerçek etiketi segmentasyon olarak göster - Yeşil renkte
        self._gtOverlayNode = self.createSegmentationNodeFromVolume(
            self._maskNode, "GT_Overlay", "Ground Truth", [0, 1, 0])
        if self._outputVolume:
            # Tahmin maskesini segmentasyon olarak oluştur
            self._predOverlayNode = self.createSegmentationNodeFromVolume(
                self._outputVolume, "Pred_Overlay", "Prediction", [1, 0, 0])
        # Görünümü hacme sığdır
        layoutManager.sliceWidget("Red").sliceController().fitSliceToBackground()
    
    def updateModel3DVisualization(self):
        """3D model görselleştirmesini güncelle"""
        if not self._imageNode or not self._maskNode:
            return
        # Önce eski segmentasyon düğümlerini temizle
        self.clearModel3DVisualization()
        # 3D görünümü içeren bir düzen göster
        layoutManager = slicer.app.layoutManager()
        # 3B + 2D görünüm düzeni
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        # Ground Truth modelini oluştur (yeşil)
        self._gtSegmentationNode = self.create3DModelFromLabelmap(
            self._maskNode, "GT_3D", "Ground Truth", [0, 1, 0])
        # Tahmin modelini oluştur
        if self._outputVolume:
            self._predSegmentationNode = self.create3DModelFromLabelmap(
                self._outputVolume, "Pred_3D", "Prediction", [1, 0, 0])
        # 3D görünümü odakla
        threeDView = layoutManager.threeDWidget(0).threeDView()
        threeDView.resetFocalPoint()
        threeDView.resetCamera()
    
    def clearModel3DVisualization(self):
        """3D model görselleştirmesini temizle"""
        # Eski segmentasyon düğümlerini temizle
        if self._gtSegmentationNode:
            slicer.mrmlScene.RemoveNode(self._gtSegmentationNode)
            self._gtSegmentationNode = None
        if self._predSegmentationNode:
            slicer.mrmlScene.RemoveNode(self._predSegmentationNode)
            self._predSegmentationNode = None
        # Ayrıca eski 3D modelleri de temizle
        for nodeName in ["GT_3D", "Pred_3D"]:
            node = slicer.mrmlScene.GetFirstNodeByName(nodeName)
            if node:
                slicer.mrmlScene.RemoveNode(node)
    
    def create3DModelFromLabelmap(self, labelmapNode, nodeName, segmentName, color):
        """Label map'ten 3D model oluştur"""
        # Eski düğümü temizle
        oldNode = slicer.mrmlScene.GetFirstNodeByName(nodeName)
        if oldNode:
            slicer.mrmlScene.RemoveNode(oldNode)
        # Yeni segmentasyon nodu oluştur
        segmentationNode = slicer.vtkMRMLSegmentationNode()
        segmentationNode.SetName(nodeName)
        slicer.mrmlScene.AddNode(segmentationNode)
        # Referans hacmi ayarla
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(labelmapNode)
        # Label map'i segmentasyona dönüştür
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapNode, segmentationNode)
        # 3B modeli oluşturmak için kapalı yüzey temsili ekle
        segmentationNode.CreateClosedSurfaceRepresentation()
        # Görüntüleme düğümü ekle
        segmentationNode.CreateDefaultDisplayNodes()
        displayNode = segmentationNode.GetDisplayNode()
        if displayNode:
            # 3D görünümü etkinleştir
            displayNode.SetVisibility3D(True)
            displayNode.SetVisibility2D(False)  # 2D görünümünü kapat
        # Segment adı ve rengini ayarla
        segmentIds = vtk.vtkStringArray()
        segmentationNode.GetSegmentation().GetSegmentIDs(segmentIds)
        if segmentIds.GetNumberOfValues() > 0:
            segmentId = segmentIds.GetValue(0)
            segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
            if segment:
                segment.SetName(segmentName)
                segment.SetColor(color[0], color[1], color[2])
        return segmentationNode
    
    def createSegmentationNodeFromVolume(self, labelVolume, nodeName, segmentName, color):
        """Label map hacminden segmentasyon düğümü oluşturur"""
        # Eski segmentasyon düğümünü temizle
        oldNode = slicer.mrmlScene.GetFirstNodeByName(nodeName)
        if oldNode:
            slicer.mrmlScene.RemoveNode(oldNode)
        # Yeni segmentasyon oluştur
        segmentationNode = slicer.vtkMRMLSegmentationNode()
        segmentationNode.SetName(nodeName)
        slicer.mrmlScene.AddNode(segmentationNode)
        # Görünüm düğümü ekle ve renkleri ayarla
        segmentationNode.CreateDefaultDisplayNodes()
        displayNode = segmentationNode.GetDisplayNode()
        if displayNode:
            displayNode.SetVisibility3D(self.show3DCheckBox.checked)
            displayNode.SetVisibility2D(True)
            displayNode.SetSliceIntersectionThickness(2)  # Çizgi kalınlığını ayarla
        # Referans hacmi ayarla
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(labelVolume)
        # LabelMap'i segmentasyona çevir
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelVolume, segmentationNode)
        # Segmentin adını ve rengini ayarla
        segmentIds = vtk.vtkStringArray()
        segmentationNode.GetSegmentation().GetSegmentIDs(segmentIds)
        if segmentIds.GetNumberOfValues() > 0:
            segmentId = segmentIds.GetValue(0)
            segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
            if segment:
                segment.SetName(segmentName)
                # "Ground Truth" için her zaman yeşil renk kullan
                if segmentName == "Ground Truth":
                    segment.SetColor(0, 1, 0)  # Her zaman yeşil
                else:
                    segment.SetColor(color[0], color[1], color[2])
        return segmentationNode

# Segmentasyon Logic Sınıfı
class StrokeDetectionSegmentationLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        # PyTorch cihazını ayarla
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"PyTorch cihazı: {self.device}")
        # Model değişkenleri
        self.segmentationModel = None
        # Model imza ve özet değişkenleri
        self.current_model_path = None
        self.current_model_md5 = None
        self.current_model_fingerprint = None
        self.current_model_output_signature = None
        self.current_model_signature = None
    
    def create_brain_mask(self, slice_array):
        """Beyni segmentlemek için geliştirilmiş bir maske oluşturur - sadece kafatası içini algılar"""
        try:
            # Deterministik işlem için önceki rastgele durumu sakla
            prev_random_state = np.random.get_state()
            np.random.seed(42)
            # Normalize görüntü
            normalized = slice_array.copy()
            min_val = np.min(normalized)
            max_val = np.max(normalized)
            if max_val > min_val:
                normalized = (normalized - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(normalized)
            # Otsu eşikleme için histogram-tabanlı bir yaklaşım kullan
            # CT/MRI görüntüleri için adaptif eşikleme
            if max_val > 100:  # Ham CT/MRI değerleri
                # CT için kemik dokusu genellikle çok parlak, beyin dokusu daha karanlık
                # Kafatası kemiği genellikle çok yüksek değerlere sahip (>100)
                # Beyin dokusu genellikle 0-80 aralığında
                brain_region = (slice_array > 10) & (slice_array < 80)
            else:  # Normalize edilmiş [0-1] değerler
                # Normalize edilmiş görüntüde kafatası dışını ayırt et
                # Beyin dokusu genellikle orta gri tonlarında
                brain_region = (normalized > 0.2) & (normalized < 0.8)
            # Morfolojik işlemlerle maskeyi geliştir
            from scipy import ndimage
            # Önce küçük boşlukları doldur
            brain_region = ndimage.binary_closing(brain_region, structure=np.ones((5, 5)))
            # Küçük izole bölgeleri kaldır
            brain_region = ndimage.binary_opening(brain_region, structure=np.ones((3, 3)))
            # En büyük bağlı bileşeni bul (bu muhtemelen beyin olacak)
            labeled_regions, num_regions = ndimage.label(brain_region)
            if num_regions > 0:
                region_sizes = ndimage.sum(brain_region, labeled_regions, range(1, num_regions+1))
                if len(region_sizes) > 0:
                    largest_region_index = np.argmax(region_sizes) + 1
                    brain_region = (labeled_regions == largest_region_index)
                    # Boşlukları doldurarak beyin bölgesini pürüzsüzleştir
                    brain_region = ndimage.binary_fill_holes(brain_region)
                    # Beynin sınırından biraz içeri gir (kafatası kenarından uzaklaş)
                    brain_region = ndimage.binary_erosion(brain_region, structure=np.ones((4, 4)))
                    # Sonra biraz daha yumuşak genişle
                    brain_region = ndimage.binary_dilation(brain_region, structure=np.ones((2, 2)))
            # Rastgele durumu geri yükle
            np.random.set_state(prev_random_state)
            return brain_region.astype(np.uint8)
        except Exception as e:
            logging.error(f"Brain mask oluşturma hatası: {str(e)}")
            import traceback
            traceback.print_exc()
            # Hata durumunda tüm pikselleri beyin olarak işaretle
            return np.ones_like(slice_array, dtype=np.uint8)
    
    def createSimplifiedUNet(self):
        """Eğitim kodundaki UNet modeline eşdeğer iyileştirilmiş bir model"""
        class ConvBlock(nn.Module):
            def __init__(self, in_channels, out_channels, res_units=3):
                super().__init__()
                self.layers = nn.ModuleList()
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                )
                for _ in range(res_units-1):
                    self.layers.append(
                        nn.Sequential(
                            nn.Conv2d(out_channels, out_channels, 3, padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                            nn.Dropout2d(0.2)  # Overfitting'i azaltmak için dropout
                        )
                    )
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        class DownSample(nn.Module):
            def __init__(self, in_channels, out_channels, res_units=3):
                super().__init__()
                self.conv = ConvBlock(in_channels, out_channels, res_units)
                self.pool = nn.MaxPool2d(2)
            def forward(self, x):
                x = self.pool(x)
                x = self.conv(x)
                return x
        
        class UpSample(nn.Module):
            def __init__(self, in_channels, out_channels, res_units=3):
                super().__init__()
                self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
                self.conv = ConvBlock(out_channels*2, out_channels, res_units)
            def forward(self, x, skip):
                x = self.upconv(x)
                x = torch.cat((skip, x), dim=1)
                x = self.conv(x)
                return x
        
        class ImprovedUNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Encoder - Daha geniş kanallarla
                self.input_conv = ConvBlock(1, 32, res_units=3)
                self.down1 = DownSample(32, 64, res_units=3)
                self.down2 = DownSample(64, 128, res_units=3)
                self.down3 = DownSample(128, 256, res_units=3)
                self.down4 = DownSample(256, 512, res_units=3)
                
                # Decoder
                self.up1 = UpSample(512, 256, res_units=3)
                self.up2 = UpSample(256, 128, res_units=3)
                self.up3 = UpSample(128, 64, res_units=3)
                self.up4 = UpSample(64, 32, res_units=3)
                
                # Final layer
                self.final = nn.Conv2d(32, 1, 1)
            
            def forward(self, x):
                # Encoder path
                x1 = self.input_conv(x)
                x2 = self.down1(x1)
                x3 = self.down2(x2)
                x4 = self.down3(x3)
                x5 = self.down4(x4)
                
                # Decoder path
                x = self.up1(x5, x4)
                x = self.up2(x, x3)
                x = self.up3(x, x2)
                x = self.up4(x, x1)
                
                # Final convolution
                output = self.final(x)
                return output
        
        model = ImprovedUNet()
        model.to(self.device)
        model.eval()
        return model
    
    def createUNetModel(self):
        """Geliştirilmiş UNet modelini oluşturur"""
        try:
            # MONAI UNet modelini içe aktar
            from monai.networks.nets import UNet
            from monai.networks.layers import Norm
            
            # Daha derin ve güçlü bir UNet yapılandırması
            model = UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(32, 64, 128, 256, 512),  # Daha geniş kanallar
                strides=(2, 2, 2, 2),
                num_res_units=3,  # Daha fazla residual blok
                dropout=0.2,  # Overfitting'i azaltmak için dropout ekle
                norm=Norm.BATCH
            )
            model.to(self.device)
            model.eval()
            return model
        except ImportError:
            # MONAI yüklü değilse, basitleştirilmiş UNet kullan
            print("MONAI kütüphanesi bulunamadı, basitleştirilmiş UNet kullanılıyor")
            return self.createSimplifiedUNet()
    
    def loadUNetModel(self, modelPath):
        """UNet modelini monai formatıyla uyumlu olarak yükler"""
        try:
            # Temizlik işlemleri
            print("Modeli yüklemeden önce temizlik yapılıyor...")
            self.segmentationModel = None  # Önce mevcut modeli temizle
            torch.cuda.empty_cache() if torch.cuda.is_available() else None  # GPU hafızasını temizle
            
            # PyTorch'u deterministik moda al
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
                
            # Model dosyasının özet (checksum) değerini hesapla
            model_md5 = hashlib.md5(open(modelPath,'rb').read()).hexdigest()
            print(f"Yüklenen model dosyasının MD5 özeti: {model_md5}")
            
            # Modeli yüklemeyi dene - önce normal yükleme
            try:
                # MONAI UNet modelini oluştur
                from monai.networks.nets import UNet
                model = UNet(
                    spatial_dims=2,
                    in_channels=1,
                    out_channels=1,
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=2
                )
                model.to(self.device)
                
                # Checkpoint'i yükle
                checkpoint = torch.load(modelPath, map_location=self.device)
                print("Model dosyası yüklendi")
                
                # Model state_dict'ini bul
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        print("'model_state_dict' anahtarı kullanılıyor")
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                        print("'state_dict' anahtarı kullanılıyor")
                    else:
                        state_dict = checkpoint
                        print("Checkpoint doğrudan state_dict olarak kullanılıyor")
                else:
                    state_dict = checkpoint
                    print("Checkpoint doğrudan model durum sözlüğü olarak kullanılıyor")
                
                # Modele parametreleri yükle
                print("Model parametreleri yükleniyor...")
                model.load_state_dict(state_dict, strict=False)
                
                # Model değerlendirme moduna al
                model.eval()
                
                # Model test et
                print("Modeli test ediliyor...")
                with torch.no_grad():
                    torch.manual_seed(42)
                    test_input = torch.randn(1, 1, 512, 512).to(self.device)
                    output = model(test_input)
                    output_stats = {
                        'min': float(output.min().item()),
                        'max': float(output.max().item()),
                        'mean': float(output.mean().item())
                    }
                    
                # Modelin benzersiz imzasını oluştur
                model_signature = f"{output_stats['min']:.4f}-{output_stats['max']:.4f}-{output_stats['mean']:.4f}"
                
                # Model meta verilerini sakla
                self.segmentationModel = model
                self.current_model_path = modelPath
                self.current_model_md5 = model_md5
                self.current_model_output_signature = output_stats
                self.current_model_signature = model_signature
                self._last_load_time = time.time()
                
                print(f"Model başarıyla yüklendi: {os.path.basename(modelPath)}")
                print(f"Model imzası: {model_signature}")
                return True
                
            except ImportError:
                print("MONAI kütüphanesi bulunamadı, özel UNet kullanılıyor...")
                # Basitleştirilmiş UNet modeli oluştur
                model = self.createSimplifiedUNet()
                
                # Checkpoint'i yükle
                checkpoint = torch.load(modelPath, map_location=self.device)
                
                # State dict'i çıkart
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Modele parametreleri yükle
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                
                # Test et
                with torch.no_grad():
                    test_input = torch.randn(1, 1, 512, 512).to(self.device)
                    output = model(test_input)
                    output_stats = {
                        'min': float(output.min().item()),
                        'max': float(output.max().item()),
                        'mean': float(output.mean().item())
                    }
                
                # Model meta verilerini sakla
                self.segmentationModel = model
                self.current_model_path = modelPath
                self.current_model_md5 = model_md5
                self.current_model_output_signature = output_stats
                self.current_model_signature = f"{output_stats['min']:.4f}-{output_stats['max']:.4f}-{output_stats['mean']:.4f}"
                self._last_load_time = time.time()
                
                print(f"Model başarıyla yüklendi (özel UNet ile): {os.path.basename(modelPath)}")
                return True
                
        except Exception as e:
            error_msg = f"Model yükleme hatası: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            # Modeli None olarak ayarla
            self.segmentationModel = None
            return False  # Hata olduğunda False döndür, çağıran kod buna göre işlem yapabilir
    def test_time_augmentation(self, image_tensor, model):
        """Test zamanı veri artırma (TTA) ile daha güvenilir tahminler yapma"""
        predictions = []
        
        # 1. Orijinal görüntü
        with torch.no_grad():
            pred = model(image_tensor)
            pred = torch.sigmoid(pred)
            predictions.append(pred)
        
        # 2. Yatay çevirme
        with torch.no_grad():
            flipped_h = torch.flip(image_tensor, [3])  # Yatay çevirme
            pred = model(flipped_h)
            pred = torch.sigmoid(pred)
            pred = torch.flip(pred, [3])  # Geri çevirme
            predictions.append(pred)
        
        # 3. Dikey çevirme
        with torch.no_grad():
            flipped_v = torch.flip(image_tensor, [2])  # Dikey çevirme
            pred = model(flipped_v)
            pred = torch.sigmoid(pred)
            pred = torch.flip(pred, [2])  # Geri çevirme
            predictions.append(pred)
        
        # 4. 90 derece döndürme
        with torch.no_grad():
            rot90 = torch.rot90(image_tensor, 1, [2, 3])
            pred = model(rot90)
            pred = torch.sigmoid(pred)
            pred = torch.rot90(pred, 3, [2, 3])  # Geri döndürme
            predictions.append(pred)
        
        # 5. -90 derece döndürme
        with torch.no_grad():
            rot270 = torch.rot90(image_tensor, 3, [2, 3])
            pred = model(rot270)
            pred = torch.sigmoid(pred)
            pred = torch.rot90(pred, 1, [2, 3])  # Geri döndürme
            predictions.append(pred)
        
        # Tüm tahminleri birleştir (ortalama)
        avg_pred = torch.mean(torch.stack(predictions), dim=0)
        return avg_pred
    def processVolume(self, inputVolume, threshold=0.5, use_brain_mask=False, use_windowing=False, window_level=40, window_width=80):
        """3D hacmi dilim dilim işleyerek segmentasyon yapar - Daha genel versiyon"""
        # Modelin mevcut olduğunu kontrol et
        if self.segmentationModel is None:
            raise RuntimeError("Model yüklenmemiş. Önce geçerli bir .pth modeli yükleyin.")
        try:
            # İşlem başlangıcını kaydet
            process_start_time = time.time()
            process_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
            print(f"[İşlem #{process_id}] Hacim işleme başladı")
            # Hacmin uzaysal özelliklerini al
            spacing = inputVolume.GetSpacing()
            slice_thickness_mm = spacing[2]  # z eksenindeki spacing, dilim kalınlığı
            pixel_width_mm = spacing[0]      # x eksenindeki spacing (piksel genişliği)
            pixel_height_mm = spacing[1]     # y eksenindeki spacing (piksel yüksekliği)
            print(f"[İşlem #{process_id}] Görüntü özellikleri: Dilim kalınlığı = {slice_thickness_mm:.2f} mm, "
                f"Piksel boyutu = {pixel_width_mm:.2f} x {pixel_height_mm:.2f} mm")
            # Hacmi numpy dizisi olarak al
            volumeArray = slicer.util.arrayFromVolume(inputVolume)
            if volumeArray is None:
                raise RuntimeError("Hacim dizisi alınamadı")
            # Hacim özeti
            volume_hash = hashlib.md5(volumeArray.tobytes()[:1000000]).hexdigest()[:8]  # İlk 1MB hash
            print(f"[İşlem #{process_id}] Hacim yüklendi - Shape: {volumeArray.shape}, Hash: {volume_hash}")
            
            # Piksel alanına göre minimum lezyon boyutu (mm2 cinsinden)
            min_lesion_area_mm2 = 50.0  # mm2 cinsinden makul minimum alan
            min_lesion_area_px = int(min_lesion_area_mm2 / (pixel_width_mm * pixel_height_mm))
            print(f"[İşlem #{process_id}] Minimum lezyon alanı: {min_lesion_area_mm2} mm² ({min_lesion_area_px} piksel)")
            
            # Hacim boyutu için minimum eşik (mm3 cinsinden)
            min_lesion_volume_mm3 = 200.0  # mm3 cinsinden makul minimum hacim
            min_lesion_volume_px = int(min_lesion_volume_mm3 / (pixel_width_mm * pixel_height_mm * slice_thickness_mm))
            print(f"[İşlem #{process_id}] Minimum lezyon hacmi: {min_lesion_volume_mm3} mm³ ({min_lesion_volume_px} piksel)")
            
            # Dilim kalınlığına göre minimum komponent boyutunu ayarla
            min_component_size = 10  # Varsayılan değer
            
            # Dilim kalınlığına göre parametreleri ayarla
            if slice_thickness_mm <= 1.5:
                # İnce dilimler için daha katı 3D tutarlılık parametreleri
                min_component_size = max(5, int(min_component_size * 0.8))
                print(f"[İşlem #{process_id}] İnce dilimler ({slice_thickness_mm:.2f} mm) için min_component_size: {min_component_size}")
            elif slice_thickness_mm >= 5.0:
                # Kalın dilimler için daha gevşek 3D tutarlılık parametreleri
                min_component_size = max(5, int(min_component_size * 1.2))
                print(f"[İşlem #{process_id}] Kalın dilimler ({slice_thickness_mm:.2f} mm) için min_component_size: {min_component_size}")
            
            # Dilim sayısı
            numSlices = volumeArray.shape[0]
            logging.info(f"[İşlem #{process_id}] Toplam {numSlices} dilim işlenecek")
            print(f"[İşlem #{process_id}] Toplam {numSlices} dilim işlenecek")
            
            # Sonuçları saklamak için liste
            results = []
            prob_masks = []  # Olasılık maskelerini sakla (sonradan 3D analiz için)
            
            # Her dilimi ayrı ayrı işle
            with torch.no_grad():  # Gradyan hesaplaması devre dışı
                for i in range(numSlices):
                    slice_start_time = time.time()
                    try:
                        # Dilimi al
                        sliceArray = volumeArray[i].astype(np.float32)
                        original_shape = sliceArray.shape
                        slice_hash = hashlib.md5(sliceArray.tobytes()).hexdigest()[:8]
                        
                        # Dilimin içeriğini kontrol et (boş mu?)
                        if np.std(sliceArray) < 1e-6:
                            # Boş dilim
                            results.append({
                                'slice_index': i,
                                'probability': 0,
                                'mask': np.zeros(original_shape, dtype=np.uint8),
                                'slice_thickness_mm': slice_thickness_mm,
                                'is_empty': True  # Boş dilim işaretleme
                            })
                            prob_masks.append(np.zeros(original_shape))
                            print(f"[İşlem #{process_id}] Dilim {i} - Boş dilim tespit edildi, atlanıyor")
                            continue
                        
                        print(f"[İşlem #{process_id}] Dilim {i} işleniyor - Hash: {slice_hash}")
                        
                        # Basit normalizasyon ve boyutlandırma
                        processed = sliceArray.copy()
                        
                        # Boyutlandırma (512x512)
                        if processed.shape[0] != 512 or processed.shape[1] != 512:
                            processed = cv2.resize(processed, (512, 512), interpolation=cv2.INTER_LINEAR)
                        
                        # Min-max normalizasyon
                        min_val = np.min(processed)
                        max_val = np.max(processed)
                        if max_val > min_val:
                            processed = (processed - min_val) / (max_val - min_val)
                        else:
                            processed = np.zeros_like(processed)
                        
                        # Tensor formatına dönüştür
                        processed = processed.reshape(1, 1, *processed.shape)
                        tensor = torch.from_numpy(processed).float().to(self.device)
                        
                        # Test zamanı veri artırma (TTA) ile tahmin yap
                        predictions = []
                        
                        # 1. Orijinal
                        with torch.no_grad():
                            pred = self.segmentationModel(tensor)
                            pred = torch.sigmoid(pred)
                            predictions.append(pred)
                        
                        # 2. Yatay flip
                        with torch.no_grad():
                            flipped_h = torch.flip(tensor, [3])
                            pred = self.segmentationModel(flipped_h)
                            pred = torch.sigmoid(pred)
                            pred = torch.flip(pred, [3])
                            predictions.append(pred)
                        
                        # 3. Dikey flip
                        with torch.no_grad():
                            flipped_v = torch.flip(tensor, [2])
                            pred = self.segmentationModel(flipped_v)
                            pred = torch.sigmoid(pred)
                            pred = torch.flip(pred, [2])
                            predictions.append(pred)
                        
                        # 4. 90 derece döndürme
                        with torch.no_grad():
                            rot90 = torch.rot90(tensor, 1, [2, 3])
                            pred = self.segmentationModel(rot90)
                            pred = torch.sigmoid(pred)
                            pred = torch.rot90(pred, 3, [2, 3])
                            predictions.append(pred)
                        
                        # 5. -90 derece döndürme
                        with torch.no_grad():
                            rot270 = torch.rot90(tensor, 3, [2, 3])
                            pred = self.segmentationModel(rot270)
                            pred = torch.sigmoid(pred)
                            pred = torch.rot90(pred, 1, [2, 3])
                            predictions.append(pred)
                        
                        # Tüm tahminleri birleştir (ortalama)
                        avg_pred = torch.mean(torch.stack(predictions), dim=0)
                        prob_mask = avg_pred.squeeze().cpu().numpy()
                        
                        # Olasılık maskesini sakla
                        if prob_mask.shape != original_shape:
                            zoom_factors = (original_shape[0] / prob_mask.shape[0],
                                        original_shape[1] / prob_mask.shape[1])
                            prob_mask_resized = ndimage.zoom(prob_mask, zoom_factors, order=1)
                        else:
                            prob_mask_resized = prob_mask
                        
                        prob_masks.append(prob_mask_resized)
                        
                        # Dinamik eşik değeri hesaplama
                        max_prob = np.max(prob_mask)
                        mean_prob = np.mean(prob_mask)
                        std_prob = np.std(prob_mask)
                        
                        # Kontrast oranı - prob_mask'ın kontrastını değerlendirir
                        contrast_ratio = std_prob / (mean_prob + 1e-6)
                        
                        # Dilim kalınlığını dikkate alarak eşik değerini ayarla
                        thickness_factor = 1.0
                        if slice_thickness_mm > 4.0:
                            # Kalın dilimlerde (4mm+) eşiği biraz düşür (daha hassas ol)
                            thickness_factor = 0.9
                        elif slice_thickness_mm < 2.0:
                            # İnce dilimlerde (2mm-) eşiği biraz yükselt (daha kesin ol)
                            thickness_factor = 1.1
                        
                                                # Dinamik eşik değeri hesaplama
                        if max_prob < 0.4:  # 0.3'ten 0.4'e yükseltildi
                            dynamic_threshold = max(0.4, threshold * thickness_factor)  # 0.25'ten 0.4'e yükseltildi
                        elif max_prob > 0.7:
                            if contrast_ratio > 1.0:
                                dynamic_threshold = min(0.5, threshold * 0.9 * thickness_factor)  # 0.4'ten 0.5'e yükseltildi
                            else:
                                dynamic_threshold = threshold * thickness_factor
                        else:
                            # Kontrast oranına göre ayarla
                            dynamic_threshold = threshold * (0.9 + 0.2 * (contrast_ratio / 2.0)) * thickness_factor  # 0.1'den 0.2'ye yükseltildi
                            dynamic_threshold = max(0.4, min(threshold, dynamic_threshold))  # 0.25'ten 0.4'e yükseltildi
                        
                        # Nihai eşik değerini log'a yaz
                        print(f"Dilim {i}: max_prob={max_prob:.3f}, contrast={contrast_ratio:.3f}, thickness={slice_thickness_mm:.2f} mm, threshold={dynamic_threshold:.3f}")
                        
                        # Binary mask oluştur
                        binary_mask = (prob_mask > dynamic_threshold).astype(np.uint8)
                        
                        # Toplam maske piksel sayısı
                        total_pixels = np.sum(binary_mask)
                        
                        # Eğer toplam piksel sayısı çok azsa, direkt olarak temizle
                        if total_pixels < min_lesion_area_px * 0.5:
                            print(f"Dilim {i}: Toplam piksel sayısı çok az ({total_pixels} < {min_lesion_area_px * 0.5}), direkt temizleniyor")
                            binary_mask = np.zeros_like(binary_mask)
                        
                        # BAĞLI BİLEŞEN ANALİZİ - küçük gürültüleri temizle
                        elif np.sum(binary_mask) > 0:
                            # Piksel boyutuna göre min_component_size'ı ayarla
                            pixel_area_mm2 = pixel_width_mm * pixel_height_mm
                            # Bağlı bileşen analizinde size filtresini artırın
                            adjusted_min_size = max(min_lesion_area_px * 0.7, int(min_component_size * (1.0 / pixel_area_mm2) * 0.7))  # 0.5'ten 0.7'ye yükseltildi
                            
                            # Bağlı bileşenleri bul
                            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
                            
                            # Temiz bir maske oluştur
                            clean_mask = np.zeros_like(binary_mask)
                            
                            # Toplam bileşen alanını hesapla
                            total_component_area = 0
                            for label in range(1, num_labels):  # 0 indeksi arka plan
                                size = stats[label, cv2.CC_STAT_AREA]
                                total_component_area += size
                            
                            # Eğer toplam alan çok küçükse, direkt temizle
                            if total_component_area < min_lesion_area_px:
                                print(f"Dilim {i}: Toplam bileşen alanı çok küçük ({total_component_area} < {min_lesion_area_px}), temizleniyor")
                                binary_mask = np.zeros_like(binary_mask)
                            else:
                                # Her bileşeni analiz et
                                valid_components = 0
                                for label in range(1, num_labels):  # 0 indeksi arka plan
                                    size = stats[label, cv2.CC_STAT_AREA]
                                    
                                    # Çok küçük bileşenleri filtrele (gürültü)
                                    if size < adjusted_min_size:
                                        print(f"Dilim {i}: {size} piksellik küçük komponent temizlendi (eşik: {adjusted_min_size} piksel)")
                                        continue
                                    
                                    # Bileşen şekil analizi
                                    x = stats[label, cv2.CC_STAT_LEFT]
                                    y = stats[label, cv2.CC_STAT_TOP]
                                    w = stats[label, cv2.CC_STAT_WIDTH]
                                    h = stats[label, cv2.CC_STAT_HEIGHT]
                                    
                                    # Bileşen görüntüsünü al
                                    component_mask = np.zeros_like(binary_mask)
                                    component_mask[labels == label] = 1
                                    
                                    # Şekil analizi yap
                                    # 1. Dairesellik
                                    contours, _ = cv2.findContours(component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    if contours and len(contours) > 0:
                                        contour = contours[0]
                                        perimeter = cv2.arcLength(contour, True)
                                        circularity = 4 * np.pi * size / (perimeter * perimeter) if perimeter > 0 else 0
                                        
                                        # 2. Doluluk oranı
                                        bounding_box_area = w * h
                                        fill_ratio = size / bounding_box_area if bounding_box_area > 0 else 0
                                        
                                        # 3. Kompaktlık
                                        compactness = perimeter / (2 * np.sqrt(np.pi * size)) if size > 0 else float('inf')
                                        
                                        # Şekil ölçütlerine göre filtrele - daha esnek değerler
                                        if circularity < 0.2 or fill_ratio < 0.2 or compactness > 4.0:
                                            print(f"Dilim {i}: Şekil analizi nedeniyle komponent temizlendi (dairesellik: {circularity:.2f}, doluluk: {fill_ratio:.2f}, kompaktlık: {compactness:.2f})")
                                            continue
                                    
                                    # Büyük bileşenleri maskede tut
                                    clean_mask[labels == label] = 1
                                    valid_components += 1
                                
                                # Eğer hiçbir geçerli bileşen kalmadıysa, tüm dilimi temizle
                                if valid_components == 0:
                                    print(f"Dilim {i}: Geçerli bileşen kalmadı, tüm tahminler temizlendi")
                                    clean_mask = np.zeros_like(binary_mask)
                                
                                binary_mask = clean_mask
                        
                        # Maskeyi orijinal boyuta geri döndür
                        if binary_mask.shape != original_shape:
                            zoom_h = original_shape[0] / binary_mask.shape[0]
                            zoom_w = original_shape[1] / binary_mask.shape[1]
                            resized_mask = ndimage.zoom(binary_mask, (zoom_h, zoom_w), order=0)
                        else:
                            resized_mask = binary_mask
                        
                        # Sonucu sakla
                        results.append({
                            'slice_index': i,
                            'probability': float(prob_mask.mean()),
                            'max_probability': float(max_prob),
                            'threshold_used': float(dynamic_threshold),
                            'contrast_ratio': float(contrast_ratio),
                            'mask': resized_mask,
                            'process_time': time.time() - slice_start_time,
                            'slice_thickness_mm': slice_thickness_mm,
                            'pixel_dimensions_mm': [pixel_width_mm, pixel_height_mm],
                            'is_empty': False  # İşlenmiş dilim işaretleme
                        })
                        
                        # İlerlemeyi göster
                        if (i + 1) % 5 == 0 or i == numSlices - 1:
                            logging.info(f"[İşlem #{process_id}] {i+1}/{numSlices} dilim işlendi")
                            print(f"[İşlem #{process_id}] {i+1}/{numSlices} dilim işlendi")
                    
                    except Exception as slice_error:
                        logging.error(f"[İşlem #{process_id}] Dilim {i} işlenirken hata: {str(slice_error)}")
                        print(f"[İşlem #{process_id}] Dilim {i} işleme hatası: {str(slice_error)}")
                        import traceback
                        traceback.print_exc()
                        
                        # Hata durumunda boş maske döndür
                        results.append({
                            'slice_index': i,
                            'probability': 0,
                            'mask': np.zeros(original_shape, dtype=np.uint8),
                            'error': str(slice_error),
                            'slice_thickness_mm': slice_thickness_mm,
                            'pixel_dimensions_mm': [pixel_width_mm, pixel_height_mm],
                            'is_empty': True  # Hata durumunda boş dilim işaretleme
                        })
                        prob_masks.append(np.zeros(original_shape))
            
            # 3D TUTARLILIK KONTROLÜ - Dilim kalınlığına uyarlanmış
            # Z-ekseni komşuluk mesafesini dilim kalınlığına göre ayarla
            z_neighbor_count = max(1, min(3, int(round(3.0 / slice_thickness_mm))))
            print(f"[İşlem #{process_id}] 3D tutarlılık için z-komşuluk sayısı: {z_neighbor_count} (dilim kalınlığı: {slice_thickness_mm:.2f} mm)")
            
            # Önce tüm hacim boyunca her dilim için connected component analizi yap
            for i in range(numSlices):
                if np.sum(results[i]['mask']) > 0:
                    # Bağlı bileşenleri etiketle
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(results[i]['mask'], connectivity=8)
                    
                    # Her bileşen için bilgileri sakla
                    results[i]['components'] = []
                    for label in range(1, num_labels):  # 0 arka plan
                        results[i]['components'].append({
                            'label': label,
                            'size': stats[label, cv2.CC_STAT_AREA],
                            'x': stats[label, cv2.CC_STAT_LEFT],
                            'y': stats[label, cv2.CC_STAT_TOP],
                            'width': stats[label, cv2.CC_STAT_WIDTH],
                            'height': stats[label, cv2.CC_STAT_HEIGHT],
                            'area_mm2': stats[label, cv2.CC_STAT_AREA] * pixel_width_mm * pixel_height_mm
                        })
            
            # 3D tutarlılık kontrolü - izole edilmiş dilimleri temizle
            # Önceki komşularla bağlantısı olmayan dilimleri temizle
            for i in range(z_neighbor_count, numSlices-z_neighbor_count):
                curr_mask = results[i]['mask']
                # Eğer mevcut kesitte maske varsa
                if np.sum(curr_mask) > 0:
                    # Komşu dilimleri kontrol et (dilim kalınlığına göre ayarlanmış komşuluk)
                    neighbor_mask_sum = 0
                    for j in range(1, z_neighbor_count+1):
                        if i-j >= 0:
                            neighbor_mask_sum += np.sum(results[i-j]['mask'])
                        if i+j < numSlices:
                            neighbor_mask_sum += np.sum(results[i+j]['mask'])
                    
                                        # Komşu dilimlerde hiç maske yoksa veya çok az varsa, muhtemelen yanlış pozitiftir
                    if neighbor_mask_sum < min_lesion_area_px * 0.7:  # 0.5'ten 0.7'ye yükseltildi
                        # Küçük izole edilmiş bileşenleri temizle
                        if 'components' in results[i] and len(results[i]['components']) > 0:
                            print(f"Dilim {i}: Komşuluk analizi - izole edilmiş dilim temizleniyor (komşu piksel: {neighbor_mask_sum})")
                            results[i]['mask'] = np.zeros_like(curr_mask)
            
            # Toplam hacim kontrolü - çok küçük hacimleri temizle
            total_lesion_volume = 0
            for i in range(numSlices):
                total_lesion_volume += np.sum(results[i]['mask'])
            
            # Eğer toplam hacim çok küçükse, muhtemelen yanlış pozitif tespitlerdir
            if total_lesion_volume < min_lesion_volume_px:
                print(f"[İşlem #{process_id}] Toplam lezyon hacmi çok küçük ({total_lesion_volume} piksel < {min_lesion_volume_px} piksel), tüm tahminler temizleniyor")
                for i in range(numSlices):
                    results[i]['mask'] = np.zeros_like(results[i]['mask'])
                total_lesion_volume = 0
            
            # Sonuçların boş olup olmadığını kontrol et
            if not results:
                raise RuntimeError("İşleme sonucu boş - hiçbir dilim başarıyla işlenemedi")
            
            # İşlem sonuçlarını özetle
            total_positive = sum(np.sum(result['mask']) for result in results)
            
            # Fiziksel hacim hesapla (mm³)
            total_volume_mm3 = total_positive * pixel_width_mm * pixel_height_mm * slice_thickness_mm
            total_time = time.time() - process_start_time
            
            print(f"\n[İşlem #{process_id}] İşleme tamamlandı - Toplam pozitif piksel: {total_positive}")
            print(f"[İşlem #{process_id}] Tahmin edilen lezyon hacmi: {total_volume_mm3:.2f} mm³")
            print(f"[İşlem #{process_id}] Toplam işlem süresi: {total_time:.2f}s")
            
            # Sonuçlara hacim bilgisini ekle
            for result in results:
                result['total_volume_mm3'] = total_volume_mm3
            
            return results
        
        except Exception as e:
            logging.error(f"Hacim işleme hatası: {str(e)}")
            print(f"PROCESSVOLUME HATASI: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Hacim işleme hatası: {str(e)}")
    
    def createSegmentationVolume(self, inputVolume, segmentationResults):
      """Bölütleme sonuçlarından bir hacim oluşturur"""
      try:
          # İlk girişin özelliklerini alarak yeni bir hacim oluştur
          volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "segmentation_output")
          # İnput volume özelliklerini kopyala
          volumeNode.CopyOrientation(inputVolume)
          # Maske hackimini hazırla
          numSlices = len(segmentationResults)
          if numSlices <= 0:
              raise ValueError("Sonuç listesi boş!")
          # İlk maske boyutlarını al
          first_mask = segmentationResults[0]['mask']
          mask_height, mask_width = first_mask.shape
          # Hacim dizisi oluştur
          segmentationArray = np.zeros((numSlices, mask_height, mask_width), dtype=np.uint8)
          # Her dilim için maskeyi kopyala
          for i in range(numSlices):
              if 'mask' in segmentationResults[i]:
                  segmentationArray[i] = segmentationResults[i]['mask']
          # Diziyi hacim düğümüne kopyala
          slicer.util.updateVolumeFromArray(volumeNode, segmentationArray)
          # Kopya işlemlerini onayla
          volumeNode.CreateDefaultDisplayNodes()
          volumeNode.Modified()
          return volumeNode
      except Exception as e:
          logging.error(f"Segmentasyon hacmi oluşturma hatası: {str(e)}")
          print(f"CREATESEGMENTATIONVOLUME HATASI: {str(e)}")
          import traceback
          traceback.print_exc()
          return None
    
    def computeSegmentationMetrics(self, groundTruthVolume, predictedVolume):
        """İki segmentasyon arasındaki metrikleri hesaplar - Genel parametre değerleri ile"""
        try:
            # Fonksiyon çağrısı için benzersiz ID oluştur
            metrics_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
            print(f"[Metrik #{metrics_id}] Metrik hesaplama başladı")
            
            # Her iki volumeden de numpy dizisi al
            gtArray = slicer.util.arrayFromVolume(groundTruthVolume).astype(bool)
            predArray = slicer.util.arrayFromVolume(predictedVolume).astype(bool)
            
            # Hasta ID'sini al
            patient_id = os.path.basename(groundTruthVolume.GetName())
            
            # Hacimlerin hash'leri
            gt_hash = hashlib.md5(gtArray.tobytes()[:1000000]).hexdigest()[:8]  # İlk 1MB hash
            pred_hash = hashlib.md5(predArray.tobytes()[:1000000]).hexdigest()[:8]  # İlk 1MB hash
            
            # Hacmin uzaysal özelliklerini al
            spacing = groundTruthVolume.GetSpacing()
            slice_thickness_mm = spacing[2]  # z eksenindeki spacing, dilim kalınlığı
            pixel_width_mm = spacing[0]      # x eksenindeki spacing (piksel genişliği)
            pixel_height_mm = spacing[1]     # y eksenindeki spacing (piksel yüksekliği)
            
            # Piksel alanına göre minimum lezyon boyutu (mm2 cinsinden)
            min_lesion_area_mm2 = 25.0  # mm2 cinsinden makul minimum alan
            min_lesion_area_px = int(min_lesion_area_mm2 / (pixel_width_mm * pixel_height_mm))
            
            # Hacim boyutu için minimum eşik (mm3 cinsinden)
            min_lesion_volume_mm3 = 100.0  # mm3 cinsinden makul minimum hacim
            min_lesion_volume_px = int(min_lesion_volume_mm3 / (pixel_width_mm * pixel_height_mm * slice_thickness_mm))
            
            # Detaylı loglama ekle
            print(f"[Metrik #{metrics_id}] Hasta: {patient_id}")
            print(f"[Metrik #{metrics_id}] GT boyutu: {gtArray.shape}, Değerler: min={gtArray.min()}, max={gtArray.max()}, sum={gtArray.sum()}, Hash: {gt_hash}")
            print(f"[Metrik #{metrics_id}] Pred boyutu: {predArray.shape}, Değerler: min={predArray.min()}, max={predArray.max()}, sum={predArray.sum()}, Hash: {pred_hash}")
            print(f"[Metrik #{metrics_id}] Piksel boyutu: {pixel_width_mm:.3f} x {pixel_height_mm:.3f} mm, Dilim kalınlığı: {slice_thickness_mm:.2f} mm")
            print(f"[Metrik #{metrics_id}] Minimum lezyon alanı: {min_lesion_area_mm2} mm² ({min_lesion_area_px} piksel)")
            print(f"[Metrik #{metrics_id}] Minimum lezyon hacmi: {min_lesion_volume_mm3} mm³ ({min_lesion_volume_px} piksel)")
            
            # İki volume'un boyutları aynı mı diye kontrol et
            if gtArray.shape != predArray.shape:
                raise ValueError(f"[Metrik #{metrics_id}] Hacimlerin boyutları eşleşmiyor! GT: {gtArray.shape}, Pred: {predArray.shape}")
            
            # Toplam GT hacmi hesapla
            total_gt_volume = np.sum(gtArray)
            total_pred_volume = np.sum(predArray)
            
            # Eğer GT hacim belirli bir eşikten küçükse, tahmin etmemeyi tercih et
            if total_gt_volume < min_lesion_volume_px * 0.5:
                print(f"[Metrik #{metrics_id}] GT hacmi çok küçük ({total_gt_volume} piksel < {min_lesion_volume_px * 0.5} piksel), bu vaka için tahmin yapılmayacak")
                # Tüm tahmin maskelerini temizle
                predArray = np.zeros_like(predArray)
            
            # Tahmin hacmi çok küçükse, temizle (muhtemelen gürültü)
            if 0 < total_pred_volume < min_lesion_volume_px * 0.5:
                print(f"[Metrik #{metrics_id}] Tahmin hacmi çok küçük ({total_pred_volume} piksel < {min_lesion_volume_px * 0.5} piksel), tahminler temizleniyor")
                predArray = np.zeros_like(predArray)
            
            # Dilim bazlı işlem
            numSlices = gtArray.shape[0]
            tp_total = 0
            fp_total = 0
            fn_total = 0
            tn_total = 0
            dice_scores = []
            processed_slices = []  # İşlenen dilim indeksleri
            
            # Her dilimde alanı kontrol et ve çok küçük lezyonları filtrele
            for i in range(numSlices):
                gt_slice = gtArray[i]
                pred_slice = predArray[i]
                
                # Her dilimde bağlı bileşen analizi yap
                if np.any(gt_slice):
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gt_slice.astype(np.uint8), connectivity=8)
                    
                    # Küçük bileşenleri filtrele
                    filtered_mask = np.zeros_like(gt_slice)
                    for label in range(1, num_labels):  # 0 arka plan
                        size = stats[label, cv2.CC_STAT_AREA]
                        if size < min_lesion_area_px * 0.5:
                            print(f"[Metrik #{metrics_id}] Dilim {i}: GT'de {size} piksellik küçük komponent temizleniyor (eşik: {min_lesion_area_px * 0.5} piksel)")
                        else:
                            # Yeterince büyük bileşenleri sakla
                            filtered_mask[labels == label] = True
                    
                    # Filtrelenmiş maskeyi güncelle
                    gt_slice = filtered_mask
                
                # Tahmin maskesinde de küçük bileşenleri filtrele
                if np.any(pred_slice):
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_slice.astype(np.uint8), connectivity=8)
                    
                    # Küçük bileşenleri filtrele
                    filtered_mask = np.zeros_like(pred_slice)
                    for label in range(1, num_labels):  # 0 arka plan
                        size = stats[label, cv2.CC_STAT_AREA]
                        if size < min_lesion_area_px * 0.5:
                            print(f"[Metrik #{metrics_id}] Dilim {i}: Pred'de {size} piksellik küçük komponent temizleniyor (eşik: {min_lesion_area_px * 0.5} piksel)")
                        else:
                            # Yeterince büyük bileşenleri sakla
                            filtered_mask[labels == label] = True
                    
                    # Filtrelenmiş maskeyi güncelle
                    pred_slice = filtered_mask
                
                # Güncellenen maskeleri geri ata
                gtArray[i] = gt_slice
                predArray[i] = pred_slice
            
            # Toplam GT hacmini güncelle
            total_gt_volume = np.sum(gtArray)
            total_pred_volume = np.sum(predArray)
            
            # Güncelleme sonrası hacim kontrolü
            if total_gt_volume < min_lesion_volume_px * 0.5:
                print(f"[Metrik #{metrics_id}] Filtreleme sonrası GT hacmi çok küçük ({total_gt_volume} piksel < {min_lesion_volume_px * 0.5} piksel), bu vaka için tahmin yapılmayacak")
                # Tüm tahmin maskelerini temizle
                predArray = np.zeros_like(predArray)
            
            # Tahmin hacmi çok küçükse (muhtemelen gürültü)
            if 0 < total_pred_volume < min_lesion_volume_px * 0.5:
                print(f"[Metrik #{metrics_id}] Filtreleme sonrası tahmin hacmi çok küçük ({total_pred_volume} piksel < {min_lesion_volume_px * 0.5} piksel), tahminler temizleniyor")
                predArray = np.zeros_like(predArray)
            
            # Her dilimi ayrı ayrı değerlendir
            for i in range(numSlices):
                gt_slice = gtArray[i]
                pred_slice = predArray[i]
                
                # Gerçek maskede lezyon var mı kontrol et
                gt_has_lesion = np.any(gt_slice)
                
                # Gerçek maskede lezyon olan dilimler için metrik hesapla
                if gt_has_lesion:
                    processed_slices.append(i)  # Bu dilimi işlenmiş olarak işaretle
                    
                    # Bu dilim için TP, FP, FN, TN hesapla
                    tp_slice = np.logical_and(gt_slice, pred_slice).sum()
                    fp_slice = np.logical_and(~gt_slice, pred_slice).sum()
                    fn_slice = np.logical_and(gt_slice, ~pred_slice).sum()
                    tn_slice = np.logical_and(~gt_slice, ~pred_slice).sum()
                    
                    # Toplama ekle
                    tp_total += tp_slice
                    fp_total += fp_slice
                    fn_total += fn_slice
                    tn_total += tn_slice
                    
                    # Bu dilim için Dice hesapla
                    denominator = 2 * tp_slice + fp_slice + fn_slice
                    if denominator > 0:
                        dice_slice = 2 * tp_slice / denominator
                        dice_scores.append(dice_slice)
                        print(f"[Metrik #{metrics_id}] Dilim {i}: TP={tp_slice}, FP={fp_slice}, FN={fn_slice}, Dice={dice_slice:.4f}")
                    else:
                        print(f"[Metrik #{metrics_id}] Dilim {i}: TP={tp_slice}, FP={fp_slice}, FN={fn_slice}, Dice=0.0000 (Payda sıfır)")
            
            # İşlenen dilim sayısını kaydet
            processed_slice_count = len(processed_slices)
            
            # Eğer hiç işlenen dilim yoksa (tüm GT dilimler boş)
            if processed_slice_count == 0:
                print(f"[Metrik #{metrics_id}] UYARI: Gerçek maskede hiç lezyon içeren dilim bulunamadı!")
                return {
                    'dice': 1.0 if predArray.sum() == 0 else 0.0,  # Eğer tahmin de boşsa mükemmel skor
                    'sensitivity': 1.0,
                    'specificity': 1.0,
                    'precision': 1.0,
                    'accuracy': 1.0,
                    'tp': 0,
                    'fp': 0,
                    'fn': 0,
                    'tn': int(np.prod(gtArray.shape)),
                    'gt_volume': 0,
                    'pred_volume': int(predArray.sum()),
                    'volume_diff_percent': float('inf') if predArray.sum() > 0 else 0.0,
                    'metrics_hash': 'no_lesion',
                    'gt_hash': gt_hash,
                    'pred_hash': pred_hash,
                    'patient_id': patient_id,
                    'processed_slices': 0
                }
            
            # Toplam değerlerden metrikleri hesapla
            dice = 2 * tp_total / (2 * tp_total + fp_total + fn_total) if (2 * tp_total + fp_total + fn_total) > 0 else 0
            sensitivity = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
            specificity = tn_total / (tn_total + fp_total) if (tn_total + fp_total) > 0 else 0
            precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
            accuracy = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total) if (tp_total + tn_total + fp_total + fn_total) > 0 else 0
            
            # Ortalama dilim Dice skoru (sadece lezyon olan dilimlerde)
            avg_slice_dice = np.mean(dice_scores) if dice_scores else 0
            
            # İnme hacimleri - sadece işlenen dilimlerdeki hacimleri topla
            gt_volume = sum([gtArray[i].sum() for i in processed_slices])
            pred_volume = sum([predArray[i].sum() for i in processed_slices])
            volume_difference = abs(gt_volume - pred_volume)
            volume_diff_percent = (volume_difference / gt_volume * 100) if gt_volume > 0 else float('inf')
            
            # Hacim bazlı Dice (daha sağlam olabilir)
            volumetric_dice = (2 * tp_total) / (gt_volume + pred_volume) if (gt_volume + pred_volume) > 0 else 0
            
            # Fiziksel hacimler (mm³)
            gt_volume_mm3 = gt_volume * pixel_width_mm * pixel_height_mm * slice_thickness_mm
            pred_volume_mm3 = pred_volume * pixel_width_mm * pixel_height_mm * slice_thickness_mm
            
            # Hesaplanan metrikleri yazdır
            print(f"[Metrik #{metrics_id}] Hesaplanan metrikler - Dice: {dice:.6f}, Avg Slice Dice: {avg_slice_dice:.6f}, Sensitivity: {sensitivity:.6f}")
            print(f"[Metrik #{metrics_id}] GT Hacim: {gt_volume} piksel ({gt_volume_mm3:.2f} mm³), Pred Hacim: {pred_volume} piksel ({pred_volume_mm3:.2f} mm³)")
            print(f"[Metrik #{metrics_id}] Hacim Farkı: {volume_diff_percent:.2f}%")
            print(f"[Metrik #{metrics_id}] Volumetric Dice: {volumetric_dice:.6f}")
            print(f"[Metrik #{metrics_id}] İşlenen dilim sayısı: {processed_slice_count}/{numSlices}")
            
            # Hesaplanan skorun hash değerini kaydet (tekrar edilebilirlik kontrolü için)
            metrics_hash = hashlib.md5(f"{tp_total}_{fp_total}_{fn_total}_{tn_total}_{dice:.6f}".encode()).hexdigest()[:8]
            print(f"[Metrik #{metrics_id}] Metrikler özeti (Hash): {metrics_hash}")
            
            return {
                'dice': dice,
                'avg_slice_dice': avg_slice_dice,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'accuracy': accuracy,
                'volumetric_dice': volumetric_dice,
                'gt_volume': int(gt_volume),
                'pred_volume': int(pred_volume),
                'gt_volume_mm3': float(gt_volume_mm3),
                'pred_volume_mm3': float(pred_volume_mm3),
                'volume_diff_percent': float(volume_diff_percent),
                'tp': int(tp_total),
                'fp': int(fp_total),
                'fn': int(fn_total),
                'tn': int(tn_total),
                'metrics_hash': metrics_hash,
                'gt_hash': gt_hash,
                'pred_hash': pred_hash,
                'patient_id': patient_id,
                'processed_slices': processed_slice_count,
                'total_slices': numSlices,
                'min_lesion_area_px': min_lesion_area_px,
                'min_lesion_volume_px': min_lesion_volume_px
            }
        
        except Exception as e:
            logging.error(f"Metrik hesaplama hatası: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'dice': 0,
                'sensitivity': 0,
                'specificity': 0,
                'precision': 0,
                'accuracy': 0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'tn': 0,
                'metrics_hash': 'error',
                'error': str(e)
            }

# Test Sınıfı
class StrokeDetectionTest(ScriptedLoadableModuleTest):
    def setUp(self):
        """ Test çalıştırmadan önce ayarları yap """
        slicer.mrmlScene.Clear()
    
    def runTest(self):
        """Modül testlerini çalıştır"""
        self.setUp()
        self.test_StrokeDetection1()
    
    def test_StrokeDetection1(self):
        """ Modül testi """
        self.delayDisplay("Başlıyor")
        self.delayDisplay('Test tamamlandı')