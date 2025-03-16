##  **Projekt: Zpracování obrazů a testování klasifikace**  

Tento projekt se zaměřuje na předzpracování obrazů, segmentaci a následné testování klasifikace pomocí neuronových sítí. Byly implementovány různé metody odstranění šumu, segmentace a klasifikace, přičemž výsledky byly vyhodnoceny pomocí standardních metrik.  

##  **Struktura projektu**  

📂 **preprocessing** - Implementace metod odstranění šumu  
- `gauss_filter.py` - Aplikace Gaussova filtru pro odstranění šumu  
- `median_filter.py` - Použití mediánového filtru  
- `cnn_denoise_saltPepperNois.py` - Použití neuronové sítě k odstranění šumu Sůl a pepř
- `cnn_denoise_gaussNois.py` - Použití neuronové sítě k odstranění Gaussova šumu 

📂 **segmentation** - Implementace segmentačních metod  
- `watershed.py` - Segmentace pomocí metody vododělové transformace  
- `unet_segmentation.py` - Segmentace pomocí neuronové sítě U-Net  

📂 **metrics** - Výpočet metrik hodnocení  
- `calculate_denoisMetrics.py` - Výpočet metrik pro odstranění šumu (PSNR, SSIM, MSE)  
- `calculate_segmentMetrics.py` - Výpočet metrik pro segmentaci (IoU, Dice, Precision, F1-score)  
- `classification_test.py` - Testování vlivu předzpracování na klasifikaci pomocí ResNet50  

📂 **utils** - Pomocné funkce  
- `image_loader.py` - Funkce pro načítání obrázků  
- `image_saver.py` - Úložení výsledků  

📂 **Model_NN** - Trénování neuronových sítí  

Trénovaný model neuronové sítě ResNet50 je dostupný ke stažení na Google Drive:  
🔗 **[Odkaz na model](https://drive.google.com/drive/folders/1pXBWklBxM1nPzNuzwh6qU9bLhBW55nAo?usp=drive_link)**  

Stáhněte soubory a umístěte je do složky `Model_NN` před spuštěním testů klasifikace nebo segmentace.  

- Obsahuje čtyři skripty pro trénování různých modelů:
- `CNN_training_gausNois.py` - Trénování neuronové sítě k eliminaci Gaussova šumu
- `CNN_training_saltPepper.py` - Trénování neuronové sítě k eliminaci mediánového šumu
- `ResNet50_training.py` - Trénování neuronové sítě k provedení klasifikačního testu
- `unet_training.py` - Trénování neuronové sítě U-net pro segmentaci

**🗂 Popis složky `Datasets`**  

Složka Datasets obsahuje kompletní dataset a modely potřebné k reprodukci experimentů v rámci této práce. Z důvodu velké velikosti dat je celý dataset dostupný ke stažení na Google Drive:  
🔗 **[Odkaz na dataset](https://drive.google.com/drive/folders/1pXBWklBxM1nPzNuzwh6qU9bLhBW55nAo?usp=drive_link)**  

Po stažení datasetu je nutné jej rozbalit a umístit do kořenového adresáře projektu.  

Složka `Datasets` obsahuje všechny soubory potřebné pro experimenty s předzpracováním obrazů, segmentací a klasifikací. Data jsou organizována do několika podadresářů podle jejich účelu:  

### **📁 Původní dataset**  
- **`Cat_original`**, **`Dog_original`** - původní nezpracované obrázky koček a psů před jakýmkoli předzpracováním.  
- **`images`**, **`mask`** - dataset rentgenových snímků plic a odpovídajících segmentačních masek pro experimenty se segmentací.  

### **📁 Normalizovaná data**  
- **`Cat_Normalized`**, **`Dog_Normalized`** - normalizované obrázky koček a psů, které byly převedeny na jednotnou velikost a rozsah hodnot pixelů.  

### **📁 Zašuměné obrázky**  
- **`Cat_GausNois`**, **`Dog_GausNois`** - obrázky s přidaným gaussovským šumem.  
- **`Cat_SaltPepper`**, **`Dog_SaltPepper`** - obrázky s přidaným šumem typu sůl a pepř.  

### **📁 Výsledky odstranění šumu (`Result_Denoising`)**  
Složka obsahuje podadresáře s obrázky po aplikaci různých metod odstranění šumu:  
- **`Cat_CNN_GausDenois`**, **`Dog_CNN_GausDenois`** - odstranění gaussovského šumu pomocí CNN.  
- **`Cat_CNN_SaltPepperDenois`**, **`Dog_CNN_SaltPepperDenois`** - odstranění šumu typu sůl a pepř pomocí CNN.  
- **`Cat_GausDenois`**, **`Dog_GausDenois`** - odstranění gaussovského šumu pomocí Gaussova filtru.  
- **`Cat_MedianDenois`**, **`Dog_MedianDenois`** - odstranění šumu typu sůl a pepř pomocí mediánového filtru.  

### **📁 Výsledky segmentace (`Result_Segmentation`)**  
Složka obsahuje segmentační výstupy dvou metod:  
- **`unet`** - segmentační masky vytvořené pomocí neuronové sítě U-Net.  
- **`region_based`** - segmentační masky vytvořené metodou regionální segmentace (watershed).  

### **📁 Předtrénované neuronové sítě (`.h5` modely)**  
Ve složce se nacházejí čtyři předtrénované modely neuronových sítí:  
- **`denoising_cnn_model_gaussian.h5`** - model CNN pro odstranění gaussovského šumu.  
- **`denoising_cnn_model_saltpepper.h5`** - model CNN pro odstranění šumu typu sůl a pepř.  
- **`unet_model.h5`** - model U-Net pro segmentaci plicních snímků.  
- **`ResNet50.h5`** - model ResNet50 pro klasifikaci obrázků koček a psů.  

### **📁 Výsledky měření metrik (`.csv` soubory)**  
- **`metric_gaussian_vs_cnn.csv`** - porovnání metrik odstranění gaussovského šumu pomocí tradičního filtru a CNN.  
- **`metric_median_vs_cnn.csv`** - porovnání metrik odstranění šumu typu sůl a pepř pomocí mediánového filtru a CNN.  

---  
📌 *Tato složka obsahuje kompletní dataset a modely potřebné k reprodukci experimentů v rámci této práce.*

📜 `main.py` - Hlavní soubor pro ověření syntaktické správnosti všech skriptů  
📜 `requirements.txt` - Seznam všech požadovaných knihoven  
📜 `README.md` - Tento soubor  

## **Použití**  

1. **Instalace závislostí**  
   ```bash
   pip install -r requirements.txt
   ```  

2. **Předzpracování obrazů**  
   Spusťte odpovídající skripty v `preprocessing/`, které aplikují šum na obrázky a následně provádějí odstranění šumu pomocí tradičních filtrů nebo neuronových sítí.  

3. **Segmentace obrazů**  
   Spusťte skripty v `segmentation/` pro segmentaci rentgenových snímků pomocí metody rozvodí (watershed) nebo neuronové sítě U-Net.  

4. **Výpočet metrik**  
   Spusťte odpovídající skripty v `metrics/` pro výpočet metrik kvality odstranění šumu (PSNR, SSIM, MSE) a metrik segmentace (IoU, Dice, Precision, F1-score).  

5. **Testování klasifikace**  
   Spusťte `classification_test.py` v `metrics/` pro testování vlivu předzpracování na klasifikaci pomocí modelu ResNet50.  

6. **Trénování neuronových sítí**  
   - Pokud je potřeba trénovat modely od začátku, spusťte příslušné skripty ve složce `Model_NN/`:  
     - `train_denoising_cnn.py` - trénuje CNN pro odstranění šumu.  
     - `train_unet.py` - trénuje U-Net pro segmentaci rentgenových snímků.  
     - `train_resnet.py` - trénuje ResNet50 pro klasifikaci obrázků koček a psů.  
   - Po dokončení trénování budou modely automaticky uloženy ve formátu `.h5`.  

## **Požadavky**  
- Python 3.8+  
- TensorFlow, OpenCV, NumPy, scikit-learn  
- Grafická karta (doporučeno, ale není nutné)  

## **Autor**  
Tento projekt byl vytvořen jako součást bakalářské práce a slouží k experimentálnímu porovnání metod předzpracování obrazů.
