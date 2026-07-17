
from ultralytics import YOLO
import os
import cv2
import random
import shutil
import torch


def MakeReady(
    Path_weights: str = "runs/detect/train/weights/best.pt",
    Path_img: str ="imagesfortuning/",
    Output_dir: str ="imagesfortuning",
    Conf_thres: float = 0.50):

    try:    
        os.makedirs(Output_dir, exist_ok=True)
    
        model = YOLO(Path_weights)
        supported_ext = {".jpg", ".jpeg", ".png"}
        
        ask = input("Görseller etiketlenmeye başlanıyor. Sonuçları görseller üzerinde görmek ister misiniz? (Y/n)\n").lower()
    
        for img_name in os.listdir(Path_img):
            if not any(img_name.lower().endswith(ext) for ext in supported_ext):
                continue
    
            img_path = os.path.join(Path_img, img_name)
            if ask == 'y': 
                results = model.predict(img_path,save=True,exist_ok=True,show_labels=False,show_conf=False,conf=Conf_thres)[0]
            else:
                results = model.predict(img_path,conf=Conf_thres)[0]
                
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
    
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            txt_path = os.path.join(Output_dir, txt_name)
    
            with open(txt_path, "w") as f:
                for box in results.boxes:
                    cls = int(box.cls[0])
    
                    if cls != 0:
                        continue
                    
                    xywh = box.xywhn[0].tolist()
                    x_center = xywh[0]
                    y_center = xywh[1]
                    bw = xywh[2]
                    bh = xywh[3]
    
                    f.write(f"0 {x_center} {y_center} {bw} {bh}\n")
        if ask == 'y':
            print("✅ Görsel işleme başarılı. Tüm görseller etiketlendi ve etiketli görseller \"predict\" içderisine kaydedildi.\n")
        else:   
            print("✅ Görsel işleme başarılı. Tüm görseller etiketlendi.\n")
        
        ask = input("Görselleri eğitime uygun ayırmak ister misin? (Y/n)\n").lower()
        if ask == 'y':
            Split_Dataset()
            exit("⭕ Program tamamlandı. Kapatılıyor...\n")
        else:
            exit("⭕ Program tamamlandı. Kapatılıyor...\n")
            
    except Exception as e:
        exit(f"❌ Hata oluştu. Görseller etiketlenemedi.\nError: {e}")   


def Split_Dataset(
    Path_img: str = "imagesfortuning/",
    Output_dir: str = "dataset_tuning",
    split_ratio: float = 0.8):

    try:
        img_files = [f for f in os.listdir(Path_img) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(img_files)

        split_point = int(len(img_files) * split_ratio)
        train_files = img_files[:split_point]
        valid_files = img_files[split_point:]

        for split in ["train", "valid"]:
            os.makedirs(os.path.join(Output_dir, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(Output_dir, split, "labels"), exist_ok=True)

        def copy_files(file_list, split_name):
            for f in file_list:
                base = os.path.splitext(f)[0]
                img_src = os.path.join(Path_img, f)
                label_src = os.path.join(Path_img, base + ".txt")

                img_dst = os.path.join(Output_dir, split_name, "images", f)
                label_dst = os.path.join(Output_dir, split_name, "labels", base + ".txt")

                shutil.copy(img_src, img_dst)

                if os.path.exists(label_src):
                    shutil.copy(label_src, label_dst)
                else:
                    print(f"⚠️ UYARI: Etiket dosyası bulunamadı. {f}\n")

        copy_files(train_files, "train")
        copy_files(valid_files, "valid")

        print("✅ Dataset başarıyla eğitime uygun ayrıldı.\n")
        
        ask = input("Eğitim için split edilen veriler silinsin mi? (Y/n)\nSilinen veriler dataset_tuning klasörü içinde korunacaktır. Sadece imagesfortuning klasöründen kaldırılacaktır.\n").lower()
        if ask == 'y':
            Clear_imagesfortuning()
            
        ask = input("Fine_Tuning işlemine başlatılsın mı? (Y/n)\n").lower()
        if ask == 'y':
            YoloFineTuning()
            exit()
        else:
            exit("⭕ Program tamamlandı. Kapatılıyor...\n")

    except Exception as e:
        exit(f"❌ Hata oluştu. Dataset eğitime uygun ayrılamadı.\nError details: {e}\n")        
                
    
def YoloFineTuning(Path_weights: str = "runs/detect/train/weights/best.pt"):
    try:  
        model = YOLO(Path_weights)
        if(torch.cuda.is_available()):
            print(f"✅ CUDA mevcut, {torch.cuda.get_device_name(0)} kullanılıyor. \n")
            model.train(
                data="data_tuning.yaml",
                epochs=25,        
                imgsz=640,
                batch=16,
                lr0=0.001)
            
        else:
            print("⚠️ UYARI: CUDA mevcut değil.\n")
            model = YOLO(Path_weights)
            model.train(
                data="data_tuning.yaml",
                epochs=25,        
                imgsz=320,
                batch=8,
                lr0=0.001)
            
        print("✅ Model iyileştirme başarılı.\n")
        ask = input("Eğitime dahile edilen veriler silinsin mi? (Y/n)\n").lower()
        if ask == 'y':
            Clear_dataset_tuning()
            exit("⭕ Program tamamlandı. Kapatılıyor...\n")
        else:
            exit("⭕ Program tamamlandı. Kapatılıyor...\n")
        
    except Exception as e:
        exit(f"❌ Model iyileştirme başarısız oldu.\nError: {e}")
        

def Clear_imagesfortuning():
    folder_path = "imagesfortuning"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)
    print("⚠️ UYARI: imagesfortuning dosyası temizlendi. Tüm veriler dataset_tuning içerisinde mevcuttur.\n")
    
    
def Clear_dataset_tuning():
    folder_path = "dataset_tuning"
    shutil.rmtree(folder_path)
    print("⚠️ UYARI: dataset_tuning dosyası temizlendi.\n")    
        

if __name__ == '__main__':
    MakeReady()