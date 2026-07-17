from ultralytics import YOLO
import torch

def BrainModelCreat(
        yolomodel: str = "yolov8m.pt", 
        epochs_size: int = 50, 
        img_size: int = 640,  
        batch_size: int = 16, 
        lr0: float = 0.001,
        patience: int = 30
        ):
    try:
        if(torch.cuda.is_available()):
            print(f"✅ CUDA mevcut, {torch.cuda.get_device_name(0)} kullanılıyor. \n")
            model = YOLO(yolomodel)
            model.train(data="data.yaml",epochs=epochs_size,imgsz=img_size,batch=batch_size,lr0=lr0,patience=patience,optimizer='AdamW')
            print("✅ Model kullanım için hazır.\n")
            return model
        else:
            print("⚠️ UYARI: CUDA mevcut değil.\n")
            model = YOLO(yolomodel)
            model.train(data="data.yaml",epochs=epochs_size,imgsz=img_size,batch=batch_size,lr0=lr0,patience=patience,optimizer='AdamW')
        print("✅ Model kullanım için hazır.\n")
        return model
    except Exception as e:
        print("❌ Model eğitilemedi, eğitim başarısız.\n")
        print(f"Error: {e}")
        return 0

if __name__ == "__main__":
    model = BrainModelCreat()
