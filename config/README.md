# NOTE For Multitask Classify
## STRUCT DATASET
```
├── DATASET
│   ├── train
│   │   ├── ear
│   │   │   ├── benh1
│   │   │   └── benh2
│   │   ├── nose
│   │   │   ├── benh1
│   │   │   └── benh2
│   │   ├── throat
│   │   │   ├── benh1
│   │   │   └── benh2
│   ├── val
│   │   ├── ear
│   │   │   ├── benh1
│   │   │   └── benh2
│   │   ├── nose
│   │   │   ├── benh1
│   │   │   └── benh2
│   │   ├── throat
│   │   │   ├── benh1
└── └── └── └── benh2
```

## HOW TO CONFIG
```
root_dataset: "/DATASET"          <- Đường dẫn đến thư mục dataset như cấu trúc ở trên
output_path: "/Output"          <- Đường dẫn đầu ra để lưu kết quả
batch_size: 64          
model_pretrained: "google/vit-base-patch16-224" 
epochs: 10
learning_rate: 0.001
```