# إعداد بيانات Camargo

## 1. تحميل الداتا

حمّل الداتا من:
- http://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al  
- أو Mendeley (3 أجزاء)

## 2. مكان المجلد

ضع المجلد بهذا الشكل:

```
exoskeleton_system_using_biomedical_sensor_data/
├── Data_repository_for_Camargo/
│   ├── AB06/
│   ├── AB07/
│   └── README.txt
└── exoskeleton_pipeline/
```

## 3. التشغيل

```powershell
cd exoskeleton_pipeline
$env:CAMARGO_ROOT = "D:\Downloads\exoskeleton_system_using_biomedical_sensor_data\Data_repository_for_Camargo"
python train.py --max-files 20 --modes treadmill
```

## 4. بدون الداتا (تجربة)

```powershell
python train.py --demo --demo-trials 12
```

هذا **ليس** بديلاً عن Camargo للبحث — فقط للاختبار.
