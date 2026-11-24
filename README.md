# CPE342 Kaggle: Player Intelligence System

ซอร์สโค้ดสุดท้ายที่ใช้สร้างไฟล์คำตอบส่ง Kaggle ของรายวิชา CPE342 (Machine Learning) ครบทั้ง 5 task
แต่ละ task อยู่ในโฟลเดอร์ของตัวเองและมีโน้ตบุ๊กหลักที่รันได้ตั้งแต่อ่านข้อมูลดิบจนได้ไฟล์คำตอบ ส่วนโค้ดที่ใช้ทดลองระหว่างทางแยกไว้ในโฟลเดอร์ `experiments/`

## โครงสร้าง repo

```text
CPE342-KAGGLE/
├── task1/               Task 1 Cheater Detection (binary, macro F2)
│   └── task1.ipynb
├── task2/               Task 2 Player Segmentation (4 classes, macro F1)
│   ├── task2.ipynb
│   └── features.py, models.py, naive_bayes.py, ensemble.py   โมดูลที่โน้ตบุ๊กเรียกใช้
├── task3/               Task 3 Spending Prediction (regression, 1/(1+NMAE))
│   ├── task3.ipynb
│   └── t3_features.py, t3_models.py                        โมดูลที่โน้ตบุ๊กเรียกใช้
├── task4/               Task 4 Game Title Detection (images, macro F1)
│   ├── task4.ipynb      อธิบายวิธี ตรวจคะแนน รวมโมเดล และสร้างคำตอบ
│   ├── train.py         fine-tune โมเดล pretrained (ต้องใช้ GPU)
│   ├── train_all.sh     คำสั่งฝึกทั้ง 3 โมเดล x 5 fold
│   ├── pack_preds.py    รวมผลราย fold เป็น preds/<model>.npz
│   └── preds/           ความน่าจะเป็น OOF และ test ของโมเดลที่ฝึกแล้ว
├── task5/               Task 5 Account Security Monitoring (anomaly detection, F3)
│   └── task5.ipynb
├── experiments/         โค้ดทดลองทั้งหมดที่ไม่ได้อยู่ใน pipeline สุดท้าย (แยกตาม task)
├── make_submission.py   รวมคำตอบ 5 task เป็น submission.csv
├── requirements.txt
└── README.md
```

แต่ละโน้ตบุ๊กบันทึกผลลัพธ์ (ตาราง กราฟ คะแนน) ไว้ในไฟล์แล้ว จึงเปิดอ่านได้ทันทีโดยไม่ต้องรันใหม่ และเขียนคำตอบของ task นั้นไว้ที่ `taskN/output/taskN_pred.csv`

## สรุปวิธีและคะแนน

| Task | วิธีสุดท้าย | คะแนน validation |
|---|---|---|
| 1 | LightGBM + XGBoost + CatBoost อย่างละ 4 seed (12 โมเดล) เฉลี่ยความน่าจะเป็น, กฎ box จากช่วงค่าของ cheater ใน train, threshold 0.37 จาก OOF | macro F2 (OOF) ≈ 0.781 |
| 2 | GBDT 3 ตัว + MLP แล้ว log-mix กับ naive Bayes ที่จำลอง label noise ด้วย EM, ปรับน้ำหนักคลาสบน OOF | macro F1 (OOF) ≈ 0.757 |
| 3 | hurdle model: ความน่าจะเป็นที่จะใช้จ่าย + log-normal median + L1 ratio model รวมด้วยน้ำหนักแยกกลุ่ม whale | 1/(1+NMAE) (OOF) ≈ 0.794 |
| 4 | EVA02 Large + ConvNeXt V2 Large (pretrained) fine-tune ด้วย generic robustness augmentation อย่างละ 5 fold แล้วเฉลี่ยน้ำหนักเท่ากัน | macro F1 (OOF) 0.995 บนภาพสะอาด, 0.993 บน degraded proxy |
| 5 | ค้นหากลุ่มผิดปกติ 7 กลุ่มที่แยกตัวด้วยช่องว่างของ histogram + spike ช่วงที่ 4 + posterior ของกลุ่มที่ 7 | ไม่มี label (ประเมินจาก leaderboard ได้ F3 ≈ 0.995) |

คะแนน public leaderboard ของชุดคำตอบนี้คือ **0.84253** (เฉลี่ย 5 task)

## การเตรียมข้อมูล

ดาวน์โหลดข้อมูลจากหน้าแข่งขัน Kaggle แล้ววางไว้ในโฟลเดอร์ `data/` ที่ root ของ repo (โฟลเดอร์นี้ไม่ได้ commit ขึ้น GitHub)

```text
data/
├── 01_binary_classification/      train.csv, test.csv
├── 02_multiclass_classification/  train.csv, test.csv
├── 03_regression/                 train.csv, test.csv
├── 04_image_classification/       train.csv, test.csv, train/*.jpg, test/*.jpg
└── 05_anomaly_detection/          test.csv
```

หากเก็บข้อมูลไว้ที่อื่น ให้ตั้ง environment variable `CPE342_DATA` ชี้ไปยังโฟลเดอร์นั้นก่อนเปิด Jupyter

## การติดตั้ง

ทดสอบกับ Python 3.12

```bash
pip install -r requirements.txt
```

Task 4 ส่วนการฝึกโมเดลต้องใช้ PyTorch ที่รองรับ CUDA (ติดตั้งตามคำแนะนำของ pytorch.org ให้ตรงกับรุ่น CUDA ของเครื่อง) ส่วนการสร้างคำตอบจากผลทำนายที่บันทึกไว้ใช้ CPU ได้

## วิธีรัน

1. เปิดโน้ตบุ๊กในโฟลเดอร์ของแต่ละ task แล้วสั่ง Run All (ต้องรันโดยให้ working directory อยู่ในโฟลเดอร์ของ task นั้น) หรือใช้คำสั่ง
   ```bash
   cd task1
   jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 task1.ipynb
   ```
2. เมื่อได้ `output/taskN_pred.csv` ครบทั้ง 5 task ให้รวมเป็นไฟล์ส่ง Kaggle
   ```bash
   python make_submission.py
   ```

เวลารันโดยประมาณบนโน้ตบุ๊ก CPU 18 คอร์

| Task | เวลา | หมายเหตุ |
|---|---|---|
| 1 | ราว 1 ชั่วโมง | ฝึก GBDT 12 ชุด x 5 fold |
| 2 | ราว 1.5 ชั่วโมง | CatBoost ใช้เวลามากที่สุด |
| 3 | ราว 30 นาที | |
| 4 | ไม่ถึง 1 นาที | ใช้ผลทำนายใน `task4/preds/`; การฝึกใหม่ทั้งหมดใช้ GPU ราว 27 ชั่วโมง (ดู `task4/train_all.sh`) |
| 5 | ราว 3 นาที | |

ผลของ Task 1 ถึง 3 ที่ฝึกใหม่อาจต่างจากไฟล์ที่ส่งจริงเล็กน้อยในบางแถวที่อยู่ใกล้ threshold เนื่องจากรุ่นของไลบรารีและการคำนวณแบบขนาน
(Task 2 ตั้งจำนวน thread ของแต่ละโมเดลไว้ในโน้ตบุ๊กให้ตรงกับรอบที่ส่งจริง เพราะ XGBoost ให้ผลต่างกันเล็กน้อยเมื่อเปลี่ยนจำนวน thread)
เมื่อรันครบทุกโน้ตบุ๊กแล้ว `submission.csv` ที่ได้ตรงกับไฟล์ที่ส่ง Kaggle ทุกแถว ยกเว้น Task 3 หนึ่งแถว ซึ่งเปลี่ยนเพราะแก้ให้ค่าเติม missing คำนวณจาก train เท่านั้น

## กติกาที่ปฏิบัติตาม

- ทุกขั้นตอนที่มีการ fit (โมเดล, ค่าเติม missing, encoder, rank, bin edge, threshold, น้ำหนัก blend) ใช้เฉพาะข้อมูล train หรือ OOF ของ train ข้อมูล test ใช้เพื่อทำนายเท่านั้น และไม่มี pseudo-label
- ไม่ใช้ `player_id` เป็นฟีเจอร์
- ไม่ใช้ AutoML ไฮเปอร์พารามิเตอร์กำหนดเองหรือเลือกจาก cross-validation
- ใช้โมเดล pretrained เฉพาะ Task 4
- Task 4 ใช้ augmentation แบบทั่วไปที่ไม่ได้เลือกจากการดูภาพ test และ proxy สำหรับเลือก epoch สร้างจากภาพ train ใน validation fold เท่านั้น
  (augmentation ด้านสีชุดเดิมที่ออกแบบหลังจากเห็นภาพ test ถูกตัดออกและฝึกโมเดลใหม่ทั้งหมด ดูรายละเอียดใน `task4/task4.ipynb`)
- Task 5 ไม่มีข้อมูล train ตามที่โจทย์กำหนด จึงเป็น unsupervised บนข้อมูลที่ให้มาและไม่ใช้ label ใด ๆ

## โฟลเดอร์ experiments

เก็บโค้ดที่ใช้สำรวจข้อมูลและทดลองแนวทางต่าง ๆ ก่อนได้ pipeline สุดท้าย แยกตาม task พร้อมคำอธิบายใน `experiments/README.md`
โค้ดส่วนนี้เก็บไว้เพื่ออ้างอิงที่มาของการตัดสินใจ ไม่จำเป็นต้องรันเพื่อสร้างคำตอบ และบางไฟล์อ้างอิงไฟล์ผลลัพธ์ระหว่างทางที่ไม่ได้อยู่ใน repo
