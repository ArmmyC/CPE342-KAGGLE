# Experiments

โค้ดที่ใช้สำรวจข้อมูลและทดลองแนวทางต่าง ๆ ก่อนได้ pipeline สุดท้ายใน `task1/` ถึง `task5/`
เก็บไว้เพื่อแสดงที่มาของการตัดสินใจ ไม่จำเป็นต้องรันเพื่อสร้างคำตอบ สคริปต์ส่วนใหญ่อ่านหรือเขียนไฟล์ผลลัพธ์ระหว่างทาง (`.npy`, `preds/`, `out/`) ที่ไม่ได้รวมอยู่ใน repo
และ path ในเครื่องที่เคยใช้ถูกแทนด้วย `<project_root>/`

## การจัดโฟลเดอร์

แต่ละ task แบ่งตามรอบการทำงาน

| รอบ | ช่วงเวลา | เนื้อหา |
|---|---|---|
| `round1_first_data` | ข้อมูลชุดแรก | baseline, GBDT, การสร้างฟีเจอร์, โมเดลโครงสร้างของ Task 3, วิธี anomaly detection แบบทั่วไปของ Task 5 |
| `round2_new_data` | หลังผู้สอนปล่อยข้อมูลชุดใหม่ | สำรวจโครงสร้างของ generator อย่างละเอียด ค้นหากลุ่มผิดปกติของ Task 5 ตรวจ residual ของ Task 1 ถึง 3 |
| `round3_push` | ช่วงปรับปรุงรอบสุดท้าย | สร้าง Task 2 และ Task 3 ใหม่ให้ทุกขั้นตอน fit บน train เท่านั้น, ทดลอง nested CV, naive Bayes ที่จำลอง label noise |

`common/` เก็บ helper ที่หลายสคริปต์ใช้ร่วมกัน และ `eda/` เก็บสคริปต์สำรวจข้อมูลทั่วไป

## สิ่งที่ทดลองและผลที่ได้ (สรุป)

### Task 1 (cheater detection)
- GBDT (LightGBM, XGBoost, CatBoost) ให้ผลดีที่สุด การรวมหลาย seed เพิ่ม macro F2 จาก 0.7790 เป็น 0.7806
- พบว่า label ในส่วนหางของบางฟีเจอร์ถูกกำหนดด้วยกฎ (เช่น `reports_received`, `crosshair_placement`, `account_age_days`) จึงเพิ่มกฎ box ที่ได้จากช่วงค่าของ cheater ใน train
- ลอง QDA, LDA, copula naive Bayes, GAM, neural network, การสร้างฟีเจอร์คู่ และการใช้แถวที่ไม่มี label ทั้งหมดไม่ดีกว่า GBDT (`t1_hunt*.py`, `t1_qda.py`, `t1_cop.py`, `push/exp_lgb.py`)
- การวิเคราะห์ residual ทุกคู่ฟีเจอร์ (`t1_resid.py`) ไม่พบโครงสร้างเหลือ จึงสรุปว่า Task 1 อยู่ใกล้ขีดจำกัดของสัญญาณรบกวนแล้ว (ราว 0.78)

### Task 2 (player segmentation)
- ฟีเจอร์แทบเป็นอิสระต่อกันเมื่อรู้คลาส naive Bayes แบบ histogram จึงทำได้ใกล้เคียง GBDT (`t2_nb.py`, `nb.py`)
- segment มีโครงสร้างเป็นสองปัจจัยแบบเกือบ binary และมี label noise ราว 1% จึงพัฒนา naive Bayes ที่ประมาณ matrix การสลับ label ด้วย EM (`round3_push/nb2.py`, `e5.py`)
- การรวม GBDT, MLP และ EM naive Bayes แบบ log-mix และการปรับน้ำหนักคลาสบน OOF ให้ผลดีที่สุด (`final_fix.py`)

### Task 3 (spending prediction)
- พบสูตรที่แน่นอนระหว่างคอลัมน์ (เช่น ยอดใช้จ่ายย้อนหลังกับจำนวนธุรกรรม) จึงใช้เติมค่าที่หายไปแบบ row-wise (`t3_struct.py`, `common3.py`)
- เปรียบเทียบ hurdle model, log-normal median และ L1 ratio model แล้วรวมด้วยน้ำหนักแยกกลุ่ม whale ด้วย nested CV (`exp1` ถึง `exp4`)

### Task 4 (game title detection)
- ทดลอง embedding จากโมเดล pretrained แบบ frozen (`t4_embed.py`) ซึ่งด้อยกว่าการ fine-tune ทั้งโมเดล
- ทดลองจัดกลุ่มภาพที่คล้ายกันใน train ด้วย kNN บน embedding (`t4_grp.py`) เพื่อตรวจว่ามีภาพซ้ำหรือภาพต่อเนื่องที่ทำให้ validation สูงเกินจริงหรือไม่
- รอบแรกของการ fine-tune ใช้ augmentation ด้านสีที่รุนแรง (invert, หมุน hue, สลับช่องสี) ซึ่งออกแบบหลังจากเห็นภาพ test สคริปต์ชุดนั้นไม่ได้รวมไว้ใน repo
  เพราะขัดกับกติกา และโมเดลทั้งหมดถูกฝึกใหม่ด้วย augmentation แบบทั่วไป (AUG 5) ใน `task4/train.py`
- ผลบน public leaderboard ของโมเดลชุดใหม่: ConvNeXt V2 Large + EVA02 Large อย่างละ 2 fold ได้ 0.84127, เพิ่ม ConvNeXt V2 เป็น 5 fold ได้ 0.84173 และครบ 5 fold ทั้งสองโมเดลได้ 0.84253
- EVA02 Base (AUG 5) ฝึกครบ 5 fold แต่ไม่ช่วย blend บน OOF จึงไม่ใช้ ส่วน ConvNeXt V2 Large กับ augmentation ที่แรงขึ้น (AUG 6) ฝึกได้เพียง 1 fold ก่อนการแข่งขันจบ

### Task 5 (anomaly detection)
- วิธีทั่วไป (Isolation Forest, local z-score, drift score, synthetic injection) ให้คะแนนรวมที่แยกกลุ่มผิดปกติได้ไม่ครบ (`round1_first_data`)
- การดู histogram และมุมของคู่ฟีเจอร์พบกลุ่มผิดปกติที่แยกตัวชัดเจน 7 กลุ่ม (`t5_gap*.py`, `t5_pairs*.py`, `t5_g7*.py`, `t5_spikes*.py`) ซึ่งกลายเป็นวิธีสุดท้าย
- การสแกนเพิ่มเติม (`t5_g8.py`) ไม่พบกลุ่มที่ 8
