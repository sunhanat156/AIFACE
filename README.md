# AIFACE — ระบบ AI ตรวจจับใบหน้าและระบุตัวตน

ระบบจดจำใบหน้าผ่านเว็บสำหรับควบคุมการเข้าห้อง / ลงเวลาเข้า-ออก
ใช้ FastAPI + InsightFace (SCRFD ตรวจจับใบหน้า + ArcFace สร้าง face vector 512 มิติ)
พร้อมหน้า Dashboard, ลงทะเบียนผู้ใช้, Check-In, ประวัติการเข้าถึง (ค้นหาตามชื่อ/วันที่) และตั้งค่าระบบ

---

## ดาวน์โหลดและใช้งาน (ไม่ต้องลง Python)

1. ไปที่หน้า **[Releases](../../releases/latest)** แล้วดาวน์โหลดไฟล์ `AIFACE-windows.zip`
2. แตกไฟล์ zip (คลิกขวา → Extract All)
3. เปิดโฟลเดอร์ `AIFACE` แล้วดับเบิลคลิก **`AIFACE.exe`**
4. รอโหลดโมเดล AI สักครู่ เบราว์เซอร์จะเปิดหน้า Dashboard ให้อัตโนมัติ

> ต้องการ Windows 10/11 64-bit และกล้องเว็บแคม
> ถ้า Windows SmartScreen เตือน ให้กด **More info → Run anyway** (โปรแกรมไม่ได้ลงลายเซ็นดิจิทัล)
> ปิดโปรแกรม: ปิดหน้าต่างสีดำ

ข้อมูลทั้งหมด (ฐานข้อมูล `faceguard.db`, `config.json`, รูปใน `captured_faces/`) จะถูกสร้างไว้ในโฟลเดอร์เดียวกับ `AIFACE.exe`
โปรแกรมที่ดาวน์โหลดไปเริ่มต้นแบบว่าง ยังไม่มีใบหน้าใคร — ต้องลงทะเบียนเองที่เมนู **Users**

---

## วิธีทดสอบ

1. หน้า **Dashboard** จะเห็นภาพจากกล้องสด (ถ้าขึ้น `NO CAMERA` กดปุ่ม 🔄 หรือเลือกกล้องอื่น)
2. เมนู **Users** → **Register New User** → พิมพ์ชื่อ → **Auto Capture ×5** → **Register**
3. กลับไปหน้า **Dashboard** แล้วยืนหน้ากล้อง ระบบจะสแกนให้อัตโนมัติเมื่อเจอใบหน้า
   - กรอบ **เขียว** + ชื่อ = รู้จัก (Access Granted)
   - กรอบ **แดง** + Unknown = ไม่รู้จัก (Access Denied)
4. เมนู **Access Logs** ดูประวัติพร้อมภาพ ค้นหาตามชื่อ / ช่วงวันที่ / แผนกได้

---

## การทำงานโดยย่อ

| ขั้น | ทำอะไร | ใช้ |
|---|---|---|
| 1 | เช็กว่าภาพเปลี่ยน (มีการเคลื่อนไหว) | CPU ~0.1 ms |
| 2 | มีการเคลื่อนไหว → เช็กว่ามีใบหน้าไหม (SCRFD 256px) | CPU ~13 ms |
| 3 | เจอใบหน้า → ระบุตัวตน (SCRFD 640px + ArcFace) ยืนยันหลายเฟรมก่อนบันทึก | GPU (DirectML) |

ถ้าไม่มีใครอยู่หน้ากล้อง GPU จะไม่ทำงาน และภาพพรีวิวลดเหลือ 10 fps
คนเดิมที่ยืนอยู่จะถูกบันทึกครั้งเดียวต่อการเข้ามาหนึ่งครั้ง

---

## รันจากซอร์สโค้ด (สำหรับนักพัฒนา)

ต้องมี Python 3.12

```
pip install -r requirements.txt
python launcher.py
```

หรือดับเบิลคลิก `START_AIFACE.bat` (ครั้งแรก InsightFace จะดาวน์โหลดโมเดล buffalo_l อัตโนมัติ)

### สร้าง AIFACE.exe เอง

```
pip install pyinstaller
python -m PyInstaller AIFACE.spec --noconfirm
```

ผลลัพธ์อยู่ที่ `dist/AIFACE/`

---

## การตั้งค่า (เมนู Settings)

| ค่า | ความหมาย |
|---|---|
| **Tolerance** | ค่าน้อย = เข้มงวดขึ้น ค่าแนะนำ **0.40–0.45** |
| **Camera Index** | เลขกล้อง เริ่มที่ `0` ถ้ามีหลายกล้องลอง `1`, `2` |

ค่าที่บันทึกมีผลทันที เก็บไว้ใน `config.json`

---

## โครงสร้างโปรเจกต์

```
AIFACE/
├─ launcher.py          ← ตัวเปิดระบบ (หาพอร์ตว่าง, เปิดเบราว์เซอร์)
├─ AIFACE.spec          ← สูตร build เป็น .exe
├─ START_AIFACE.bat     ← เปิดจากซอร์ส
├─ backend/
│  ├─ main.py           ← FastAPI app
│  ├─ paths.py          ← ตำแหน่งไฟล์ข้อมูล (ซอร์ส / .exe)
│  ├─ api/routes.py     ← REST API
│  ├─ core/ai_processor.py ← กล้อง + ตรวจจับ + จดจำใบหน้า
│  └─ database/models.py   ← ตาราง SQLAlchemy (SQLite)
└─ frontend/            ← หน้าเว็บ (HTML/CSS/JS)
```

## REST API (บางส่วน)

| Method | Endpoint | คำอธิบาย |
|---|---|---|
| GET | `/api/video_feed` | สตรีมวิดีโอ MJPEG |
| POST | `/api/scan` | สแกนใบหน้าจากกล้อง 1 ครั้ง (`?auto=1` = โหมดอัตโนมัติ) |
| GET | `/api/users` | รายชื่อผู้ใช้ |
| POST | `/api/register?name=...` | ลงทะเบียนใบหน้า |
| GET | `/api/logs?name=&date_from=&date_to=&department=` | ประวัติ + ค้นหา |
| GET/POST | `/api/config` | อ่าน/บันทึกการตั้งค่า |
