#Language model --> เป็นโมเดลที่ใช้ทำนายคำถัดไป  ex: my --> name      เป็นการสร้างโมเดลที่มีความรู้ในด้านภาษาศาสตร์
#                                               my name --> is 

import torch.utils
import torch.utils.data
import tqdm.auto
import transformers # เป็น library Hugging face
import datasets
import numpy as np
import torch
import tqdm
import pandas as pd
import os

loot_path = r"D:\machine_learning_AI_Builders\บท4\NLP\Paraphrase_Detection"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

################################################################################# load & setup dataset ########################################################################################################

row_dataset = datasets.load_dataset("glue","mrpc") # GLUE (General Language Understanding Evaluation) Benchmark คือชุดของงานทดสอบมาตรฐานที่ใช้ในการประเมินประสิทธิภาพของโมเดลการประมวลผลภาษาธรรมชาติ (NLP)
                                                   # # mrpc =  Microsoft Research Paraphrase Corpus เป็นชุดข้อมูลที่ใช้ในการประเมินและฝึกสอนโมเดลการประมวลผลภาษาธรรมชาติ (NLP) 
                                                        # โดยเฉพาะในงานตรวจจับประโยคที่มีความหมายเหมือนกันหรือคล้ายกัน (Paraphrase Detection)

# ตัดคำ or โหลด Tokenizer สำหรับ BERT model    
                             
#  โมเดล bert-base-uncased และ bert-base-cased เป็นเวอร์ชันของโมเดล BERT ที่มีการฝึกอบรมด้วยข้อมูลที่แตกต่างกันในแง่ของการจัดการตัวพิมพ์ใหญ่และตัวพิมพ์เล็ก
#           : bert-base-uncased --> Apple", "apple", และ "APPLE" จะถูกมองว่าเป็นคำเดียวกัน (ตัวพิมพ์ใหญ่และตัวพิมพ์เล็กจะถูกแปลงเป็นตัวพิมพ์เล็กทั้งหมด) 
#           : bert-base-cased  -->  "Apple" และ "apple"       จะถูกมองว่าเป็นคำที่แตกต่างกัน. (นโมเดลนี้ ตัวพิมพ์ใหญ่และตัวพิมพ์เล็กจะถูกเก็บรักษาไว้.)

tokenizer = transformers.BertTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-cased") ##โหลดตัวตัดคำ (tokenizer) ที่ได้ถูกฝึกมาแล้ว (pretrained) สำหรับโมเดล bert-base-cased
                                                                                                        #ทำหน้าที่ รับคำ-->ตัดคำ-->map คำ ให้เป็น dict lookup สำหรับ model bert-base-cased

def encoder_function(sent):
    result = tokenizer(sent["sentence1"],sent["sentence2"],truncation=True)#truncation=True เป็นการกำหนดให้ tokenizer ทำการตัดข้อความที่มีความยาวเกินค่าที่กำหนดใน model_max_length ให้พอดีกับขีดจำกัดนี้
    return(result)

num_labels = len(set(row_dataset["train"]["label"]))

tokenized_data = row_dataset.map(encoder_function,batched=True)
data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_data = tokenized_data.remove_columns(['sentence1', 'sentence2','idx'])
tokenized_data = tokenized_data.rename_column("label",'labels')
tokenized_data.set_format("torch")
print(tokenized_data)


train_loader = torch.utils.data.DataLoader(tokenized_data["train"],num_workers=0,shuffle=True,batch_size=8,collate_fn=data_collator)
val_loader = torch.utils.data.DataLoader(tokenized_data["validation"],num_workers=0,shuffle=False,batch_size=8,collate_fn=data_collator)
test_loder = torch.utils.data.DataLoader(tokenized_data["test"],num_workers=0,shuffle=False,batch_size=8,collate_fn=data_collator)

############################################################################### create model #################################################################################################

model = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path="bert-base-cased",num_labels=num_labels)
for batch in train_loader:
    break
output = model(**batch) # ** ใน Python ใช้สำหรับการแตกชุดข้อมูล (unpacking) ของ dictionary หรือ mapping object ให้อยู่ในรูปแบบของ arguments ในฟังก์ชัน โดยจะส่งค่าทุกคู่คีย์-ค่าใน dictionary ไปเป็นอาร์กิวเมนต์คีย์เวิร์ด



optimizer = transformers.AdamW(model.parameters(),lr=5e-5)

metrics = datasets.load_metric("glue","mrpc")


############################################################################### set paramiters #################################################################################################

num_epochs = 1
num_train_steps = num_epochs * len(train_loader)


lr_scheduler = transformers.get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_training_steps=num_train_steps,
    num_warmup_steps=0

)

columns = ["epoch","train_loss","valid_loss","accuracy","f1_score"]
csv_df = pd.DataFrame(columns=columns)
csv_file_name = "result_log.csv" 
log_csv_path = os.path.join(loot_path,csv_file_name)

bast_val_loss = float("inf")
acc=0
############################################################################### train model #################################################################################################

print("\n...Training...\n")

model.to(device)

for epoch in range(num_epochs):

    train_loss,val_loss =0,0
    metrics = datasets.load_metric("glue","mrpc")
    model.train()

    for batch in tqdm.auto.tqdm(train_loader):
        optimizer.zero_grad()
        batch = {k:v.to(device) for k,v in batch.items()}
        output = model(**batch)  # ** ใน Python ใช้สำหรับการแตกชุดข้อมูล (unpacking) ของ dictionary หรือ mapping object ให้อยู่ในรูปแบบของ arguments ในฟังก์ชัน โดยจะส่งค่าทุกคู่คีย์-ค่าใน dictionary ไปเป็นอาร์กิวเมนต์คีย์เวิร์ด
        loss = output.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step() # lr_scheduler.step() ใช้เพื่อปรับอัตราการเรียนรู้ (learning rate) ของโมเดล จะถูกเรียกทุกครั้งหลังจากที่ optimizer ทำการปรับค่าของพารามิเตอร์ในแต่ละ batch เพื่ออัปเดตอัตราการเรียนรู้ตามที่กำหนดใน scheduler

        train_loss += loss.item()*len(train_loader)

    model.eval()

    for batch in tqdm.tqdm(val_loader):
        batch = {k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            output = model(**batch)
        logits = output.logits
        pred = torch.argmax(logits,dim=-1)
        metrics.add_batch(predictions=pred, references=batch["labels"])
        loss = output.loss
        val_loss += loss.item() * len(val_loader)
    
    val_loss /= len(val_loader.dataset)
    train_loss /= len(train_loader.dataset)

    metric = metrics.compute()
    accuracy  = metric["accuracy"]
    f1_score = metric["f1"]
    print(f"{epoch+1}/{num_epochs}\nTraing_loss : {train_loss}, Valid_loss : {val_loss}, Accuracy : {accuracy}, F1-scroe : {f1_score}")
    
    each_epoch_log = {f"{columns[0]}":int(epoch)+1,
                      f"{columns[1]}":train_loss,
                      f"{columns[2]}":val_loss,
                      f"{columns[3]}":accuracy,
                      f"{columns[4]}":f1_score
                      }
    csv_df = pd.concat([csv_df,pd.DataFrame([each_epoch_log])],ignore_index=True,axis=0)
    csv_df.to_csv(log_csv_path,index=False)

    if accuracy > acc:
        acc = accuracy
        model_path = os.path.join(loot_path,"model")
        model.save_pretrained(model_path )
        tokenizer.save_pretrained(model_path )
        print(f"\n***** Save Complete ******\n")

