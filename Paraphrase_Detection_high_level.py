#Language model --> เป็นโมเดลที่ใช้ทำนายคำถัดไป  ex: my --> name      เป็นการสร้างโมเดลที่มีความรู้ในด้านภาษาศาสตร์
#                                               my name --> is 

import transformers # เป็น library Hugging face
import datasets
import numpy as np

################################################################################# load & setup dataset ########################################################################################################

row_dataset = datasets.load_dataset("glue","mrpc") # GLUE (General Language Understanding Evaluation) Benchmark คือชุดของงานทดสอบมาตรฐานที่ใช้ในการประเมินประสิทธิภาพของโมเดลการประมวลผลภาษาธรรมชาติ (NLP)
                                                   # # mrpc =  Microsoft Research Paraphrase Corpus เป็นชุดข้อมูลที่ใช้ในการประเมินและฝึกสอนโมเดลการประมวลผลภาษาธรรมชาติ (NLP) 
                                                        # โดยเฉพาะในงานตรวจจับประโยคที่มีความหมายเหมือนกันหรือคล้ายกัน (Paraphrase Detection)

# ตัดคำ or โหลด Tokenizer สำหรับ BERT model    
                             
#  โมเดล bert-base-uncased และ bert-base-cased เป็นเวอร์ชันของโมเดล BERT ที่มีการฝึกอบรมด้วยข้อมูลที่แตกต่างกันในแง่ของการจัดการตัวพิมพ์ใหญ่และตัวพิมพ์เล็ก
#           : bert-base-uncased --> Apple", "apple", และ "APPLE" จะถูกมองว่าเป็นคำเดียวกัน (ตัวพิมพ์ใหญ่และตัวพิมพ์เล็กจะถูกแปลงเป็นตัวพิมพ์เล็กทั้งหมด) 
#           : bert-base-cased  -->  "Apple" และ "apple"       จะถูกมองว่าเป็นคำที่แตกต่างกัน. (นโมเดลนี้ ตัวพิมพ์ใหญ่และตัวพิมพ์เล็กจะถูกเก็บรักษาไว้.)
num_labels = len(set(row_dataset["train"]["label"]))


tokenizer = transformers.BertTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-cased") ##โหลดตัวตัดคำ (tokenizer) ที่ได้ถูกฝึกมาแล้ว (pretrained) สำหรับโมเดล bert-base-cased
                                                                                                        #ทำหน้าที่ รับคำ-->ตัดคำ-->map คำ ให้เป็น dict lookup สำหรับ model bert-base-cased

def encoder_function(sent):
    result = tokenizer(sent["sentence1"],sent["sentence2"],truncation=True)#truncation=True เป็นการกำหนดให้ tokenizer ทำการตัดข้อความที่มีความยาวเกินค่าที่กำหนดใน model_max_length ให้พอดีกับขีดจำกัดนี้
    return(result)

tokenized_data = row_dataset.map(encoder_function,batched=True)
data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
        #Data Collator คือออบเจกต์ที่ช่วยในการเตรียมข้อมูลสำหรับการป้อนเข้าโมเดลระหว่างการฝึก (training) หรือการประเมินผล (evaluation) โดยเฉพาะอย่างยิ่งการจัดการกับ batch ของข้อมูลที่มีความยาวไม่เท่ากัน.
        #transformers.DataCollatorWithPadding เป็น Data Collator ที่มาพร้อมกับการเติมข้อมูล (padding) เพื่อให้ batch ของข้อมูลมีความยาวเท่ากัน.
        #tokenizer=tokenizer หมายถึงการใช้ tokenizer ที่ระบุ (ในที่นี้คือ tokenizer ที่โหลดมาจากโมเดล bert-base-cased หรือ bert-base-uncased) ในการตัดคำและเติมข้อมูล.


##################################################################################### create model ########################################################################################################
#เพิ่ม classifier head ด้วย AutoModelForSequenceClassification
model = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path="bert-base-cased",num_labels=num_labels)



#################################################################### กำหนดข้อมูลที่ส่งไปให้ฟังก์ชัน  training (Training argument) ########################################################################################################

training_args = transformers.TrainingArguments(output_dir="test-trainer",evaluation_strategy="epoch",num_train_epochs=3) # output_dir="test-trainer" กำหนดชื่อไดเรกทอรี (directory) ที่จะใช้บันทึกผลการฝึก
                                                                                                      # กำหนดการประเมินผลโมเดลหลังจากแต่ละ epoch


#################################################################################### Create Metirc ########################################################################################################

def compute_metrics(eval_pred):
    metrics = datasets.load_metric("glue","mrpc")
    pred,label = eval_pred
    pred = np.argmax(pred,axis=-1)
    return (metrics.compute(predictions=pred,references=label))


###################################################################################### fine tune ########################################################################################################

trainer = transformers.Trainer(
    model, # โมเดลที่ผ่านการเพิ่ม classifier head แล้ว
    training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

###################################################################################### test model on test set ########################################################################################################


preds = trainer.predict(test_dataset=tokenized_data["test"])
print(preds)
metrics = datasets.load_metric("glue","mrpc")
pred = np.argmax(preds.predictions,axis=-1)
print(metrics.compute(predictions=pred,references=preds.label_ids))

#{'accuracy': 0.664927536231884, 'f1': 0.7986062717770035}/0
#{'accuracy': 0.8098550724637681, 'f1': 0.8575152041702867}/1




