<strong>IndicTrans</strong>:
<br>
---

+ To Evaluate the translated articles on the predicted ouput of IndicTrans we need to load the models and unzip them.
```
!wget https://ai4b-public-nlu-nlg.objectstore.e2enetworks.net/indic2en.zip
!unzip indic2en.zip
```
+ You can get the predictions for your original articles from the IndicTrans model using the below code(Telugu to English Translations)
```
from inference.engine import Model
model = Model('indic-en')
def process_line(line):
    hi=str(line)
    processed_line = model.translate_paragraph(hi, 'te', 'en')
    return processed_line
input_file_path = 'original_texts_te.txt'
output_file_path = 'Indic_te.txt'
with open(input_file_path, 'r') as input_file, open(output_file_path, 'a') as output_file:
    # Read each line from the input file
    for line in input_file:
      processed_line = process_line(line)
      output_file.write(processed_line + '\n')
input_file.close()
output_file.close()
```
+ Below code is for Hindi to English Translations
```
def process_line(line):
    hi=str(line)
    processed_line = model.translate_paragraph(hi, 'hi', 'en')
    return processed_line
input_file_path = 'original_texts_hi.txt'
output_file_path = 'Indic_hi.txt'
with open(input_file_path, 'r') as input_file, open(output_file_path, 'a') as output_file:
     for line in input_file:
         processed_line = process_line(line)
         output_file.write(processed_line + '\n')
input_file.close()
output_file.close()
```
+ To Detokenize the files we used below bash command
```
!cat {translated_texts_hi.txt,translated_texts_te.txt,Indic_te.txt,Indic_hi.txt} | sacremoses normalize > {translated_texts_hi.detok.txt,translated_texts_te.detok.txtIndic_te.detok.txt,Indic_hi.detok.txt}
```
+ Below command of secrebleu is used to fetch the BLEU and chrF++ metrics 
```
!sacrebleu translated_texts_hi.detok.txt -i Indic_hi.detok.txt -m bleu chrf ter --chrf-word-order 2
```
<strong>IndicTrans2</strong>:
<br>
---
+ To Evaluate the translated articles on the predicted ouput of IndicTrans2 we need to load the models and unzip them.
```
!wget https://indictrans2-public.objectstore.e2enetworks.net/it2_preprint_ckpts/indic-en-preprint.zip
!unzip indic-en-preprint.zip
```
+ You can get the predictions for your original articles from the IndicTrans model using the below code(Telugu to English Translations)
```
from inference.engine import Model
model = Model('indic-en-preprint/fairseq_model', model_type="fairseq")
def process_line(line):
    hi=str(line)
    processed_line = model.translate_paragraph(hi,'tel_Telu', 'eng_Latn')
    return processed_line


input_file_path = 'original_texts_te.txt'     # Replace with the path to your input file
output_file_path = 'Indic2_te.txt'   # Replace with the path to your output file

# Open the input file for reading and the output file for appending
with open(input_file_path, 'r') as input_file, open(output_file_path, 'a') as output_file:
    # Read each line from the input file
    for line in input_file:
      processed_line = process_line(line)
      output_file.write(processed_line + '\n')

# Close the files
input_file.close()
output_file.close()
```
+ Below code is for Hindi to English Translations
```
from inference.engine import Model
model = Model('indic-en-preprint/fairseq_model', model_type="fairseq")
def process_line(line):
    hi=str(line)
    processed_line = model.translate_paragraph(hi,'hin_Deva', 'eng_Latn')
    return processed_line


input_file_path = 'original_texts_hi.txt'     # Replace with the path to your input file
output_file_path = 'Indic2_hi.txt'   # Replace with the path to your output file

# Open the input file for reading and the output file for appending
with open(input_file_path, 'r') as input_file, open(output_file_path, 'a') as output_file:
    # Read each line from the input file
    for line in input_file:
      processed_line = process_line(line)
      output_file.write(processed_line + '\n')

# Close the files
input_file.close()
output_file.close()
```
+ Detokenization and Fetching the metrics is same as IndicTrans

<strong>Googletrans</strong>:
<br>
---
```
!pip install googletrans
rom googletrans import Translator

def translate_file(input_file, output_file, dest_language='en'):
    # Initialize the translator
    translator = Translator()

    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        input_text = file.read()

    # Translate the text
    translation = translator.translate(input_text, dest=dest_language)

    # Write the translated text to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(translation.text + '\n')

# Example usage
input_file_path = 'original_texts_hi.txt'   # Replace with the path to your Hindi input file
output_file_path = 'googletrans_hi.txt'  # Replace with the desired path for the output file

translate_file(input_file_path, output_file_path, dest_language='en')
from googletrans import Translator

def translate_file(input_file, output_file, dest_language='en'):
    # Initialize the translator
    translator = Translator()

    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        input_text = file.read()

    # Translate the text
    translation = translator.translate(input_text, dest=dest_language)

    # Write the translated text to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(translation.text + '\n')

# Example usage
input_file_path = 'original_texts_te.txt'   # Replace with the path to your Hindi input file
output_file_path = 'googletrans_te.txt'  # Replace with the desired path for the output file

translate_file(input_file_path, output_file_path, dest_language='en')
```
+ Detokenization and Fetching the metrics is same as IndicTrans

<strong>Results</strong>
<br>
---
| Model          | BLEU       | chrF2++          | 
| ---------------- | ------------- | ------------- | 
|IndicTrans(Hindi To English)       |{score: 23.6, verbose : 70.7/43.6/29.6/21.2(BP = 0.633 ,ratio = 0.686 ,hyp_len = 11356, ref_len = 16552)}| {score:46.7,nrefs: 1,case: mixed,eff: yes,nc: 6,nw: 2}| 
|IndicTrans2(Hindi To English)      |{score: 41.8, verbose : 75.8/55.0/42.7/34.0(BP = 0.843 ,ratio = 0.854, hyp_len = 14136, ref_len = 16552)}| {score: 62.0,nrefs: 1,case: mixed,eff: yes,nc: 6,nw: 2}        | 
|Google Trans(Hindi To English)    |{score: 58.9, verbose : 75.7/62.7/53.9/46.8(BP = 1.000 ,ratio = 1.138 ,hyp_len = 18839 ,ref_len = 16552)}| {score: 80.0,nrefs: 1,case: mixed,eff: yes,nc: 6,nw: 2}| 
|IndicTrans(Telugu To English)      |{score: 27.7, verbose : 65.0/38.2/24.9/16.9(BP = 0.868 ,ratio = 0.876 ,hyp_len = 15708, ref_len = 17936)} |{score: 52.7,nrefs: 1,case: mixed,eff: yes,nc: 6,nw: 2}|        
|IndicTrans2(Telugu To English) |{score: 41.9, verbose : 73.0/50.2/37.3/28.5(BP = 0.994 ,ratio = 0.946, hyp_len = 16967 ,ref_len = 17936)}|{score: 64.3,nrefs: 1,case: mixed,eff: yes,nc: 6,nw: 2}|        
|Google Trans(Telugu To English)         |{score: 92.6, verbose : 94.0/92.9/92.1/91.3(BP = 1.000 ,ratio = 1.048, hyp_len = 18802, ref_len = 17936)}| {score: 97.4,nrefs: 1,case: mixed,eff: yes,nc: 6,nw: 2}|  

<br>

<strong> Metric Values</strong>
<br>
---
<strong>'BLEU Score'</strong>
**Definition:** Metric that measures the similarity between machine-generated translations and reference translations by counting overlapping n-grams (word sequences) and computing a precision-based score.

<strong>'chrF++ Score'</strong>
**Definition:** Adding word unigrams (individual words) and bigrams (word pairs) to chrF leads to improved correlations with human assessments. This combination, referred to as "chrF++."

<strong>'Brevity Penalty'</strong>
**Definition:** "BP" stands for "Brevity Penalty." The brevity penalty is a component of the BLEU score that penalizes machine translations that are significantly shorter than the reference translations.

<strong>'Number of Characters (nc)'</strong>
**Definition:** "nc" likely stands for "Number of Characters." This metric will evaluate translation quality at a character-level granularity.

<strong>'Number of Words (nw)'</strong>
**Definition:** "nw" likely stands for "Number of Words." This metric will take into account word-level information when evaluating translation quality.

<strong>'Score'</strong>
**Definition:** The "score" represents the actual evaluation score obtained for a specific translation or set of translations.

<strong>'Hypothesis Length (hyp_len)'</strong>
**Definition:** This indicates the length of the machine-generated hypothesis (translation) being evaluated.

<strong>'Reference Length (ref_len)'</strong> 
**Definition:** This number represents the length of the reference translation against which the machine-generated hypothesis is being evaluated.
<br>

<strong> Event Extraction Using Chat gpt</strong>:
<br>
-------


+ Events were Extracted by sending the article as an input to chatgpt with extra prompting and loaded the outputs in Event_Extraction_Chatgpt.csv file which is present in Event_Extraction_Using_Chatgpt Folder .

```
import os
import openai
import pandas as pd
api_key = os.environ['OPENAI_API_KEY']
openai.api_key = api_key

def generate_predictions(content):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": content+ "Given a news article, please extract relevant events in the form of dictionaries. Each event should include keys for 'Disease', 'Location,' 'Incident' (either 'case' or 'death'), 'Incident_Type' (either 'new' or 'total'), and 'Number.' If the 'Disease' key is not present in an event, do not include the event in the result. Additionally, please make sure that no duplicate events are included in the list. Provide the extracted events as a list of dictionaries. If no events are extracted, the result should be an empty list."}]
    )
    return completion.choices[0].message['content']
data = pd.read_csv('output.csv')

with open('event.txt', 'w') as txt_file:
    # Iterate over rows and generate predictions
    for index, row in data.iterrows():
        trans_article = row['article']
        if pd.notna(trans_article):  # Skip NaN values
            prediction = generate_predictions(trans_article)
            data.at[index, 'predicted_label'] = prediction
            # Write the prediction to the text file
            txt_file.write(prediction)
        txt_file.write('\n')

data.to_csv('Event_Extraction_Chatgpt.csv', index=False)
```
+ Then the soft precision,Hard Precision ,Soft Recall,Hard Recall,F1 score were calculated for True_Labels(Manual Event Extraction),Pred_Label(Predictions by chat gpt).The jupiter notebook code.ipynb contains the code for it which is present in Event_Extraction_Using_Chatgpt Folder .
+ <strong>Soft Match</strong>:
Soft match is the fraction of the number of keys in a predicted event matched with the GT event out of the total keys'
+ <strong>Hard Match</strong>:
'Hard-match returns 1 if all the keys match between predicted and gt events, else return 0
+ Soft scores and Hard scores for every row were calculated and mentioned in Soft_Hard_Scores_Chat_gpt.csv file which is present in Event_Extraction_Using_Chatgpt Folder .
```
Average Soft-match score:  0.7669937555753785
Soft Precision:  0.6115220483641531
Soft Recall:  0.7669937555753785
Soft F1:  0.6804907004352982
----------------------
Average Hard-match score:  0.49687778768956287
Hard Precision:  0.3961593172119488
Hard Recall:  0.49687778768956287
Hard F1:  0.4408389394538979
```
+ Above are the Average Hard Scores and Soft Scores.
  
<strong>Added more data of articles(Event_1.csv) and also a city.csv file</strong>:

+ city.csv file contains synonyms of every place.The variant column are the diff names to a single location .so if we find variant in an event then we need to replace it with the value.so that we dont have different names for same locations and it is present in Event_Extraction_Using_Chatgpt Folder.
+ And also we have removed the rows which contain empty true events.
```
Average Soft-match score:  0.8137603795966784
Soft Precision:  0.6725490196078431
Soft Recall:  0.8137603795966784
Soft F1:  0.7364465915190552
----------------------
Average Hard-match score:  0.42823250296559906
Hard Precision:  0.353921568627451
Hard Recall:  0.42823250296559906
Hard F1:  0.3875469672571122
```
+ Above are the Average Hard score and Soft score after this changes and we can find an improvement in the soft scores.
+ Soft_Hard_Scores_Chat_gpt_1.csv contains scores of these changes which is present in Event_Extraction_Using_Chatgpt Folder.

<strong>Created a disease.csv file and worked on it </strong>:
+ File named disease.csv consist synonyms of every disease.The variant column are the diff names to a single disease.so if we find variant in an event then we need to replace it with the value.so that we dont have different names for same disease and it is present in Event_Extraction_Using_Chatgpt Folder.
```
Average Soft-match score:  0.8441281138790027
Soft Precision:  0.7094715852442665
Soft Recall:  0.8441281138790027
Soft F1:  0.7709642470205842
----------------------
Average Hard-match score:  0.5860023724792408
Hard Precision:  0.49252243270189433
Hard Recall:  0.5860023724792408
Hard F1:  0.5352112676056339
```
+ Above are the Average Hard score and Soft score after this changes and we can find an improvement in every score.
+ Soft_Hard_Scores_Chat_gpt_final.csv contains scores of these changes which is present in Event_Extraction_Using_Chatgpt Folder.

<strong>Installation of requirements/dependencies</strong>:
<br>
---
1.Installing the requirements by running the pip install -r requirements.txt



