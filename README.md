# Contents

### Data preparation notebook
[data_preprocessing.ipynb](notebooks/data_preprocessing.ipynb)

### Fine-tuning notebook
[ORPO_Fine_tuning_LLama3_8B.ipynb](notebooks/ORPO_Fine_tuning_LLama3_8B.ipynb)

### Model evaluation notebook
[model_evaluation.ipynb](notebooks/model_evaluation.ipynb)

# Approach and model choice

The approach I took to the medical question-answering system was to fine-tune an open-source Large Language Model, customizing it with domain-specific medical data to improve its accuracy and reliability in providing appropriate answers.
It was challenging to choose between this approach and a RAG-based one, but given the short time to deliver the system and the risk of the retrieval system surfacing relevant content related to entirely different diseases and medical conditions, I opted for this option.
The difficult aspect of the fine-tuning option was the significant amount of computational resources required. To address this and achieve the best results with minimal preparation time and GPU usage, we used the Odds Ratio Preference Optimization (ORPO) algorithm with Quantized Low-Rank Adaptation (QLoRA).

# Data preparation

The original dataset contained question-and-answer pairs. To convert it into a preferences dataset, we had to craft incorrect answers to serve as the "rejected answer", while the correct answers were designated as the "chosen answer".

Also we noted that 5.8% of the questions had more than one answer, even after removing the samples where both question and answer were the same. To avoid data leakage and facilitate evaluation, we decided to summarize the multiple answers when there was more than one.

For both crafting the incorrect answers and summarizing the multiple ones, we used the Gemini 1.5 Flash model; there are more details in the data_preprocessing.ipynb notebook.

In the correct answers, we included a disclaimer stating: "This information is for educational purposes only and should not be considered medical advice.  I cannot make diagnoses.  Always consult a qualified healthcare professional with any health concerns or before making any decisions related to your medical care".

Finally we moved on to the fine tuning stage.

# Model training

Using Google Colab, we loaded the Llama3 8b model, applied quantization to 4-bit precision using the bitsandbytes library, reduced the number of trainable parameters with the PEFT library, and applied the ORPO optimization, which combines Supervised Fine-Tuning (SFT) and preference alignment into a single objective function.

Using the NVIDIA A100 40GB GPU, we were able to train for 1 epoch using the entire final dataset of 14,981 samples, and it took 10 hours and 20 minutes.

# Model evaluation

We took the LLaMA 3 8B base model as our baseline. We used the groq inference service to submit questions to the original model and collect responses, and we submitted questions to the fine-tuned model and collected responses.
One important note about the inference settings is that we restricted the fine tuned model responses to the maximum of 150 new tokens to speed up the processing. For this reason some responses are truncated in the middle of a sentence.

# Model evaluation

We evaluated the model using a set of metrics, starting from the simpler ones based on ngrams BLEU and ROUGE, then proceeding to a Sentence Transformer (SBERT) similarity score evaluation.
The fine-tuned model outperformed the base model on all metrics:

### BLEU:  
Llama3 8B (base model):    0.039  
Fine tuned model:          0.079  
Improvement:               103.91%  

![bleu](images/bleu.png)

### ROUGE-L sentence-level:  
Llama3 8B (base model):    0.154  
Fine tuned model:          0.229  
Improvement:                 48.40%  

![rouge](images/rouge.png)

### Sentence Transformers (SBERT) average similarity:  
Llama3 8B (base model):    0.624  
Fine tuned model:          0.632  
Improvement:                  1.23%  

![sbert average](images/sbert_average.png)

### Sentence Transformers (SBERT) median similarity:
Llama3 8B (base model):    0.624  
Fine tuned model:          0.648  
Improvement:                  3.90%  

![sbert median](images/sbert_median.png)

# Samples

### Sample #1
```
QUESTION
------------------------------
Is Pfeiffer syndrome inherited ?


CORRECT ANSWER
------------------------------
This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
Disclaimer: This information is for educational purposes only and should not be considered medical advice.  I cannot provide diagnoses.  Always consult with a qualified healthcare professional for any health concerns or before making any decisions related to your medical care.


FINE TUNED MODEL RESPONSE
similarity 0.884
------------------------------
Is Pfeiffer syndrome inherited ? Pfeiffer syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from an affected parent. Other cases result from new mutations in the gene and occur in people with no history of the disorder in their family.
Disclaimer: This information is for educational purposes only and should not be considered medical advice.  I cannot provide diagnoses.  Always consult with a qualified professional for any health concerns or before making any decisions related to your medical care and treatment. I cannot answer personal medical questions.


BASE MODEL RESPONSE
similarity 0.447
------------------------------
Pfeiffer syndrome is a rare genetic disorder that is typically inherited in an autosomal dominant pattern. This means that a single copy of the mutated gene is enough to cause the condition.

Pfeiffer syndrome is caused by mutations in the FGFR1, FGFR2, or FGFR3 genes, which are responsible for encoding fibroblast growth factor receptors. These receptors play a crucial role in bone development and growth.

Most cases of Pfeiffer syndrome are caused by a spontaneous mutation that occurs in the egg or sperm during conception, rather than being inherited from a parent. However, there is a small chance that a person with Pfeiffer syndrome can pass the mutated gene to their children, making it a potential concern for future pregnancies.

The risk of inheritance depends on the type of mutation and the mode of inheritance:

* Autosomal dominant inheritance: If one parent has Pfeiffer syndrome, each child has a 50% chance of inheriting the mutated gene and developing the condition.
* Autosomal recessive inheritance: If both parents are carriers of a Pfeiffer syndrome-causing mutation, each child has a 25% 
chance of inheriting two copies of the mutated gene (one from each parent) and developing the condition.
* Mosaicism: In some cases, a person with Pfeiffer syndrome may have mosaicism, where some cells in their body have the mutated gene and others do not. This can make it more difficult to predict the likelihood of inheritance.

It's important to note that Pfeiffer syndrome is a rare condition, and genetic counseling is recommended for families affected by the disorder to understand the inheritance pattern and make informed decisions about future pregnancies.
```

### Sample #2
```
QUESTION
------------------------------
Do you have information about Cancer Chemotherapy


CORRECT ANSWER
------------------------------
Summary : Normally, your cells grow and die in a controlled way. Cancer cells keep forming without control. Chemotherapy is drug therapy that can kill these cells or stop them from multiplying. However, it can also harm healthy cells, which causes side effects.    During chemotherapy you may have no side effects or just a few. The kinds of side effects you have depend on the type and dose of chemotherapy you get. Side effects vary, but common ones are nausea, vomiting, tiredness, pain and hair loss. Healthy cells usually recover after chemotherapy, so most side effects gradually go away.    Your course of therapy will 
depend on the cancer type, the chemotherapy drugs used, the treatment goal and how your body responds. You may get treatment every day, every week or every month. You may have breaks between treatments so that your body has a chance to build new healthy cells. You might take the drugs by mouth, in a shot or intravenously.    NIH: National Cancer Institute
Disclaimer: This information is for educational purposes only and should not be considered medical advice.  I cannot provide diagnoses.  Always consult with a qualified healthcare professional for any health concerns or before making any decisions related to your medical care.


FINE TUNED MODEL RESPONSE
similarity 0.551
------------------------------
Do you have information about Cancer Chemotherapy?
Cancer chemotherapy is the use of drugs to treat cancer. Chemotherapy is often used to treat cancer that has spread to other parts of the body. It is also used to treat cancer that has come back after treatment. Chemotherapy is often used with other treatments, such as surgery or radiation therapy.
Chemotherapy works by killing cancer cells. It does this by damaging the cells' ability to grow and divide. The drugs used in chemotherapy are called cytotoxic drugs. They are designed to kill cells that are dividing quickly. This includes cancer cells, but it also includes normal cells that divide quickly, such as hair cells and cells in the lining of the mouth and intestines.
Chemotherapy is usually given in cycles. This means that the drugs are


BASE MODEL RESPONSE
similarity 0.576
------------------------------
Cancer chemotherapy is a type of cancer treatment that uses one or more anti-cancer drugs to destroy cancer cells. Here's an overview of chemotherapy:

**What is chemotherapy?**

Chemotherapy is a systemic treatment, meaning it targets cancer cells throughout the body. It works by:

1. Interfering with cancer cell growth and division
2. Killing cancer cells or stopping them from multiplying
3. Reducing the size of tumors

**How does chemotherapy work?**

Chemotherapy drugs can be classified into several categories based on their mechanism of action:

1. **Alkylating agents**: These drugs damage DNA, making it difficult for cancer cells to divide and grow. Examples include cyclophosphamide and ifosfamide.
2. **Antimetabolites**: These drugs interfere with the production of nucleic acids, such as DNA and RNA, which are essential for cancer cell growth. Examples include methotrexate and fluorouracil.
3. **Antitumor antibiotics**: These drugs interfere with DNA replication and repair, making it difficult for cancer cells to grow and divide. Examples include doxorubicin and daunorubicin.
4. **Mitotic inhibitors**: These drugs prevent cancer cells from dividing by interfering with the process of cell division. Examples include taxanes (e.g., paclitaxel) and vinca alkaloids (e.g., vincristine).
5. **Targeted therapies**: These drugs target specific molecules or pathways that are involved in cancer cell growth and survival. Examples include trastuzumab (Herceptin) and imatinib (Gleevec).

**Types of chemotherapy regimens**

Chemotherapy regimens can be categorized based on the frequency and duration of treatment:

1. **Single-agent chemotherapy**: One drug is used to treat the cancer.
2. **Combination chemotherapy**: Multiple drugs are used together to treat the cancer.
3. **Induction chemotherapy**: A high-dose regimen used to induce a response in patients with advanced cancer.
4. **Consolidation chemotherapy**: A lower-dose regimen used to maintain a response after initial treatment.
5. **Maintenance chemotherapy**: A low-dose regimen used to prevent relapse after initial treatment.

**Side effects of chemotherapy**

Chemotherapy can cause a range of side effects, including:

1. **Hair loss**: Temporary or permanent hair loss due to the destruction of hair follicles.
2. **Fatigue**: Feeling tired or weak due to the destruction of bone marrow cells.
3. **Nausea and vomiting**: Feeling sick or throwing up due to the destruction of stomach lining cells.
4. **Mucositis**: Inflammation of the mucous membranes, leading to mouth sores, diarrhea, or rectal bleeding.
5. **Neutropenia**: A decrease in white blood cells, making it harder for the body to fight infections.
6. **Anemia**: A decrease in red blood cells, leading to fatigue, weakness, or shortness of breath.
7. **Thrombocytopenia**: A decrease in platelets, leading to easy bruising or bleeding.

**New developments in chemotherapy**

Researchers are continually working to improve chemotherapy by:

1. **Developing new drugs**: Targeting specific molecular pathways or mechanisms of action.
2. **Combining chemotherapy with other treatments**: Such as immunotherapy, targeted therapy, or radiation therapy.
3. **Personalizing chemotherapy**: Tailoring treatment to individual patients based on their genetic profile, tumor characteristics, or response to treatment.
4. **Improving delivery methods**: Using nanoparticles, liposomes, or other delivery systems to increase the effectiveness and reduce side effects of chemotherapy.

**Conclusion**

Cancer chemotherapy is a complex and evolving field, with ongoing research aimed at improving treatment outcomes and reducing side effects. While chemotherapy can be effective in treating various types of cancer, it is essential to discuss the potential benefits and risks with a healthcare provider to determine the best course of treatment for an individual patient.
```
  
  
### Sample #3

```
QUESTION
------------------------------
What is (are) Rheumatic Fever ?


CORRECT ANSWER
------------------------------
Rheumatic fever is an inflammatory condition that may develop after infection with group A Streptococcus bacteria, such as strep throat or scarlet fever. It is primarily diagnosed in children between the ages of 6 and 16 and can affect the heart, joints, nervous system and/or skin. Early signs and symptoms include sore throat; swollen red tonsils; fever; headache; and/or muscle and joint aches. Some affected people develop rheumatic heart disease, which can lead to serious inflammation and carring of the heart valves. It is not clear why some people who are infected with group A Streptococcus bacteria go on to develop rheumatic fever, while others do not; however, it appears that some families may have a genetic susceptibility to develop the condition. Treatment usually includes antibiotics and/or anti-inflammatory medications.
Disclaimer: This information is for educational purposes only and should not be considered medical advice.  I cannot provide diagnoses.  Always consult with a qualified healthcare professional for any health concerns or before making any decisions related to your medical care.


FINE TUNED MODEL RESPONSE
similarity 0.774
------------------------------
What is (are) Rheumatic Fever ??
Rheumatic fever is a disease that can occur after a strep throat infection. It can cause painful joints, a rash, and heart 
problems. It is most common in children ages 5 to 15.
What causes Rheumatic Fever?
Rheumatic fever is caused by a strep throat infection. The strep bacteria can also cause scarlet fever and tonsillitis.
What are the signs and symptoms of Rheumatic Fever?
The signs and symptoms of rheumatic fever may include:
Joint pain and swelling, especially in the knees, ankles, elbows, and wrists
A rash that looks like hives
A red, raised rash on the palms of the hands and soles of the feet
Swelling of the lymph nodes in


BASE MODEL RESPONSE
similarity 0.670
------------------------------
Rheumatic fever (RF) is a serious inflammatory disease that can occur in people of all ages, but it is most common in children and adolescents. It is caused by a bacterial infection, usually streptococcal pharyngitis (strep throat), and is characterized by inflammation of the joints, heart, skin, and brain.

There are two main types of rheumatic fever:

1. **Acute rheumatic fever (ARF)**: This is the most common form of the disease, and it occurs within 2-3 weeks after a strep throat infection. Symptoms include:
	* Fever
	* Joint pain and swelling (especially in the knees, elbows, and wrists)
	* Skin rash (erythema marginatum)
	* Carditis (inflammation of the heart)
	* Neurological symptoms (such as headache, confusion, and seizures)
2. **Chronic rheumatic fever (CRF)**: This is a more severe and long-term form of the disease, which can occur months or even years after the initial strep throat infection. Symptoms include:
	* Recurring episodes of joint pain and swelling
	* Heart valve damage (leading to heart failure or arrhythmias)
	* Neurological symptoms (such as memory loss, personality changes, and seizures)

Rheumatic fever is caused by an abnormal immune response to a streptococcal infection, which leads to the production of antibodies that attack the body's own tissues. The disease is more common in developing countries and in areas with poor living conditions.

If left untreated, rheumatic fever can lead to serious complications, such as:

* Heart valve damage
* Heart failure
* Arrhythmias
* Neurological damage
* Recurring episodes of joint pain and swelling

Treatment for rheumatic fever typically involves antibiotics to eliminate the streptococcal infection, as well as medications to reduce inflammation and alleviate symptoms. In severe cases, hospitalization may be necessary to manage complications.

Prevention is key, and this involves:

* Prompt treatment of strep throat infections with antibiotics
* Regular throat swabs to detect strep throat infections
* Good hygiene practices, such as frequent handwashing and covering the mouth and nose when coughing or sneezing

It's essential to seek medical attention if you or your child experiences symptoms of rheumatic fever, as early treatment can help prevent long-term damage and complications.
```

# Model performance and potential improvements

Although the model outperformed the baseline model on all metrics, a closer look at the model output reveals that the results were overstated.

The idea of adding a disclaimer to the correct answers was to enforce a desired policy, as this would be a desired behaviour of the model, making clear its limitations and advising people to seek appropriate medical help.

Instead, what happened was that the disclaimer in the model output boosted all the similarity metrics.

Looking closely at the model output, we get the impression that the optimisation may have compromised the model's performance.

Considering the cost and complexity of doing a full alignment of the model parameters using DPO, RLHF or any other method, it might make sense to try the RAG method, taking special care in the retrieval part not to mix different topics in context.

In fact, an interesting approach could be to start with an intent classification, where the question is first classified into a specific topic, and then surface the relevant content about that topic and let the model process the context and answer the question based on the right piece of information.
