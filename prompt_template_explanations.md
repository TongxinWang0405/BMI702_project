# Prompt Template
We use prompt templates to expand class labels in our datasets. 

Depending on the information available, we classified our data into three categories: datasets with tissue label only, tissue + condition labels, and detailed meta-data.

We will sample our data to contain 20% from tissue only, 30% from tissue + condition, 50% from tissue + meta-txt

The following are prompt template we currently adopted.

## For datasets with only tissue name
Given tissue name {Tissue}:
```python
[
    "An ultrasound image of {Tissue}.",
    "A B-mode ultrasound showing {Tissue}.",
    "Sonographic appearance of {Tissue}.",
    "This is an ultrasound image of {Tissue}.",
    "A grayscale ultrasound image demonstrating {Tissue}.",
    "Ultrasound findings consistent with {Tissue}.",
    "A clinical ultrasound scan of {Tissue}.",
    "An echographic image of {Tissue}.",
    "This sonogram shows {Tissue}.",
    "A diagnostic ultrasound image of {Tissue}, obtained for clinical evaluation.",
]
```

## For datasets with tissue name and condition(e.g. benign vs. malignant, normal)
Given tissue name {Tissue} and condition {Condition}:
```python
[
    "An ultrasound image of {Tissue} with {Condition} findings.",
    "A B-mode ultrasound of {Tissue}, consistent with {Condition}.",
    "Sonographic appearance of {Condition} {Tissue}.",
    "This ultrasound demonstrates {Tissue} exhibiting features of {Condition}.",
    "A clinical ultrasound scan of {Tissue}, indicative of {Condition} pathology.",
    "Echographic findings of {Tissue} showing {Condition} characteristics.",
    "This sonogram of {Tissue} is consistent with a {Condition} diagnosis.",
    "A diagnostic ultrasound image of {Tissue}, with imaging features suggestive of {Condition}.",
    "Ultrasound of {Tissue} presenting sonographic signs of {Condition}.",
    "A grayscale ultrasound demonstrating {Condition} changes in {Tissue}.",
]
```

## For datasets with meta-text
We will use mroe complicated, dataset-specific logic to prepare text captions.
Given the following placeholder, potential columns to fill in from the raw dataset meta-data, and fall-back for missingness:
{PatientInfo}       Age, Gender, Is the patient pregnant                                        "a patient"
{Region}            Tissue, Tissue_composition, zone (aggregated)                               "unknown region"
{Findings}          Shape, Margin, Echogenicity, Posterior_features, consolidation/effusion     "unremarkable findings"
{Condition}         Diagnosis, Classification, BIRADS, Interpretation, zone-derived severity    "unspecified condition"
```python
[
    "Ultrasound of {Region} in {PatientInfo}. {Findings}. Assessment: {Condition}.",
    "Sonographic findings in {Region}: {Findings}. {PatientInfo}. Diagnosis: {Condition}.",
    "An ultrasound image of {Region} consistent with {Condition}. {Findings}.",
    "{Region} evaluated by sonography. {Findings}. Conclusion: {Condition}.",
    "{PatientInfo} underwent ultrasound imaging. {Findings} identified in {Region}, suggestive of {Condition}.",
    "Sonography: {Region}, {PatientInfo}. {Findings}. {Condition}.",
    "{PatientInfo}. Ultrasound of {Region} reveals {Findings}, consistent with {Condition}.",
    "{Condition} pattern on ultrasound. {Region} demonstrates {Findings}. {PatientInfo}.",
    "Sonographic examination of {Region}. {Findings}. {Condition}.",
    "Ultrasound performed on {PatientInfo}. Examination of {Region} revealed {Findings}. Impression: {Condition}.",
]
```


# References
This method has been widely adopted across published benchmarks, including works we reference in our proposal:

see Section 3.1.4. PROMPT ENGINEERING AND ENSEMBLING, CLIP paper
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya
Sutskever. Learning transferable visual models from natural language supervision. In Marina
Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine
Learning, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR,
18–24 Jul 2021.

see Supplementary Data Table 35, CONCH paper
Ming Y. Lu, Bowen Chen, Drew F. K. Williamson, Richard J. Chen, Ivy Liang, Tong Ding,
Guillaume Jaume, Igor Odintsov, Long Phi Le, Georg Gerber, Anil V. Parwani, Andrew Zhang,
and Faisal Mahmood. A visual-language foundation model for computational pathology. Nature
Medicine, 30(3):863–874, 2024.