# Revealing Interconnections between Diseases: Can Large Language Models Help?

## Repository structure
The obtaining disease interconnections' matrices are in folders `M<digit>_<methods groups names>`:
* `M1_Basic_statistics` includes the code for obtaining matrices with Fisher's exact test (`M1_1_Fisher_exact_test.ipynb`) and with Jaccard similarity (`M1_2_Jaccard_similarity.ipynb`)
* `M2_MLM` includes data preprocessing for MLM's training (`preprocess_for_mlm.ipynb`), hyperparameter optimization with Optuna (`mlm_hpo.py`), and inference of the whole code for the hyperparameters from the best Optuna trial (`mlm_best_trial.py`) 
* `M3-M4_Med_domain_pretrain` includes the code for obtaining matrices with Med-BERT (`M3_Med_BERT.ipynb`) and BioClinicalBERT (`M4_BioclinicalBERT.ipynb`)
* M5-M6_Text_pretrained includes matrices obtained from text-based methods: pretrained general-purpose BERT (`M5_Pretrained_BERT.ipynb`) and Yandex Doc Search (`M6_Yandex_Doc_Search.ipynb`)
* `M7_LLMs` includes the code for obtaining matrices with LLMs. The "row" prediction prompt was used: 
```
 I'll give you ICD categories (for example, C25, NOT C25.0!) and their descriptions. You have to tell me, If a patient has an ICD code for a given category in their medical record, what other categories of codes are also likely to be in their medical record?

        ANSWER IN JSON FORMAT:
        \{
            "comment": \<your thoughts and explanations\>,
            "answer": \<list of categories in square brackets, separated by comma, for example: [A01, C05, ..., H12]\>
        \}
        DO NOT ADD ANYTHING ELSE IN YOUR ANSWER. 
        
        TEMPLATE\_MULTI = \{\{
            icd\_code: \{\},
            description: \{\},
        \}\}
```
* `M8_Genes_network` includes the code and data provided by the authors of _Goh, Kwang-Il, et al. "The human disease network." Proceedings of the National Academy of Sciences 104.21 (2007): 8685-8690_. The jupyter notebook `paper_2007_graph.ipynb` offers transforming the disease network into graph and mapping disease names into ICD-10 codes 

The experiments are in folders:
* `exp1_Visualization_Correlation` - visualization of disease interconnetions' matrices and Spearman correlations (`plot_heatmaps.ipynb` - for MIMIC; `plot_heatmaps_closed_data.ipynb` - for Closed data)
* `exp2_Graph_based_comparison` - comparison of methods with graphs (`graphs_comparison.ipynb` - for MIMIC; `graphs_comparison_closed_data.ipynb` - for Closed data)

## Visualization of disease interconnections obtained from MIMIC-IV

![](./matrices_viz/mimic/methods_on_real_data.png)
![](./matrices_viz/mimic/pretrained_on_medical_domain.png)
![](./matrices_viz/mimic/text-based_methods.png)
![](./matrices_viz/mimic/llms.png)

## Visualization of disease interconnections obtained from Closed Clinical Data

![](./matrices_viz/closed_data/methods_on_real_data.png)
![](./matrices_viz/closed_data/pretrained_on_medical_domain.png)
![](./matrices_viz/closed_data/text-based_methods.png)
![](./matrices_viz/closed_data/llms.png)

## Spearman correlation between obtained matrices

* **MIMIC-IV dataset**  
![MIMIC-IV](./matrices_viz/mimic/spearman_corr.png)

* **Closed Clinical Data**  
![Closed Clinical Data](./matrices_viz/closed_data/spearman_corr.png)
