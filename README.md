# Revealing Interconnections between Diseases: Can Large Language Models Help?

## Repository structure
The obtaining disease interconnections' matrices are in folders `M<digit>_<methods groups names>`:
* `M1_Basic_statistics` includes the code for obtaining matrices with Fisher's exact test (`M1_1_Fisher_exact_test.ipynb`) and with Jaccard similarity (`M1_2_Jaccard_similarity.ipynb`). Jaccard similarity formula: 
$$J_{i,j} = \frac{N_{i, j}}{N_i + N_j - N_{i,j}},$$
where $N_{i, j}$ is the number of patients diagnosed with both codes $i$ and $j$, $N_i$ and $N_j$ are the number of patients diagnosed with codes $i$ and $j$, respectively.
* `M2_MLM` includes data preprocessing for MLM's training (`preprocess_for_mlm.ipynb`), hyperparameter optimization with Optuna (`mlm_hpo.py`), and inference of the whole code for the hyperparameters from the best Optuna trial (`mlm_best_trial.py`).
    - For MIMIC-IV, preprocessing follows these steps:
        1. Filtering lengths of sequences of patient ICD codes between $5$ and $100$, following prior work _Placido, Davide, et al. "A deep learning algorithm to predict risk of pancreatic cancer from disease trajectories." Nature medicine 29.5 (2023): 1113-1122._.
        2. Applying the BERT masked pretraining strategy _Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." Proceedings of the 2019 ACL, vol. 1 (2019)._: in each sequence, $15\%$ of tokens are selected for prediction; of these selected tokens, $80\%$ are replaced with \texttt{[MASK]}, $10\%$ -- with a random token, and $10\%$ are left unchanged. The model is trained with cross-entropy loss on these tokens.
        
        The model architecture comprises an embedding layer $(\text{dimension }128)$ with positional encodings, followed by $3$ Transformer encoder layers ($8$ attention heads, feed-forward dimension $512$, dropout $0.1$), and a final linear projection layer. Training runs for up to $100$ epochs using AdamW (learning rate $5\times 10^{-4}$, weight decay $0.001$) with batch size $128$, ReduceLROnPlateau scheduling, and early stopping (patience$ = 5$). Linear layers are initialized with Xavier uniform, and the embedding layer with normal initialization.

        The final model achieves a test accuracy of $0.3011$ and a test loss of $3.6263$. 

    - For Closed data: 
        The same model; however, with dropout $0.05$, AdamW (learning rate $3\times 10^{-4}$, weight decay $0.01$). In this case, we also add a mask for padding tokens to avoid their influence attention.

* `M3-M4_Med_domain_pretrain` includes the code for obtaining matrices with Med-BERT (`M3_Med_BERT.ipynb`) and BioClinicalBERT (`M4_BioclinicalBERT.ipynb`).
* M5-M6_Text_pretrained includes matrices obtained from text-based methods: pretrained general-purpose BERT (`M5_Pretrained_BERT.ipynb`) and Yandex Doc Search (`M6_Yandex_Doc_Search.ipynb`).
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
The prompt is designed to obtain multiple connections per ICD category in a single request, avoiding $O(N^2)$ API calls, which would otherwise exceed three weeks of execution time at prohibitive cost. 

* `M8_Genes_network` includes the code and data provided by the authors of _Goh, Kwang-Il, et al. "The human disease network." Proceedings of the National Academy of Sciences 104.21 (2007): 8685-8690_. The jupyter notebook `paper_2007_graph.ipynb` offers transforming the disease network into graph and mapping disease names into ICD-10 codes.

The experiments are in folders:
* `exp1_Visualization_Correlation` - visualization of disease interconnetions' matrices and Spearman correlations (`plot_heatmaps.ipynb` - for MIMIC; `plot_heatmaps_closed_data.ipynb` - for Closed data).
* `exp2_Graph_based_comparison` - comparison of methods with graphs (`graphs_comparison.ipynb` - for MIMIC; `graphs_comparison_closed_data.ipynb` - for Closed data).

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

## Graph-based comparison of obtained matrices
* **MIMIC-IV dataset**  
![MIMIC-IV](./graph_comparison/mimic/accuracy_mimic_quantiles_025_05_075_09.png)
![MIMIC-IV](./graph_comparison/mimic/precision_mimic_quantiles_025_05_075_09.png)
![MIMIC-IV](./graph_comparison/mimic/recall_mimic_quantiles_025_05_075_09.png)

* **Closed Clinical Data**  
![Closed Clinical Data](./graph_comparison/closed_data/accuracy_closed_data_quantiles_025_05_075_09.png)
![Closed Clinical Data](./graph_comparison/closed_data/precision_closed_data_quantiles_025_05_075_09.png)
![Closed Clinical Data](./graph_comparison/closed_data/recall_closed_data_quantiles_025_05_075_09.png)

## Graph-based comparison of obtained matrices -- results with additional quantiles

* **Closed Clinical Data**  
![Closed Clinical Data](./graph_comparison/closed_data/accuracy_closed_data_quantiles_025_035_05_075_08_085_09_095.png)
![Closed Clinical Data](./graph_comparison/closed_data/precision_closed_data_quantiles_025_035_05_075_08_085_09_095.png)
![Closed Clinical Data](./graph_comparison/closed_data/recall_closed_data_quantiles_025_035_05_075_08_085_09_095.png)
