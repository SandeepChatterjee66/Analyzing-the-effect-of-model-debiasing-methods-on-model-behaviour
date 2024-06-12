# Enriching-the-out-of-distribution-detection-methods-using-neighbourhood-information

* Datasets can be downloaded from:
1. Entity Classification  
   MedMentions - https://github.com/chanzuckerberg/MedMentions  
   BC5CDR - https://biocreative.bioinformatics.udel.edu/resources/biocreative-v/proceedings-biocreative5/  
   NCBI disease - https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/  
2. NLI  
   MNLI - https://cims.nyu.edu/~sbowman/multinli/  
   SNLI - https://nlp.stanford.edu/projects/snli/  
   HANS - https://github.com/tommccoy1/hans/tree/master/heuristic_finder_scripts  
3. Paraphrase Identification  
   QQP - https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs  
   PAWS - https://github.com/google-research-datasets/paws  


* For viewing the the documentation open the "index.html" file within the "documentation" folder on the browser.


* Within the folders Entity_classification, NLI and Paraphrase_idetification:

   Steps to execute the code:
   1) Generate the embeddings for Vanilla, Learning from Failure, Feature Sieve and Disentagled Feature Augmentation methods -
      * Preprocess the datasets by executing the "run.sh" file in "generate_embeddings/resources/preprocess_datasets". *NOTE*: *This is required only for entity classification task*.
      * Run the "run.sh" script in "generate_embeddings/[Debiasing_method]" folder. This will train the model and generate the prediction files and softmax score files.
      * Run the "embeddings.sh" script in "generate_embeddings/[Debiasing_method]" folder. This will generate the embeddings.
   2) Once the models are trained and embeddings are generated, these embeddings need to be manually copied to the "resources" folder within "Model_analysis" directory.
   3) After copying the embddings, generate the results for Data Distribution by first executing "run_all.sh" in "Model_analysis/Data_distribution" folder. This will detect the OOD points in the dataset. Then we execute "run_iou.sh" in the same folder to get the Jaccard index results.
   4) For getting the results for Data Representation and Error Analysis we need to execute "run.sh" files in the folder "Model_analysis/Data_representation" and "Model_analysis/Error_analysis" repectively.

  * Code execution flow :
   
         generate_embeddings/resources/preprocess_datasets/run.sh(only for Entity classification) -> generate_embeddings/[Debiasing_method]/run.sh -> generate_embeddings/[Debiasing_method]/embeddings.sh -> manually copy the embeddings file in resources folder -> Model_analysis/Data_distribution/run_all.sh -> Model_analysis/Data_distribution/run_iou.sh -> Model_analysis/Data_representation/run.sh -> Model_analysis/Error_analysis/run.sh
