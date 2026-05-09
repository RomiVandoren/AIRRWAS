# AIRRWAS
Analysis pipeline associated with manuscript "T cell-microbiome associations captured through T cell receptor convergence analysis"

![Overview of the AIRRWAS data collection and pipeline](AIRRWAS_framework.png)

---

## Repository Structure

- **0_scripts/**  
  Contains all scripts used for the analysis.  
  - **utils/** within `0_scripts` holds helper functions required by the main scripts.  

- **1_test_raw_data/**  
  For testing purposes, a small dataset of **6 samples** from the discovery cohort is provided.  
  This allows users to test the main parts of the pipeline without requiring access to the full dataset.  
  To run the full pipeline look at the data availability statement for full datasets.
  
- **TCR_microbiome_network.html**  
  The html file containing the full public TCR-microbiome interaction network.
  Download and open the file as a webpage to explore the interactive network.
  
- **AIRRWAS_framework.png**  
  The overview figure of the full AIRRWAS pipeline and the validation experiments.

- **YML files**  
  The Airrwas_env.yml is the environment needed to run all scripts except 2 and 3.
  The Convergence_clustering.yml is an environment needed to perform the clustering and convergence calculations.
   
---

## Data Availability

- **Discovery cohort**: Bulk TCR sequencing data used for the discovery cohort is derived from Ha et al.52 and Affaticati et al.53, and is available at 10.5281/zenodo.7785755. Additionally, 16S rRNA microbiome sequencing data used for the discovery cohort is available at 10.5281/zenodo.19843994.
- **Validation experiments**: Bulk TCR sequencing data generated from the validation experiments have been deposited at Zenodo as 10.5281/zenodo.19843373  

**Independent cohorts used for validation**:  
1. **IBD twins study**:  
   - Brand et al.: Microbiome metagenomics DOI:10.1053/j.gastro.2021.01.030
   - Brand et al.: TCR sequencing data DOI:10.1101/2025.10.31.685913

2. **Colorectal cancer single-cell dataset**:  
   - Pu et al.: TCR sequencing data (manuscript submitted)
   - The colorectal cancer 16S rRNA sequencing data is not publicly available due to contractual and data-use restrictions associated with the original study agreement. Access may be considered upon reasonable request and subject to approval by the data governing body and the industrial partner.

All code and analysis scripts developed for this study are publicly available at:  
[https://github.com/RomiVandoren/AIRRWAS](https://github.com/RomiVandoren/AIRRWAS)  


