<h1 align="center">
  Cross-Modality Representation Interpretation<br>
  <sub>for Single-Cell Language Models</sub>
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-%E2%89%A53.12-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python ≥3.12">
  <img src="https://img.shields.io/badge/CUDA-13.0-76B900?style=flat-square&amp;logo=nvidia&amp;logoColor=white" alt="CUDA 13.0">
  <img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.11-EE4C2C?style=flat-square&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch ≥2.11">
  <img src="https://img.shields.io/badge/Model-Cell--to--Sentence-7C3AED?style=flat-square" alt="Model: Cell-to-Sentence">
</p>


Do single-cell language models learn recognizable biological concepts, and do those concepts influence their predictions? This work studies these questions in **Cell-to-Sentence-Scale Gemma-2 (C2S)** models using **sparse autoencoders (SAEs)**. We connect internal model activations to gene associations, cell-type selectivity, and Gene Ontology enrichment, then intervene on the learned features to test their influence on generated cell-type annotations.

## Approach

C2S represents each cell as a sentence of gene names ranked by expression. Our pipeline decomposes the model's dense representations into sparse feature dictionaries that can be inspected and manipulated:

1. **Prepare cell sentences.** Filter and normalize single-cell RNA-sequencing data, rank genes by expression, and construct training, validation, and test datasets.
2. **Extract gene representations.** Pass cell sentences through a frozen C2S model and cache residual-stream activations at selected layers. When a gene name spans multiple tokens, use its final token as the gene-level representation.
3. **Train sparse autoencoders.** Learn an overcomplete dictionary with a TopK encoder and a reconstruction objective, including an auxiliary loss for inactive features.
4. **Evaluate representation fidelity.** Measure reconstruction quality, feature usage, decoder redundancy, and preservation of cell-type annotation accuracy after replacing activations with SAE reconstructions.
5. **Interpret and steer features.** Link features to genes and biological annotations, identify cell-type-selective features, and strengthen or suppress selected decoder directions during inference.

The main experiments study **C2S Gemma-2-2B at layers 10, 15, and 20**, with an eightfold dictionary expansion: **18,432 features and 512 active features per gene representation**. The pipeline also supports C2S Gemma-2-27B, which is included in the manuscript's model configurations.

## Main findings

- **Sparse reconstructions preserve annotation performance.** Replacing gene representations with SAE reconstructions retains cell-type annotation accuracy on both the training-domain dataset and an external PBMC dataset.
- **Depth introduces a trade-off.** Layer 20 provides the strongest aggregate reconstruction, while layer 10 distributes feature usage more evenly. Dictionaries remain broadly utilized, selective, and nearly nonredundant under the evaluated distribution shift.
- **Biological specificity changes across layers.** Deeper features show clearer B-cell and NK-cell associations, including B-cell receptor complexes and NK-cell-mediated immunity. CD14 monocytes are an exception: deeper features emphasize shared antigen-presentation programs, helping diagnose confusion with dendritic cells.
- **Later-layer features steer early layers more effectively.** When applied at an early layer, features learned from later layers produce stronger steering toward target cell-type annotations than features learned from that early layer itself. This suggests that the richer biological concepts captured in later layers can guide predictions more effectively when used to intervene earlier in the model.


## Data and scope

The study focuses on **B cells, CD14 monocytes, and CD56 natural killer cells**. SAEs are trained on purified PBMC data; PBMC-4k and PBMC-8k provide an external evaluation corpus. The gene vocabulary is built from each prepared dataset's training split, with vocabulary checks applied before activation extraction. The findings concern the evaluated immune populations and model predictions. Broader cell populations, finer distinctions between related cell types, and perturbation-response prediction remain directions for future work.

## Repository guide

| Path | Purpose |
| --- | --- |
| [`configs/`](configs/) | YAML settings for data preparation, extraction, SAE training, evaluation, enrichment, and steering |
| [`src/data/`](src/data/) | Single-cell preprocessing, cell-sentence construction, and activation extraction |
| [`src/sae.py`](src/sae.py) | SAE implementations |
| [`src/train.py`](src/train.py) | SAE training on cached activation shards |
| [`src/evaluate/`](src/evaluate/) | Reconstruction metrics, feature analysis, enrichment, and downstream tasks |
| [`src/steer/`](src/steer/) | Feature selection and activation interventions for cell-type annotation |
| [`results/`](results/) | Evaluation outputs and biological analyses |

## Getting started

Use **Python 3.12 or newer**. From the repository root, create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Model extraction and training are configured for CUDA and bfloat16. The manuscript's experiments used an NVIDIA GB10 with CUDA 13.0; the dependency list includes CUDA 13 support for scvi-tools. Configure the model device, precision, and batch size for your hardware.

The workflow is controlled by the YAML files in `configs/`. **Align the dataset paths, model ID, layer, and checkpoint paths before running each stage.**

```bash
# Prepare the purified PBMC dataset using configs/data_prep.yaml.
python -m src.data.data_prep --dataset purified_pbmc

# Extract activations and isolated-gene embeddings for the prepared dataset.
python -m src.data.extraction --model gemma \
  --dataset datasets/c2s_datasets/purified-pbmc-subset

# Train an SAE after aligning configs/train.yaml with the extraction outputs.
python src/train.py

# Evaluate the trained SAE after setting its paths in configs/eval_sae.yaml.
python -m src.evaluate.sae_analysis.eval_sae
```

For the main 2B experiments, set `models.gemma.model_id` in [`configs/extraction.yaml`](configs/extraction.yaml) to `vandijklab/C2S-Scale-Gemma-2-2B`, choose layer 10, 15, or 20, and use gene extraction with last-token pooling and no prompt prefix. Match these choices in [`configs/train.yaml`](configs/train.yaml), including `data.base_model: gemma-2b` and the activation directory. Training enables Weights & Biases by default; set `infrastructure.wandb_enabled: false` to use local CSV logging.

Detailed workflow documentation:

- [Data preparation and activation extraction](src/data/DATA_PIPELINES.md)
- [SAE evaluation modules](src/evaluate/sae_analysis/EVALUATION_MODULES.md)
- [Cell-type annotation and reconstruction settings](src/evaluate/downstream_tasks/cell_type_annotation/SETTINGS.md)
- [Feature steering and evaluation settings](src/steer/cell_type_annotation/SETTINGS.md)

For biological analysis, configure [`configs/gene_features.yaml`](configs/gene_features.yaml), [`configs/cell_analysis.yaml`](configs/cell_analysis.yaml), and [`configs/enrich.yaml`](configs/enrich.yaml) to connect your checkpoints, extracted representations, and feature–gene mappings. The corresponding implementations are in `src/evaluate/`.
