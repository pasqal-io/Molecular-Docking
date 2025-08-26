# A Scalable Heuristic for Molecular Docking on Neutral-Atom Quantum Processors

[![arXiv](https://img.shields.io/badge/arXiv-2508.18147-b31b1b.svg)](https://arxiv.org/abs/2508.18147)

This repository contains the official source code and data instances for the paper "A Scalable Heuristic for Molecular Docking on Neutral-Atom Quantum Processors".

### Abstract

> Molecular docking is a critical computational method in drug discovery used to predict the binding conformation and orientation of a ligand within a protein's active site. Mapping this challenge onto a graph-based problem, specifically the Maximum Weighted Independent Set (MWIS) problem, allows it to be addressed by specialized hardware such as neutral-atom quantum processors. However, a significant bottleneck has been the size mismatch between biologically relevant molecular systems and the limited capacity of near-term quantum devices. In this work, we overcome this scaling limitation by the use of a novel divide-and-conquer heuristic. This algorithm enables the solution of large-scale MWIS problems by decomposing a single, intractable graph instance into smaller sub-problems that can be solved sequentially on a neutral-atom quantum emulator, incurring only a linear computational overhead. We demonstrate the power of this approach by solving a 540-node MWIS problem representing the docking of an inhibitor to the Tumor Necrosis Factor-alpha Converting Enzyme (TACE-AS) complex. Our work enables the application of quantum methods to more complex and physically realistic molecular systems than previously possible, paving the way for tackling large-scale docking challenges on near-term quantum hardware.

---

## Repository Structure

The repository is organized as follows:

```
├── data/
│   └── instances/ TACE-AS_COMPL.graphml   # The 540-node graph instance used in the paper
├── notebooks/
│   ├── figures.ipynb # Notebook to generate all figures from the paper
│   └── prepare_instances.ipynb # Notebook detailing the BIG construction
├── src/
│   ├── draw                # Functions plots
│   ├── graph               # Graph utils and BIG construction
│   ├── mol_processing      # Molecular utils
│   └── solver              # Different solvers for MWIS
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project configuration file
└── README.md               # This file
```

---

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management and to ensure a reproducible environment.

1.  **Prerequisite: Install Poetry**
    If you do not have Poetry installed, please follow the [official installation instructions](https://python-poetry.org/docs/#installation).

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/pasqal-io/Molecular-Docking.git
    cd Molecular-Docking
    ```

3.  **Install dependencies:**
    This command will create a dedicated virtual environment for the project and install all the required packages from the `poetry.lock` file.
    ```bash
    poetry install
    ```

---

## Citation

If you use this code or our work in your research, please cite our paper.

### BibTeX

```bibtex
@misc{garrigues2025scalable,
      title={{A Scalable Heuristic for Molecular Docking on Neutral-Atom Quantum Processors}},
      author={Mathieu Garrigues, Victor Onofre, Wesley Coelho and S. Acheche},
      year={2025},
      eprint={2508.18147},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```

---

## License

PASQAL OPEN-SOURCE SOFTWARE LICENSE AGREEMENT (MIT-derived)

The author of the License is:
  Pasqal, a Société par Actions Simplifiée (Simplified Joint Stock Company) registered under number 849 441 522 at the Registre du commerce et des sociétés (Trade and Companies Register) of Evry – France, headquartered at 24 rue Émile Baudot – 91120 – Palaiseau – France, duly represented by its Président, M. Georges-Olivier REYMOND, Hereafter referred to as « the Licensor »

- Permission is hereby granted, free of charge, to any person obtaining a copy of this software (the “Licensee”) and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. The Software is “as is”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise arising from, out of or in connection with the Software or the use or other dealings in the Software.

- If use of the Software leads to the necessary use of any patent of the Licensor and/or any of its Affiliates (defined as a company owned or controlled by the Licensor), the Licensee is granted a royalty-free license, in any country where such patent is in force, to use the object of such patent; or use the process covered by such patent,

- Such a patent license is granted for internal research or academic use of the Licensee's, which includes use by employees and students of the Licensee, acting on behalf of the Licensee, for research purposes only.

- The License is governed by the laws of France. Any dispute relating to the License, notably its execution, performance and/or termination shall be brought to, heard and tried by the Tribunal Judiciaire de Paris, regardless of the rules of jurisdiction in the matter.
