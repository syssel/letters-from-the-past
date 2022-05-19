# Letters From the Past: Modeling Historical Sound Change Through Diachronic Character Embeddings

This repository contains the code and data for our ACL 2022 paper:

[Letters From the Past: Modeling Historical Sound Change Through Diachronic Character Embeddings](https://aclanthology.org/2022.acl-long.463)


Please note that part of the [data](##Data) used in this paper is licensed under [CC-BY-4.0](https://choosealicense.com/licenses/cc-by-sa-4.0/).

Also, if you use this code, please cite our paper:

```
@inproceedings{boldsen-paggio-2022-letters,
    title = "Letters From the Past: Modeling Historical Sound Change Through Diachronic Character Embeddings",
    author = "Boldsen, Sidsel  and
      Paggio, Patrizia",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.463",
    pages = "6713--6722",
}
```

## Dependencies


### External libraries
We use the [implementation](https://github.com/connormayer/distributional_learning) by _Mayer, C. (2020) An algorithm for learning phonological classes from distributional information. Phonology 37(1), 91-131_ to generate PPMI embeddings and to generate samples from the artificial language of Parupa:

```bash
# Clone code from "An algorithm for learning phonological classes from distributional similarity"
git clone https://github.com/connormayer/distributional_learning.git external/distributional_learning
```

Further, clone library to simulate sound change:
```
git clone https://github.com/syssel/sound-change-sim.git external/soundchangesim
```
## Code
The main code for running the experiments in the paper can be found in `diachronic-analysis.ipynb`. For documentation for how data was retrieved and treated, refer to the individual README.md in the sub-directories.