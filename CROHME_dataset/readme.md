# ICDAR CROHME2023: Competition on Recognition of Online Handwritten Mathematical Expressions
## Introduction
Here is the datasets collected for the Competitionon Recognition of Online Handwritten Mathematical Expressions in competition session of ICDAR 2023 [1].  
3 tasks are proposed with different modalities, there are on-line, off-line and bi-modal.  
For on-line task, we provide .inkml file (contain trace information, mathML and LaTeX string), and also symbol level label graph (SymLG) as ground truth. Except the new data and previous CROHME data, we also provide huge amount of artificial on-line data [2] in the train set.   
For off-line task, the .png images (scanned from paper or rendering from inkml) and symbol level label graph (SymLG) are provided. Except the new data and previous CROHME data, we use off-line images from OffHME [3] to increase the size of train set.  
For bi-modal task, both .inkml file and ,png images are provided as 2 channels input, and SymLG as ground truth.  

All the 3 tasks inherited the data collected from the previous 6 CROHME, and also the new collection 2023 in 3 sites, Nantes (France), Luleå (Sweden) and Tokyo (Japan).

## Tools 
CROHMElib (CROHME data converting tools and viewer): https://gitlab.univ-nantes.fr/crohme/crohmelib  
Lgeval (Label Graph Evaluation tools): https://gitlab.com/dprl/lgeval

## Data structure
├── IMG  
│   ├── test  
│   │   ├── CROHME2019_test   
│   │   └── CROHME2023_test  
│   ├── train  
│   │   ├── CROHME2013_train  
│   │   ├── CROHME2019  
│   │   └── OffHME  
│   └── val  
│       ├── CROHME2016_test  
│       └── CROHME2023_val  
├── INKML  
│   ├── test  
│   │   ├── CROHME2019_test  
│   │   └── CROHME2023_test  
│   ├── train  
│   │   ├── Artificial_data  
│   │   │   ├── gen_LaTeX_data_CROHME_2019  
│   │   │   ├── gen_LaTeX_data_CROHME_2023_corpus  
│   │   │   └── gen_syntatic_data  
│   │   ├── CROHME2019  
│   │   └── CROHME2023_train  
│   └── val  
│       ├── CROHME2016_test  
│       └── CROHME2023_val  
└── SymLG  
&ensp;&ensp;    ├── test  
&ensp;&ensp;    │   ├── CROHME2019_test  
&ensp;&ensp;     │   └── CROHME2023_test  
&ensp;&ensp;     ├── train  
&ensp;&ensp;     │   ├── Artificial_data  
&ensp;&ensp;     │   │   ├── gen_LaTeX_data_CROHME_2019  
&ensp;&ensp;     │   │   ├── gen_LaTeX_data_CROHME_2023_corpus  
&ensp;&ensp;     │   │   └── gen_syntactic_data  
&ensp;&ensp;    │   ├── CROHME2019_train  
&ensp;&ensp;     │   ├── CROHME2023_train  
&ensp;&ensp;     │   └── OffHME  
&ensp;&ensp;     └── val  
&ensp;&ensp;         ├── CROHME2016_test  
&ensp;&ensp;         └── CROHME2023_val  

## License and Copyright
These CROHME 2023 (Competition on Recognition of Handwritten Mathematical Expressions 2023 dataset) is Copyright © 2023, Nantes Université / Tokyo University of Agriculture and Technology / Luleå University of Technology. These CROHME 2023 is free software and data; you can redistribute it and/or modify it under the terms of the Creative Commons CC BY-NC-SA 3.0 (Attribution-NonCommercial-ShareAlike 3.0 Unported)

These CROHME+TDF 2019 (Competition on Recognition of Handwritten Mathematical Expressions and Typeset Formula Detection 2019 dataset) is Copyright © 2019, Université de Nantes / Rochester Institute of Technology / ISICAL. These CROHME 2019 is free software and data; you can redistribute it and/or modify it under the terms of the Creative Commons CC BY-NC-SA 3.0 (Attribution-NonCommercial-ShareAlike 3.0 Unported). Note that the used GTDB datasets keep their original license and copyright CC BY-NC-ND.

These CROHME 2016 (Competition on Recognition of Handwritten Mathematical Expressions 2016 dataset) is Copyright © 2016, Université de Nantes / Rochester Institute of Technology. These CROHME 2016 is free software and data; you can redistribute it and/or modify it under the terms of the Creative Commons CC BY-NC-SA 3.0 (Attribution-NonCommercial-ShareAlike 3.0 Unported)

These CROHME 2014 (Competition on Recognition of Handwritten Mathematical Expressions 2014 dataset) is Copyright © 2014, Université de Nantes / Rochester Institute of Technology. These CROHME 2014 is free software and data; you can redistribute it and/or modify it under the terms of the Creative Commons CC BY-NC-SA 3.0 (Attribution-NonCommercial-ShareAlike 3.0 Unported)


The CROHME 2013 (Competition on Recognition of Handwritten Mathematical Expressions 2013 dataset) is Copyright © 2013, Université de Nantes / CNRS. These CROHME 2013 is free software and data; you can redistribute it and/or modify it under the terms of the Creative Commons CC BY-NC-SA 3.0 (Attribution-NonCommercial-ShareAlike 3.0 Unported)

The CROHME 2013 Train set merges several existing data sets which keep their original copyrigths:

    expressmatch: University of Sao Paulo
    MathBrush: University of Waterloo
    KAIST: KAIST lab
    MfrDB: CzechTechnical University
    HAMEX: University of Nantes

The CROHME 2012 (Competition on Recognition of Handwritten Mathematical Expressions 2012 dataset) is Copyright © 11/05/2011, Université de Nantes / CNRS. These CROHME 2012 is free software and data; you can redistribute it and/or modify it under the terms of the Creative Commons CC BY-NC-SA 3.0 (Attribution-NonCommercial-ShareAlike 3.0 Unported)

These data and software are distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Creative Commons License CC BY-NC-SA 3.0 for more details. You should have received a copy of the Creative Commons License along with this program; if not, you can also find the Creative Commons licence on the Creative Commons web site.

Non-free versions of this dataset are available under terms different from those of the Creative Commons. For these alternative terms you must purchase a license from one of the authors' laboratory. Users interested in such a license should contact them for more information. 
## Contact ##
Yejing XIE, PhD Candidate, LS2N, Nantes Université, yejing.xie@univ-nantes.fr  
Harold Mouchère, Professor, LS2N, Nantes Université, harold.mouchere@univ-nantes.fr  
## References
[1] Xie, Yejing, et al. "ICDAR 2023 CROHME: Competition on Recognition of Handwritten Mathematical Expressions." International Conference on Document Analysis and Recognition. Cham: Springer Nature Switzerland, 2023.  
[2] Truong, Thanh-Nghia, Cuong Tuan Nguyen, and Masaki Nakagawa. "Syntactic data generation for handwritten mathematical expression recognition." Pattern Recognition Letters 153 (2022): 83-91.  
[3] Wang, Da-Han, et al. "ICFHR 2020 competition on offline recognition and spotting of handwritten mathematical expressions-OffRaSHME." 2020 17th International Conference on Frontiers in Handwriting Recognition (ICFHR). IEEE, 2020.

