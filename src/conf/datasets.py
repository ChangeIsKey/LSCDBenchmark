from hydra_zen import builds
from src.dataset import Dataset

DiaWug = builds(
    Dataset, 
    name="diawug", 
    cleaning=None, 
    test_on=None, 
    groupings=["1", "2"], 
    version="1.1.0", 
    pairing=["COMPARE"], 
    sampling=["annotated"],
    urls={
        "1.1.0": "https://zenodo.org/record/5791193/files/diawug.zip",
        "1.0.0": "https://zenodo.org/record/5544554/files/diawug.zip"
    }
)

DwugDe = builds(
    Dataset, 
    name="dwug_de", 
    cleaning=None, 
    test_on=None, 
    groupings=["1", "2"], 
    version="2.1.0", 
    pairing=["COMPARE"], 
    sampling=["annotated"],
    urls={
        "2.1.0": "https://zenodo.org/record/7295410/files/dwug_de.zip",
        "2.0.0": "https://zenodo.org/record/5796871/files/dwug_de.zip",
        "1.1.0": "https://zenodo.org/record/5544198/files/dwug_de.zip",
        "1.0.0": "https://zenodo.org/record/5543724/files/dwug_de.zip"
    }
)


DwugEs = builds(
    Dataset, 
    name="dwug_es", 
    cleaning=None, 
    test_on=None, 
    groupings=["1", "2"], 
    version="4.0.0", 
    pairing=["COMPARE"], 
    sampling=["annotated"],
    urls={
        "4.0.0": "https://zenodo.org/record/6433667/files/dwug_es.zip",
        "3.0.0": "https://zenodo.org/record/6433398/files/dwug_es.zip",
        "2.0.0": "https://zenodo.org/record/6433350/files/dwug_es.zip",
        "1.0.1": "https://zenodo.org/record/6433203/files/dwug_es.zip",
        "1.0.0": "https://zenodo.org/record/6300105/files/dwug_es.zip"
    }
)

