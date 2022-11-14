import uuid
from typing import NewType

import numpy as np
from pandas import Series
from pydantic import BaseModel

UseID = NewType("UseID", str)


class Use(BaseModel):
    # unique id for this specific use
    identifier: UseID
    # grouping id for the specific time period/dialect this use belongs to
    grouping: str
    # string representing one specific context of a word
    # (could be a preprocessed context, or a raw context)
    context: str
    # target word
    target: str
    # span of character indices in which the target word appears in `context`
    indices: tuple[int, int]
    # part-of-speech
    pos: str

    @classmethod
    def from_series(cls, use: Series) -> "Use":
        return cls(
            identifier=use.identifier,
            grouping=use.grouping,
            pos=use.pos,
            context=use.context_preprocessed,
            target=use.lemma.split("_")[0],
            indices=(use.target_index_begin, use.target_index_end),
        )

    def __hash__(self) -> int:
        return hash(self.identifier)

    def __lt__(self, other: "Use") -> bool:
        return self.identifier < other.identifier




# {
#         "id": "test.tropical.1",
#         "start1": 418,
#         "end1": 428,
#         "sentence1": "Despu\u00e9s de verme enaltecido por el respeto y la envidia, amado por quien yo amaba, rico, poderoso, vime herido s\u00fabitamente por la desgracia. Mi decadencia brusca pas\u00f3 ante mis ojos envuelta en humo de incendios, en olas de naufragios, en aliento de traidores, en miradas esquivas de mujer culpable, en alaridos de salvajes sediciosos, en estruendo de calderas de vapor que estallaban, en fragancia mort\u00edfera de flores tropicales, en atm\u00f3sfera espesa de epidemias asi\u00e1ticas, en horribles garabatos de escritura chinesca, en una confusi\u00f3n espantosa de injurias dichas en ingl\u00e9s, en portugu\u00e9s, en espa\u00f1ol, en tagalo, en cipayo, en japon\u00e9s, por bocas blancas, negras, rojas, amarillas, cobrizas y bozales. Ya no quedaba en m\u00ed sino el dejo nauseabundo de una navegaci\u00f3n lenta y triste en buque de vapor cuya h\u00e9lice hab\u00eda golpeado mi cerebro sin cesar d\u00eda tras d\u00eda; solo quedaban en m\u00ed la conciencia de mi ignominia y los dolores f\u00edsicos precursores de un fin desgraciado.",
#         "start2": 491,
#         "end2": 501,
#         "sentence2": "Prospectos de cuatro tintas en que se pintaban figuras altamente conmovedoras, con Hermanas de la Caridad conduciendo mendigos al Asilo; el front\u00f3n mismo del Asilo ideal con columnas griegas y un sol con la insignia triangular de Jehov\u00e1, difund\u00edan por toda la sala la idea de que all\u00ed se trabajaba para aliviar la suerte de los menesterosos. Las palabras Rifas, Grandes rifas, Tres sorteos mensuales, seis millones, impresas en colores, revoloteaban por las paredes cual bandadas de p\u00e1jaros tropicales; y como el papel en que aquellas campeaban era de ramos verdes, la fantas\u00eda loca de Isidora no hab\u00eda de esforzarse mucho para hacer de aquel recinto una especie de selva americana alumbrada por la luna. Despu\u00e9s vio el resto de la casa, que era de construcci\u00f3n reciente, mas con tan s\u00f3rdido aprovechamiento del terreno, que m\u00e1s parec\u00eda madriguera que humana vivienda.",
#         "pos": "NOUN",
#         "grp": "EARLIER",
#         "lemma": "tropical"
#     },
