# Demo image credits

The README demo uses 14 photographs of the ballerina **Anna Pavlova** (1881–1931)
from [Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:Anna_Pavlova).
Every file is marked public domain or CC0 on its Commons page. Commons pages
are the authoritative source for each file's status; check them before reuse.
The original files are not stored in this repository. The GIFs and the contact
sheet in this folder are derived from them (downscaled, cropped and aligned).

| Demo name | Commons file | Author | Date | License |
| --- | --- | --- | --- | --- |
| `pavlova-009.jpg` | [Anna Pavlova (posant avec un chat dans les bras) - btv1b531186910.jpg](https://commons.wikimedia.org/wiki/File:Anna_Pavlova_(posant_avec_un_chat_dans_les_bras)_-_btv1b531186910.jpg) | Agence Rol | 1923 | Public domain |
| `pavlova-014.jpg` | [Anna Pavlova 1926.jpg](https://commons.wikimedia.org/wiki/File:Anna_Pavlova_1926.jpg) | S P Andrew Ltd (d. 1964) | 1926 | Public domain |
| `pavlova-016.jpg` | [Anna Pavlova Cazneaux SLNSW.jpg](https://commons.wikimedia.org/wiki/File:Anna_Pavlova_Cazneaux_SLNSW.jpg) | James Abbe; copy by Harold Cazneaux | 1924 | Public domain |
| `pavlova-017.jpg` | [Anna Pavlova dancing.jpg](https://commons.wikimedia.org/wiki/File:Anna_Pavlova_dancing.jpg) | see Commons page | 1914–1916 | Public domain |
| `pavlova-022.jpg` | [Anna Pavlova, 1885-1931 LCCN2004679212.jpg](https://commons.wikimedia.org/wiki/File:Anna_Pavlova,_1885-1931_LCCN2004679212.jpg) | Library of Congress (PPOC) | 1916 | Public domain |
| `pavlova-026.jpg` | [Anna Pavlowa - Paquita.jpg](https://commons.wikimedia.org/wiki/File:Anna_Pavlowa_-_Paquita.jpg) | Unknown photographer | 1914 | Public domain |
| `pavlova-031.jpg` | [Anna Pawlowa (Anna Pavlova) Russian ballet dancer.jpg](https://commons.wikimedia.org/wiki/File:Anna_Pawlowa_(Anna_Pavlova)_Russian_ballet_dancer.jpg) | Bain News Service | 1910 | CC0 |
| `pavlova-032.jpg` | [Balletdanseres Anna Pavlova (1881-1931), in een libelle kostuum, SFA003003119.jpg](https://commons.wikimedia.org/wiki/File:Balletdanseres_Anna_Pavlova_(1881-1931),_in_een_libelle_kostuum,_SFA003003119.jpg) | unknown | c. 1915–1925 | Public domain |
| `pavlova-034.jpg` | [Gare du Nord - retour de Mme Anna Pavlova - btv1b9047491s.jpg](https://commons.wikimedia.org/wiki/File:Gare_du_Nord_-_retour_de_Mme_Anna_Pavlova_-_btv1b9047491s.jpg) | Agence de presse Meurisse | 1930 | Public domain |
| `pavlova-068.jpg` | [The Dumb Girl of Portici (1916) - 2.jpg](https://commons.wikimedia.org/wiki/File:The_Dumb_Girl_of_Portici_(1916)_-_2.jpg) | Universal Film Manufacturing Company | 1916-06 | Public domain |
| `pavlova-079.jpg` | [Pavlowa LCCN2014714377.jpg](https://commons.wikimedia.org/wiki/File:Pavlowa_LCCN2014714377.jpg) | Bain News Service, publisher | 1900 | Public domain |
| `pavlova-093.jpg` | [Balletdanseres Anna Pavlova in haar tutu staand op haar spitzen op een houten vloer voor een, SFA003003117.jpg](https://commons.wikimedia.org/wiki/File:Balletdanseres_Anna_Pavlova_in_haar_tutu_staand_op_haar_spitzen_op_een_houten_vloer_voor_een,_SFA003003117.jpg) | James Abbe | c. 1915–1925 | Public domain |
| `pavlova-094.jpg` | [Balletdanseres Anna Pavlova met een zwaan op haar schoot zittend op het gazon in haar eigen tuin, SFA003003116.jpg](https://commons.wikimedia.org/wiki/File:Balletdanseres_Anna_Pavlova_met_een_zwaan_op_haar_schoot_zittend_op_het_gazon_in_haar_eigen_tuin,_SFA003003116.jpg) | Lafayette | c. 1915–1925 | Public domain |
| `pavlova-095.jpg` | [Balletdanseres Anna Pavlova met een zwaan zittend in haar eigen tuin, circa 1920, SFA003003115.jpg](https://commons.wikimedia.org/wiki/File:Balletdanseres_Anna_Pavlova_met_een_zwaan_zittend_in_haar_eigen_tuin,_circa_1920,_SFA003003115.jpg) | Lafayette | c. 1920 | Public domain |

## Reproduce

Download the files above into `./pavlova/` (renaming them as in the first
column keeps the same IDs), then:

```sh
posemorph run ./pavlova --out ./out/pavlova --size 540x720 --fps 30 \
  --confidence .2 --alignment head-torso --mirror auto --mode cut --hold 12
posemorph render --out ./out/pavlova --mode morph --hold 8 --transition 10
posemorph report --out ./out/pavlova
```

`before-after.gif` places the inputs in filename order (left) next to
`cut.mp4` (right); `morph.gif` is `morph.mp4`; `contact-sheet.jpg` is the
report's contact sheet. GIFs were downscaled with FFmpeg's palette filters.
