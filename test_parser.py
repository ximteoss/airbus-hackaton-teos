from collections import Counter
from pathlib import Path
from lxml import etree
from src.ingestion.easa_parser import parse_easa_xml, _get_pstyle

xml_file = Path("data/regulations/easa_part_is.xml")
if not xml_file.exists():
    raise FileNotFoundError(f"Coloca el XML de EASA en {xml_file}")

clauses = parse_easa_xml(xml_file)
print(f"Total cláusulas extraídas: {len(clauses)}")
print("Desglose por nivel (IR / AMC / GM):", Counter(c.level for c in clauses))

print("\nPrimeras 10 cláusulas detectadas:")
for c in clauses[:10]:
    print(f"  - [{c.clause_id}] ({c.level}) {c.title[:60]} | {len(c.text)} chars")

if len(clauses) < 20:
    print("\n[DIAGNÓSTICO] Muestra de párrafos reales con códigos Part-IS:")
    tree = etree.parse(str(xml_file))
    shown = 0
    for el in tree.getroot().iter():
        if isinstance(el.tag, str) and etree.QName(el).localname.lower() == "p":
            txt = " ".join(" ".join(el.itertext()).split())
            if "IS.I.OR." in txt or "IS.D.OR." in txt:
                print(f"  style={_get_pstyle(el)!r} | {txt[:140]!r}")
                shown += 1
                if shown >= 15:
                    break