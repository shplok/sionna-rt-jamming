from xml.etree import ElementTree as ET

xml_path = r"/home/luisg-ubuntu/sionna_rt_jamming/data/downtown_chicago_luis/chicagoMariona.xml"
tree = ET.parse(xml_path)
root = tree.getroot()

def to_radio(bsdf_elem, radio_type):
    bsdf_elem.set("type", "itu-radio-material")
    # elimina hijos anteriores
    for c in list(bsdf_elem):
        bsdf_elem.remove(c)
    t = ET.SubElement(bsdf_elem, "string")
    t.set("name", "type")
    t.set("value", radio_type)

for bsdf in root.findall(".//bsdf"):
    t = bsdf.get("type")
    bid = bsdf.get("id", "")
    if t in ("diffuse", "twosided", "conductor", "plastic", "roughconductor"):
        if "wall" in bid:
            to_radio(bsdf, "concrete")
        elif "roof" in bid:
            to_radio(bsdf, "metal")
        else:
            # fallback razonable; cambia a 'glass' si son ventanas
            to_radio(bsdf, "concrete")

tree.write(r"/home/luisg-ubuntu/sionna_rt_jamming/data/downtown_chicago_luis/ChicagoMarionaClean.xml", encoding="utf-8", xml_declaration=True)