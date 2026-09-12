import os
from xml.etree import ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
xml_path = os.environ.get("CLEAN_XML_INPUT", os.path.join(SCRIPT_DIR, "downtown_chicago_luis", "chicagoMariona.xml"))
out_path = os.environ.get("CLEAN_XML_OUTPUT", os.path.join(SCRIPT_DIR, "downtown_chicago_luis", "ChicagoMarionaClean.xml"))

if not os.path.exists(xml_path):
    print(f"Warning: XML file not found at {xml_path}")
else:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def to_radio(bsdf_elem, radio_type):
        bsdf_elem.set("type", "itu-radio-material")
        # Remove previous children
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
                # Reasonable fallback; change to 'glass' for windows
                to_radio(bsdf, "concrete")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"Cleaned XML saved to: {out_path}")